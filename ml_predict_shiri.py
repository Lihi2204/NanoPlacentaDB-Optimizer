#!/usr/bin/env python3
"""
ML Pipeline for NanoPlacentaDB - Predict Shiri's Results
Uses scikit-learn, XGBoost, LightGBM (compatible with Python 3.12)
"""
import pandas as pd
import numpy as np
import warnings, sys, io, os, json
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb

print("=" * 65)
print("  NanoPlacentaDB - ML Pipeline: Predict Shiri's Results")
print("=" * 65)

# ── 1. LOAD & PREPARE DATA ─────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv('DB_enriched.csv', encoding='utf-8')
print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

# Replace missing markers with NaN
df.replace(['not_mentioned', 'none', '', 'NM'], np.nan, inplace=True)

# Identify Shiri's rows
shiri_mask = df['study_id'].str.contains('Katzir', na=False)
shiri_df = df[shiri_mask].copy()
train_df = df[~shiri_mask].copy()
print(f"    Shiri's rows (held out for testing): {len(shiri_df)}")
print(f"    Literature data (training): {len(train_df)}")

# Show Shiri's data
print("\n    Shiri's Gold NPs:")
for _, row in shiri_df.iterrows():
    coat = row.get('surface_coating', '?')
    size = row.get('size_nm', '?')
    print(f"      {coat}: size={size}nm, charge={row.get('surface_charge_cat','?')}, "
          f"zeta={row.get('zeta_potential_mV','?')}mV, route={row.get('exposure_route','?')}")

# ── 2. FEATURE SELECTION ────────────────────────────────────
print("\n[2] Selecting features...")

# Categorical features
CAT_FEATURES = ['core_material', 'material_class', 'shape', 'surface_coating',
                 'surface_charge_cat', 'model_type', 'species', 'exposure_route']

# Numeric features
NUM_FEATURES = ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h']

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

# Targets
TARGET_COLS = {
    'translocation_detected': 'Will the NP cross the placenta?',
    'placental_accumulation': 'Will the NP accumulate in the placenta?',
    'cytotoxicity_observed': 'Is the NP cytotoxic?',
}

# Convert targets to numeric
for col in TARGET_COLS:
    for frame in [train_df, shiri_df]:
        frame[col] = frame[col].map({
            'TRUE': 1, 'FALSE': 0, True: 1, False: 0,
            'true': 1, 'false': 0, 1: 1, 0: 0, '1': 1, '0': 0
        })

# Convert numeric features
for col in NUM_FEATURES:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    shiri_df[col] = pd.to_numeric(shiri_df[col], errors='coerce')

# Encode categorical features
label_encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    # Fit on ALL data (train + Shiri) to handle all categories
    all_vals = pd.concat([train_df[col], shiri_df[col]]).fillna('_missing_')
    le.fit(all_vals)
    train_df[col + '_enc'] = le.transform(train_df[col].fillna('_missing_'))
    shiri_df[col + '_enc'] = le.transform(shiri_df[col].fillna('_missing_'))
    label_encoders[col] = le

ENCODED_CAT = [c + '_enc' for c in CAT_FEATURES]
ML_FEATURES = ENCODED_CAT + NUM_FEATURES

# ── 3. DEFINE MODELS ────────────────────────────────────────
print("\n[3] Defining models to compare...")

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4),
    'XGBoost': xgb.XGBClassifier(n_estimators=150, random_state=42, eval_metric='logloss',
                                   use_label_encoder=False, max_depth=4, learning_rate=0.1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1,
                                     class_weight='balanced', max_depth=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
}

print(f"    {len(MODELS)} models ready")

# ── 4. TRAIN & EVALUATE FOR EACH TARGET ────────────────────
all_results = {}
best_models = {}

for target, question in TARGET_COLS.items():
    print(f"\n{'='*65}")
    print(f"  TARGET: {target}")
    print(f"  Question: {question}")
    print(f"{'='*65}")

    # Prepare data
    mask = train_df[target].notna()
    X = train_df.loc[mask, ML_FEATURES].copy()
    y = train_df.loc[mask, target].astype(int)

    # Impute missing numeric values
    for col in NUM_FEATURES:
        X[col] = X[col].fillna(X[col].median())

    print(f"\n    Data: {len(X)} rows (class 0={sum(y==0)}, class 1={sum(y==1)})")

    if len(X) < 30 or y.nunique() < 2:
        print(f"    SKIP: Insufficient data")
        continue

    # Class ratio
    ratio = y.value_counts().min() / y.value_counts().max()
    print(f"    Class ratio: {ratio:.2f} (1.0 = perfectly balanced)")

    # Cross-validation comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = []

    print(f"\n    {'Model':25s} {'Accuracy':>10s} {'AUC':>8s} {'F1':>8s}")
    print(f"    {'-'*55}")

    for name, model in MODELS.items():
        try:
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            try:
                auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                auc_mean = auc_scores.mean()
            except:
                auc_mean = 0
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

            acc = acc_scores.mean()
            f1 = f1_scores.mean()

            model_scores.append({
                'name': name, 'accuracy': acc, 'auc': auc_mean, 'f1': f1,
                'acc_std': acc_scores.std(), 'model': model
            })

            # Highlight best
            marker = ''
            print(f"    {name:25s} {acc:>9.1%} {auc_mean:>7.3f} {f1:>7.3f}{marker}")

        except Exception as e:
            print(f"    {name:25s} ERROR: {str(e)[:40]}")

    if not model_scores:
        continue

    # Sort by AUC (or accuracy if AUC not available)
    model_scores.sort(key=lambda x: (x['auc'], x['accuracy']), reverse=True)
    best = model_scores[0]

    print(f"\n    BEST MODEL: {best['name']}")
    print(f"    Accuracy: {best['accuracy']:.1%} (+/- {best['acc_std']:.1%})")
    print(f"    AUC: {best['auc']:.3f}")
    print(f"    F1: {best['f1']:.3f}")

    # ── Train best model on all data ────────────────────────
    best_model = best['model']
    best_model.fit(X, y)

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': ML_FEATURES,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n    Feature Importance (what matters most):")
        for _, row in fi.head(10).iterrows():
            fname = row['feature'].replace('_enc', '')
            bar = '#' * int(row['importance'] * 80)
            print(f"      {fname:25s} {row['importance']:>6.3f} {bar}")

    # ── Predict Shiri's results ─────────────────────────────
    print(f"\n    PREDICTING SHIRI'S RESULTS:")
    X_shiri = shiri_df[ML_FEATURES].copy()
    for col in NUM_FEATURES:
        X_shiri[col] = X_shiri[col].fillna(X[col].median())

    if len(X_shiri) > 0:
        y_pred = best_model.predict(X_shiri)
        y_prob = best_model.predict_proba(X_shiri)

        actual_values = shiri_df[target].values

        print(f"    {'Coating':20s} {'Actual':>10s} {'Predicted':>10s} {'Confidence':>12s} {'Match':>6s}")
        print(f"    {'-'*65}")

        for i in range(len(shiri_df)):
            coating = str(shiri_df.iloc[i].get('surface_coating', '?'))
            actual = actual_values[i]
            pred = y_pred[i]
            prob = y_prob[i][1]  # probability of class 1

            actual_str = 'YES' if actual == 1 else ('NO' if actual == 0 else 'N/A')
            pred_str = 'YES' if pred == 1 else 'NO'
            match = 'V' if (not np.isnan(actual) if isinstance(actual, float) else True) and actual == pred else ('?' if pd.isna(actual) else 'X')

            print(f"    {coating:20s} {actual_str:>10s} {pred_str:>10s} {prob:>11.1%} {match:>6s}")

    best_models[target] = {
        'model_name': best['name'],
        'accuracy': best['accuracy'],
        'auc': best['auc'],
        'f1': best['f1'],
        'model_object': best_model,
        'scores_table': model_scores,
    }

    all_results[target] = model_scores

# ── 5. COMPREHENSIVE SUMMARY ───────────────────────────────
print("\n\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)

print(f"\n  Database: {len(df)} records from {df['study_id'].nunique()} studies")
print(f"  Training data: {len(train_df)} rows")
print(f"  Test (Shiri): {len(shiri_df)} rows")

print(f"\n  Best models per target:")
for target, info in best_models.items():
    desc = TARGET_COLS[target]
    print(f"\n    {desc}")
    print(f"    Model: {info['model_name']}")
    print(f"    Accuracy: {info['accuracy']:.1%} | AUC: {info['auc']:.3f} | F1: {info['f1']:.3f}")

print(f"\n  Shiri's Gold NP Research Data:")
print(f"    mPEG-GNP:     P:F = 181, uptake = 101 ug/g")
print(f"    Insulin-GNP:  P:F = 186, uptake = 150 ug/g")
print(f"    Glucose-GNP:  P:F = 360, uptake = 131 ug/g, x23 yolk sac")
print(f"    All show: <1% fetal transfer = excellent placental selectivity")

# ── 6. SAVE ALL MODEL COMPARISON RESULTS ────────────────────
print("\n[4] Saving results...")

save_data = {}
for target, scores in all_results.items():
    save_data[target] = [{
        'model': s['name'],
        'accuracy': round(s['accuracy'], 4),
        'auc': round(s['auc'], 4),
        'f1': round(s['f1'], 4)
    } for s in scores]

with open('ml_results.json', 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print("    Saved ml_results.json")

# ── 7. NEW PREDICTION: What properties make the best NP? ───
print("\n" + "=" * 65)
print("  BONUS: What NP properties are most important?")
print("=" * 65)

for target, info in best_models.items():
    model = info['model_object']
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': [f.replace('_enc', '') for f in ML_FEATURES],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        desc = TARGET_COLS[target]
        print(f"\n  For '{desc}':")
        top3 = fi.head(3)
        for _, r in top3.iterrows():
            print(f"    #{_+1}: {r['feature']} (importance: {r['importance']:.3f})")

print("\n" + "=" * 65)
print("  DONE!")
print("=" * 65)
