#!/usr/bin/env python3
"""
ML-Based Experiment Optimizer for Shiri's Liposome Research
Predicts the optimal combination of:
  - Surface coating
  - Size (nm)
  - Gestational day
  - Dose
  - Targeting ligand
  - Therapeutic payload
For liposome experiments in pregnant mice.
"""
import pandas as pd
import numpy as np
import warnings, sys, io, os, json
from itertools import product
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import (GradientBoostingClassifier, ExtraTreesClassifier,
                               RandomForestClassifier, GradientBoostingRegressor)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

print("=" * 70)
print("  LIPOSOME EXPERIMENT OPTIMIZER")
print("  Finding the best experimental design for Shiri's next experiments")
print("=" * 70)

# ── 1. LOAD & PREPARE ──────────────────────────────────────
print("\n[1] Loading database...")
df = pd.read_csv('DB_enriched.csv', encoding='utf-8')
df.replace(['not_mentioned', 'none', '', 'NM'], np.nan, inplace=True)

# Convert targets
for col in ['translocation_detected', 'placental_accumulation', 'cytotoxicity_observed']:
    df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0,
                            'true': 1, 'false': 0, 1: 1, 0: 0, '1': 1, '0': 0})

# Numeric conversions
for col in ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"    Total records: {len(df)}")

# Show what liposome data we have
lipo_mask = df['core_material'].isin(['liposome', 'LNP', 'lipid_NP'])
lipo_df = df[lipo_mask]
print(f"    Liposome/LNP records: {len(lipo_df)}")
print(f"    In vivo mouse liposome: {len(lipo_df[lipo_df['species'].str.contains('mouse', na=False)])}")

# ── 2. FEATURE SETUP ───────────────────────────────────────
print("\n[2] Preparing features...")

CAT_FEATURES = ['core_material', 'material_class', 'shape', 'surface_coating',
                 'surface_charge_cat', 'model_type', 'species', 'exposure_route',
                 'targeting_ligand', 'therapeutic_payload']
NUM_FEATURES = ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h']
ML_FEATURES = [c + '_enc' for c in CAT_FEATURES] + NUM_FEATURES

# Encode categoricals - fit on ALL data
label_encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    all_vals = df[col].fillna('_missing_')
    # Also add potential new values Shiri might want to try
    extra_vals = []
    if col == 'surface_coating':
        extra_vals = ['PEG', 'DSPE-PEG', 'cholesterol_PEG', 'PEG_COOH',
                      'chitosan', 'folate', 'PEG_NH2', 'hyaluronic_acid']
    elif col == 'targeting_ligand':
        extra_vals = ['CGKRK', 'iRGD', 'folate', 'transferrin', 'RGD',
                      'NKGLRNK', 'RSGVAKS', 'peptide', 'antibody', 'none']
    elif col == 'therapeutic_payload':
        extra_vals = ['none', 'anakinra', 'dexamethasone', 'siRNA', 'mRNA',
                      'IGF1', 'curcumin', 'heparin', 'aspirin']
    elif col == 'gestational_stage':
        extra_vals = [f'GD{d}' for d in range(8, 19)]

    combined = pd.concat([all_vals, pd.Series(extra_vals)])
    le.fit(combined)
    df[col + '_enc'] = le.transform(df[col].fillna('_missing_'))
    label_encoders[col] = le

# ── 3. TRAIN MODELS ON ALL DATA ────────────────────────────
print("\n[3] Training prediction models on ALL literature data...")

TARGETS = {
    'translocation_detected': ('classification', 'MINIMIZE fetal exposure'),
    'placental_accumulation': ('classification', 'MAXIMIZE placental retention'),
    'cytotoxicity_observed': ('classification', 'MINIMIZE toxicity'),
}

trained_models = {}

for target, (task, goal) in TARGETS.items():
    mask = df[target].notna()
    X = df.loc[mask, ML_FEATURES].copy()
    y = df.loc[mask, target].astype(int)

    # Impute missing numerics
    for col in NUM_FEATURES:
        X[col] = X[col].fillna(X[col].median())

    n0, n1 = sum(y == 0), sum(y == 1)
    print(f"\n    {target}: {len(X)} samples (0={n0}, 1={n1})")
    print(f"    Goal: {goal}")

    if len(X) < 30:
        continue

    # Train ensemble of best models
    models = [
        ('GBM', GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)),
        ('XGB', xgb.XGBClassifier(n_estimators=150, max_depth=4, random_state=42,
                                    eval_metric='logloss', use_label_encoder=False)),
        ('ET', ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ('RF', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ('LGBM', lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1,
                                      class_weight='balanced', max_depth=5)),
    ]

    best_score = 0
    best_model = None
    best_name = ''

    for name, model in models:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            if auc > best_score:
                best_score = auc
                best_model = model
                best_name = name
        except:
            pass

    best_model.fit(X, y)
    trained_models[target] = {
        'model': best_model,
        'name': best_name,
        'auc': best_score,
        'medians': {col: X[col].median() for col in NUM_FEATURES}
    }
    print(f"    Best: {best_name} (AUC={best_score:.3f})")

# ── 4. GENERATE CANDIDATE EXPERIMENTS ──────────────────────
print("\n[4] Generating candidate experimental designs...")

# Define the experimental parameter space for liposomes in mice
PARAM_SPACE = {
    'core_material': ['liposome'],
    'material_class': ['lipid_based'],
    'shape': ['spherical'],
    'surface_coating': ['PEG', 'DSPE-PEG', 'cholesterol_PEG', 'PEG_COOH',
                         'chitosan', 'folate', 'hyaluronic_acid', 'PEG_NH2'],
    'surface_charge_cat': ['negative', 'neutral', 'positive'],
    'model_type': ['in_vivo_mouse'],
    'species': ['mouse'],
    'exposure_route': ['IV'],
    'targeting_ligand': ['none', 'CGKRK', 'iRGD', 'folate', 'NKGLRNK',
                          'RSGVAKS', 'RGD', 'transferrin', 'peptide'],
    'therapeutic_payload': ['none', 'anakinra', 'dexamethasone', 'siRNA',
                             'mRNA', 'curcumin'],
    'size_nm': [50, 80, 100, 120, 150, 200, 300],
    'zeta_potential_mV': [-30, -15, -5, 0, 5, 15],
    'dose_value': [1, 5, 10, 20],  # mg/kg
    'exposure_duration_h': [2, 6, 24],
}

# Gestational days to test
GD_OPTIONS = [10, 11, 12, 13, 14, 14.5, 15, 16, 17, 18]

# Generate smart combinations (not full cartesian - too many)
# Focus on most impactful parameters
print("    Generating targeted combinations...")

candidates = []
for coating in PARAM_SPACE['surface_coating']:
    for ligand in PARAM_SPACE['targeting_ligand']:
        for size in PARAM_SPACE['size_nm']:
            for charge in PARAM_SPACE['surface_charge_cat']:
                for gd in GD_OPTIONS:
                    for payload in ['none', 'anakinra', 'dexamethasone']:
                        for dose in [5, 10]:
                            for zeta in [-15, 0]:
                                candidates.append({
                                    'core_material': 'liposome',
                                    'material_class': 'lipid_based',
                                    'shape': 'spherical',
                                    'surface_coating': coating,
                                    'surface_charge_cat': charge,
                                    'model_type': 'in_vivo_mouse',
                                    'species': 'mouse',
                                    'exposure_route': 'IV',
                                    'targeting_ligand': ligand,
                                    'therapeutic_payload': payload,
                                    'size_nm': size,
                                    'zeta_potential_mV': zeta,
                                    'dose_value': dose,
                                    'exposure_duration_h': 24,
                                    'gestational_day': gd,
                                })

print(f"    Generated {len(candidates)} candidate experiments")

# ── 5. PREDICT OUTCOMES FOR ALL CANDIDATES ──────────────────
print("\n[5] Predicting outcomes for all candidates...")

cand_df = pd.DataFrame(candidates)

# Encode
for col in CAT_FEATURES:
    le = label_encoders[col]
    vals = cand_df[col].fillna('_missing_')
    # Handle unseen values
    known = set(le.classes_)
    vals = vals.apply(lambda x: x if x in known else '_missing_')
    cand_df[col + '_enc'] = le.transform(vals)

X_cand = cand_df[ML_FEATURES].copy()

# Predict each target
for target, info in trained_models.items():
    model = info['model']
    probs = model.predict_proba(X_cand)[:, 1]
    cand_df[f'{target}_prob'] = probs

# ── 6. SCORE & RANK ────────────────────────────────────────
print("\n[6] Scoring and ranking experiments...")

# Composite score:
# MAXIMIZE: placental_accumulation_prob (we want high accumulation)
# MINIMIZE: translocation_detected_prob (we want low fetal transfer)
# MINIMIZE: cytotoxicity_observed_prob (we want no toxicity)

# Score = accumulation_prob - translocation_prob - cytotoxicity_prob
# Higher is better

if 'placental_accumulation_prob' in cand_df.columns:
    accum_score = cand_df['placental_accumulation_prob']
else:
    accum_score = 0.5

if 'translocation_detected_prob' in cand_df.columns:
    trans_score = cand_df['translocation_detected_prob']
else:
    trans_score = 0.5

if 'cytotoxicity_observed_prob' in cand_df.columns:
    tox_score = cand_df['cytotoxicity_observed_prob']
else:
    tox_score = 0.5

# Weighted composite score
cand_df['composite_score'] = (
    accum_score * 2.0        # Weight: accumulation is most important
    - trans_score * 1.5       # Penalty: fetal transfer is dangerous
    - tox_score * 1.0         # Penalty: toxicity is bad
)

# Sort by composite score
cand_df = cand_df.sort_values('composite_score', ascending=False)

# ── 7. DISPLAY TOP RESULTS ─────────────────────────────────
print("\n" + "=" * 70)
print("  TOP 20 RECOMMENDED EXPERIMENTAL DESIGNS FOR LIPOSOMES")
print("  (Optimized for: high placental accumulation, low fetal transfer,")
print("   low toxicity - in pregnant mice, IV injection)")
print("=" * 70)

display_cols = ['surface_coating', 'targeting_ligand', 'therapeutic_payload',
                'size_nm', 'surface_charge_cat', 'zeta_potential_mV',
                'dose_value', 'gestational_day']

prob_cols = [c for c in cand_df.columns if c.endswith('_prob')]

top20 = cand_df.head(20)
for rank, (_, row) in enumerate(top20.iterrows(), 1):
    print(f"\n  #{rank}  Score: {row['composite_score']:.3f}")
    print(f"  {'─'*55}")
    print(f"    Coating:     {row['surface_coating']}")
    print(f"    Ligand:      {row['targeting_ligand']}")
    print(f"    Payload:     {row['therapeutic_payload']}")
    print(f"    Size:        {row['size_nm']} nm")
    print(f"    Charge:      {row['surface_charge_cat']} (zeta={row['zeta_potential_mV']}mV)")
    print(f"    Dose:        {row['dose_value']} mg/kg IV")
    print(f"    GD:          {row['gestational_day']}")
    if 'placental_accumulation_prob' in row:
        print(f"    >> Placental accumulation probability:  {row['placental_accumulation_prob']:.1%}")
    if 'translocation_detected_prob' in row:
        print(f"    >> Fetal transfer probability:          {row['translocation_detected_prob']:.1%}")
    if 'cytotoxicity_observed_prob' in row:
        print(f"    >> Cytotoxicity probability:            {row['cytotoxicity_observed_prob']:.1%}")

# ── 8. ANALYSIS BY PARAMETER ───────────────────────────────
print("\n\n" + "=" * 70)
print("  PARAMETER ANALYSIS: What matters most?")
print("=" * 70)

# Best coating
print("\n  BEST COATINGS (avg score):")
coat_scores = cand_df.groupby('surface_coating')['composite_score'].mean().sort_values(ascending=False)
for coat, score in coat_scores.items():
    bar = '#' * int((score + 1) * 20)
    print(f"    {coat:25s} {score:+.3f} {bar}")

# Best targeting ligand
print("\n  BEST TARGETING LIGANDS (avg score):")
lig_scores = cand_df.groupby('targeting_ligand')['composite_score'].mean().sort_values(ascending=False)
for lig, score in lig_scores.items():
    bar = '#' * int((score + 1) * 20)
    print(f"    {lig:25s} {score:+.3f} {bar}")

# Best size
print("\n  BEST SIZES (avg score):")
size_scores = cand_df.groupby('size_nm')['composite_score'].mean().sort_values(ascending=False)
for size, score in size_scores.items():
    bar = '#' * int((score + 1) * 20)
    print(f"    {size:>6.0f} nm               {score:+.3f} {bar}")

# Best GD
print("\n  BEST GESTATIONAL DAYS (avg score):")
gd_scores = cand_df.groupby('gestational_day')['composite_score'].mean().sort_values(ascending=False)
for gd, score in gd_scores.items():
    bar = '#' * int((score + 1) * 20)
    print(f"    GD {gd:<5}                {score:+.3f} {bar}")

# Best charge
print("\n  BEST CHARGE (avg score):")
charge_scores = cand_df.groupby('surface_charge_cat')['composite_score'].mean().sort_values(ascending=False)
for ch, score in charge_scores.items():
    bar = '#' * int((score + 1) * 20)
    print(f"    {ch:25s} {score:+.3f} {bar}")

# Best payload
print("\n  BEST PAYLOAD (avg score):")
pay_scores = cand_df.groupby('therapeutic_payload')['composite_score'].mean().sort_values(ascending=False)
for pay, score in pay_scores.items():
    bar = '#' * int((score + 1) * 20)
    print(f"    {pay:25s} {score:+.3f} {bar}")

# ── 9. COMPARISON WITH SHIRI'S GNP RESULTS ─────────────────
print("\n\n" + "=" * 70)
print("  COMPARISON: Best Liposome Design vs Shiri's GNP Results")
print("=" * 70)

best_lipo = cand_df.iloc[0]
print(f"""
  Shiri's Best GNP (Glucose-GNP):
    Material:   Au (gold) + mPEG + Glucose ligand
    P:F ratio:  360
    Uptake:     131 ug/g
    Fetal:      <1% transfer
    Route:      IV at GD14.5 in mice

  Recommended Best Liposome:
    Coating:    {best_lipo['surface_coating']}
    Ligand:     {best_lipo['targeting_ligand']}
    Payload:    {best_lipo['therapeutic_payload']}
    Size:       {best_lipo['size_nm']:.0f} nm
    Charge:     {best_lipo['surface_charge_cat']} ({best_lipo['zeta_potential_mV']:.0f} mV)
    Dose:       {best_lipo['dose_value']:.0f} mg/kg IV
    GD:         {best_lipo['gestational_day']}
    Predicted:  Accumulation {best_lipo.get('placental_accumulation_prob', 0):.1%},
                Fetal transfer {best_lipo.get('translocation_detected_prob', 0):.1%},
                Toxicity {best_lipo.get('cytotoxicity_observed_prob', 0):.1%}
""")

# ── 10. SAVE RESULTS ───────────────────────────────────────
print("[7] Saving results...")

# Save top 100 designs
top100 = cand_df.head(100)
save_cols = display_cols + prob_cols + ['composite_score']
top100[save_cols].to_csv('optimal_experiments.csv', index=False, encoding='utf-8')
print(f"    Saved top 100 designs to optimal_experiments.csv")

# Save full analysis
analysis = {
    'best_coating': coat_scores.to_dict(),
    'best_ligand': lig_scores.to_dict(),
    'best_size': {str(k): v for k, v in size_scores.to_dict().items()},
    'best_gd': {str(k): v for k, v in gd_scores.to_dict().items()},
    'best_charge': charge_scores.to_dict(),
    'best_payload': pay_scores.to_dict(),
    'top_design': {
        'coating': best_lipo['surface_coating'],
        'ligand': best_lipo['targeting_ligand'],
        'payload': best_lipo['therapeutic_payload'],
        'size_nm': float(best_lipo['size_nm']),
        'charge': best_lipo['surface_charge_cat'],
        'zeta_mV': float(best_lipo['zeta_potential_mV']),
        'dose_mg_kg': float(best_lipo['dose_value']),
        'gestational_day': float(best_lipo['gestational_day']),
    },
    'model_performance': {t: {'model': i['name'], 'auc': round(i['auc'], 3)}
                          for t, i in trained_models.items()},
}

with open('experiment_optimization.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print("    Saved experiment_optimization.json")

print("\n" + "=" * 70)
print("  DONE! Check optimal_experiments.csv for the full ranked list.")
print("=" * 70)
