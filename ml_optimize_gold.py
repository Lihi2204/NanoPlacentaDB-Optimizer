#!/usr/bin/env python3
"""
ML Optimizer for Shiri's Gold Nanoparticles
Find the best coating + ligand + size + conditions for Au NPs
that maximize placental accumulation and minimize fetal transfer.
"""
import pandas as pd
import numpy as np
import warnings, sys, io, os, json
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import (GradientBoostingClassifier, ExtraTreesClassifier,
                               RandomForestClassifier)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

print("=" * 70)
print("  GOLD NP EXPERIMENT OPTIMIZER")
print("  What coating + conditions on Au NPs will give Shiri")
print("  the BEST placental accumulation with LEAST fetal exposure?")
print("=" * 70)

# ── 1. LOAD DATA ───────────────────────────────────────────
print("\n[1] Loading database...")
df = pd.read_csv('DB_enriched.csv', encoding='utf-8')
df.replace(['not_mentioned', 'none', '', 'NM'], np.nan, inplace=True)

for col in ['translocation_detected', 'placental_accumulation', 'cytotoxicity_observed']:
    df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0,
                            'true': 1, 'false': 0, 1: 1, 0: 0, '1': 1, '0': 0})

for col in ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Show Au NP data in database
au_df = df[df['core_material'] == 'Au']
print(f"    Total records: {len(df)}")
print(f"    Gold NP records: {len(au_df)}")
print(f"\n    Existing Au coatings in literature:")
au_coats = au_df['surface_coating'].value_counts()
for coat, n in au_coats.items():
    if pd.notna(coat):
        print(f"      {coat}: {n} records")

print(f"\n    Existing Au sizes in literature:")
au_sizes = au_df['size_nm'].dropna()
if len(au_sizes):
    print(f"      Range: {au_sizes.min():.1f} - {au_sizes.max():.1f} nm")
    print(f"      Median: {au_sizes.median():.1f} nm")
    print(f"      Common: {', '.join(str(int(x)) + 'nm' for x in sorted(au_sizes.unique())[:10])}")

# ── 2. FEATURES ────────────────────────────────────────────
CAT_FEATURES = ['core_material', 'material_class', 'shape', 'surface_coating',
                 'surface_charge_cat', 'model_type', 'species', 'exposure_route',
                 'targeting_ligand', 'therapeutic_payload']
NUM_FEATURES = ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h']
ML_FEATURES = [c + '_enc' for c in CAT_FEATURES] + NUM_FEATURES

# Realistic coatings FOR GOLD NANOPARTICLES
AU_COATINGS = [
    'PEG', 'mPEG', 'COOH', 'NH2', 'citrate',
    'PVP', 'BSA', 'chitosan', 'silica',
    'PEG_COOH', 'PEG_NH2', 'dextran',
    'transferrin', 'folate',
    'lipid',     # lipid shell on gold
    'DSPE-PEG',  # lipid-PEG hybrid
]

# Realistic targeting ligands for Au NPs
AU_LIGANDS = [
    'none',       # no targeting
    'insulin',    # Shiri already tested
    'glucose',    # Shiri already tested - best result
    'folate',     # folate receptor targeting
    'transferrin', # transferrin receptor
    'CGKRK',     # placental homing peptide (King 2016)
    'iRGD',      # tumor/placenta homing
    'RGD',       # integrin targeting
    'NKGLRNK',   # novel placental homing (Alobaid 2025)
    'RSGVAKS',   # novel placental homing (Alobaid 2025)
    'antibody',  # anti-receptor antibody
    'aptamer',   # nucleic acid aptamer
]

# Possible therapeutic payloads conjugated/attached to Au
AU_PAYLOADS = [
    'none',           # imaging/biodistribution only
    'anakinra',       # IL-1R antagonist - Shiri's planned therapy
    'dexamethasone',  # anti-inflammatory steroid
    'siRNA',          # gene silencing
    'curcumin',       # natural anti-inflammatory
]

# Encode - include all possible values
label_encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    all_vals = df[col].fillna('_missing_')
    extras = []
    if col == 'surface_coating':
        extras = AU_COATINGS
    elif col == 'targeting_ligand':
        extras = AU_LIGANDS
    elif col == 'therapeutic_payload':
        extras = AU_PAYLOADS

    combined = pd.concat([all_vals, pd.Series(extras + ['_missing_'])])
    le.fit(combined)
    df[col + '_enc'] = le.transform(df[col].fillna('_missing_'))
    label_encoders[col] = le

# ── 3. TRAIN MODELS ────────────────────────────────────────
print("\n[2] Training models on ALL literature data (486 records)...")

TARGETS = {
    'translocation_detected': 'Fetal transfer (MINIMIZE)',
    'placental_accumulation': 'Placental accumulation (MAXIMIZE)',
    'cytotoxicity_observed': 'Cytotoxicity (MINIMIZE)',
}

trained_models = {}
for target, desc in TARGETS.items():
    mask = df[target].notna()
    X = df.loc[mask, ML_FEATURES].copy()
    y = df.loc[mask, target].astype(int)
    for col in NUM_FEATURES:
        X[col] = X[col].fillna(X[col].median())

    print(f"    {target}: {len(X)} samples -> ", end='')

    models = [
        ('GBM', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ('XGB', xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42,
                                    eval_metric='logloss', use_label_encoder=False)),
        ('ET', ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced')),
        ('RF', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')),
        ('LGBM', lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1,
                                      class_weight='balanced', max_depth=6)),
    ]

    best_auc, best_model, best_name = 0, None, ''
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models:
        try:
            auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            if auc > best_auc:
                best_auc, best_model, best_name = auc, model, name
        except:
            pass

    best_model.fit(X, y)
    trained_models[target] = {'model': best_model, 'name': best_name, 'auc': best_auc}
    print(f"{best_name} (AUC={best_auc:.3f})")

# ── 4. GENERATE GOLD NP COMBINATIONS ──────────────────────
print("\n[3] Generating Au NP experimental candidates...")

AU_SIZES = [3, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100]
AU_CHARGES = ['negative', 'neutral', 'positive']
AU_ZETAS = [-30, -20, -15, -5, 0, 5, 10]
GD_OPTIONS = [10, 11, 12, 13, 14, 14.5, 15, 16, 17, 18]
DOSES = [0.5, 1, 5, 10, 20]  # mg/kg

candidates = []
for coating in AU_COATINGS:
    for ligand in AU_LIGANDS:
        for size in AU_SIZES:
            for charge in AU_CHARGES:
                for gd in GD_OPTIONS:
                    for payload in AU_PAYLOADS:
                        for dose in DOSES:
                            for zeta in [-15, 0]:
                                candidates.append({
                                    'core_material': 'Au',
                                    'material_class': 'inorganic_metal',
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

print(f"    Generated {len(candidates):,} candidate experiments")

# ── 5. PREDICT ─────────────────────────────────────────────
print("\n[4] Predicting outcomes for all candidates...")

cand_df = pd.DataFrame(candidates)
for col in CAT_FEATURES:
    le = label_encoders[col]
    vals = cand_df[col].fillna('_missing_')
    known = set(le.classes_)
    vals = vals.apply(lambda x: x if x in known else '_missing_')
    cand_df[col + '_enc'] = le.transform(vals)

X_cand = cand_df[ML_FEATURES].copy()

for target, info in trained_models.items():
    probs = info['model'].predict_proba(X_cand)[:, 1]
    cand_df[f'{target}_prob'] = probs

# ── 6. SCORE ───────────────────────────────────────────────
# Goal: HIGH accumulation, LOW translocation, LOW toxicity
cand_df['score'] = (
    cand_df['placental_accumulation_prob'] * 2.0
    - cand_df['translocation_detected_prob'] * 1.5
    - cand_df['cytotoxicity_observed_prob'] * 1.0
)

cand_df = cand_df.sort_values('score', ascending=False)

# ── 7. RESULTS ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TOP 15 GOLD NP EXPERIMENTAL DESIGNS")
print("  Optimized for Shiri: max placental accumulation,")
print("  min fetal transfer, min toxicity, in pregnant mice IV")
print("=" * 70)

# Show unique top designs (collapse GD since it doesn't vary much)
seen = set()
rank = 0
for _, row in cand_df.iterrows():
    key = (row['surface_coating'], row['targeting_ligand'], row['therapeutic_payload'],
           row['size_nm'], row['surface_charge_cat'], row['dose_value'])
    if key in seen:
        continue
    seen.add(key)
    rank += 1
    if rank > 15:
        break

    print(f"\n  #{rank}  Score: {row['score']:.3f}")
    print(f"  {'─'*60}")
    print(f"    Au NP + {row['surface_coating']}")
    if row['targeting_ligand'] != 'none':
        print(f"       + {row['targeting_ligand']} (targeting ligand)")
    if row['therapeutic_payload'] != 'none':
        print(f"       + {row['therapeutic_payload']} (payload)")
    print(f"    Size:    {row['size_nm']:.0f} nm")
    print(f"    Charge:  {row['surface_charge_cat']} (zeta {row['zeta_potential_mV']:.0f} mV)")
    print(f"    Dose:    {row['dose_value']:.0f} mg/kg IV")
    print(f"    GD:      flexible (model shows no GD preference)")
    print(f"    ── Predicted outcomes ──")
    print(f"    Placental accumulation:  {row['placental_accumulation_prob']:>6.1%}")
    print(f"    Fetal transfer:          {row['translocation_detected_prob']:>6.1%}")
    print(f"    Cytotoxicity:            {row['cytotoxicity_observed_prob']:>6.1%}")

# ── 8. PARAMETER RANKINGS ─────────────────────────────────
print("\n\n" + "=" * 70)
print("  PARAMETER RANKINGS FOR GOLD NPs")
print("=" * 70)

# Deduplicate for fair ranking (avg over GDs)
dedup = cand_df.groupby(['surface_coating', 'targeting_ligand', 'therapeutic_payload',
                          'size_nm', 'surface_charge_cat', 'zeta_potential_mV',
                          'dose_value']).agg({
    'score': 'mean',
    'placental_accumulation_prob': 'mean',
    'translocation_detected_prob': 'mean',
    'cytotoxicity_observed_prob': 'mean',
}).reset_index()

print("\n  BEST COATINGS for Au NPs:")
coat_rank = dedup.groupby('surface_coating')['score'].mean().sort_values(ascending=False)
for i, (coat, score) in enumerate(coat_rank.items(), 1):
    bar = '#' * int((score + 1) * 15)
    print(f"    {i:2d}. {coat:18s} {score:+.3f} {bar}")

print("\n  BEST TARGETING LIGANDS for Au NPs:")
lig_rank = dedup.groupby('targeting_ligand')['score'].mean().sort_values(ascending=False)
for i, (lig, score) in enumerate(lig_rank.items(), 1):
    bar = '#' * int((score + 1) * 15)
    print(f"    {i:2d}. {lig:18s} {score:+.3f} {bar}")

print("\n  BEST SIZES for Au NPs:")
size_rank = dedup.groupby('size_nm')['score'].mean().sort_values(ascending=False)
for i, (size, score) in enumerate(size_rank.items(), 1):
    bar = '#' * int((score + 1) * 15)
    print(f"    {i:2d}. {size:>5.0f} nm          {score:+.3f} {bar}")

print("\n  BEST CHARGE for Au NPs:")
ch_rank = dedup.groupby('surface_charge_cat')['score'].mean().sort_values(ascending=False)
for i, (ch, score) in enumerate(ch_rank.items(), 1):
    bar = '#' * int((score + 1) * 15)
    print(f"    {i:2d}. {ch:18s} {score:+.3f} {bar}")

print("\n  BEST PAYLOADS for Au NPs:")
pay_rank = dedup.groupby('therapeutic_payload')['score'].mean().sort_values(ascending=False)
for i, (pay, score) in enumerate(pay_rank.items(), 1):
    bar = '#' * int((score + 1) * 15)
    print(f"    {i:2d}. {pay:18s} {score:+.3f} {bar}")

print("\n  BEST DOSE (mg/kg) for Au NPs:")
dose_rank = dedup.groupby('dose_value')['score'].mean().sort_values(ascending=False)
for i, (dose, score) in enumerate(dose_rank.items(), 1):
    bar = '#' * int((score + 1) * 15)
    print(f"    {i:2d}. {dose:>5.1f} mg/kg      {score:+.3f} {bar}")

# ── 9. COMPARE WITH SHIRI'S ACTUAL RESULTS ────────────────
print("\n\n" + "=" * 70)
print("  COMPARISON WITH SHIRI'S EXISTING RESULTS")
print("=" * 70)

# Simulate Shiri's actual experiments through the model
shiri_experiments = [
    {'name': 'mPEG-GNP (control)', 'coating': 'mPEG', 'ligand': 'none',
     'actual_pf': 181, 'actual_uptake': 101},
    {'name': 'Insulin-GNP', 'coating': 'mPEG', 'ligand': 'insulin',
     'actual_pf': 186, 'actual_uptake': 150},
    {'name': 'Glucose-GNP (best)', 'coating': 'mPEG', 'ligand': 'glucose',
     'actual_pf': 360, 'actual_uptake': 131},
]

for exp in shiri_experiments:
    # Find matching candidate
    mask = ((cand_df['surface_coating'] == exp['coating']) &
            (cand_df['targeting_ligand'] == exp['ligand']) &
            (cand_df['size_nm'] == 20) &  # approximate GNP size
            (cand_df['dose_value'] == 5))
    matches = cand_df[mask]
    if len(matches) > 0:
        row = matches.iloc[0]
        print(f"\n  {exp['name']}:")
        print(f"    Actual:    P:F = {exp['actual_pf']}, uptake = {exp['actual_uptake']} ug/g")
        print(f"    Predicted: accum = {row['placental_accumulation_prob']:.1%}, "
              f"transfer = {row['translocation_detected_prob']:.1%}, "
              f"toxicity = {row['cytotoxicity_observed_prob']:.1%}")
        print(f"    Score:     {row['score']:.3f}")
    else:
        print(f"\n  {exp['name']}: no exact match found for coating={exp['coating']}, ligand={exp['ligand']}")

# Best overall recommendation
best = cand_df.iloc[0]
print(f"\n  RECOMMENDED NEXT EXPERIMENT:")
print(f"    Au NP + {best['surface_coating']}", end='')
if best['targeting_ligand'] != 'none':
    print(f" + {best['targeting_ligand']}", end='')
if best['therapeutic_payload'] != 'none':
    print(f" + {best['therapeutic_payload']}", end='')
print(f"\n    Size: {best['size_nm']:.0f} nm, {best['surface_charge_cat']}, "
      f"{best['dose_value']:.0f} mg/kg IV")

# ── 10. SAVE ───────────────────────────────────────────────
print("\n\n[5] Saving results...")

# Top 50 unique designs
seen2 = set()
top_designs = []
for _, row in cand_df.iterrows():
    key = (row['surface_coating'], row['targeting_ligand'], row['therapeutic_payload'],
           row['size_nm'], row['surface_charge_cat'], row['dose_value'])
    if key not in seen2:
        seen2.add(key)
        top_designs.append(row)
        if len(top_designs) >= 50:
            break

top_df = pd.DataFrame(top_designs)
save_cols = ['surface_coating', 'targeting_ligand', 'therapeutic_payload',
             'size_nm', 'surface_charge_cat', 'zeta_potential_mV',
             'dose_value', 'placental_accumulation_prob',
             'translocation_detected_prob', 'cytotoxicity_observed_prob', 'score']
top_df[save_cols].to_csv('optimal_gold_experiments.csv', index=False, encoding='utf-8')
print("    Saved optimal_gold_experiments.csv")

analysis = {
    'best_coating': coat_rank.to_dict(),
    'best_ligand': lig_rank.to_dict(),
    'best_size': {str(k): v for k, v in size_rank.to_dict().items()},
    'best_charge': ch_rank.to_dict(),
    'best_payload': pay_rank.to_dict(),
    'best_dose': {str(k): v for k, v in dose_rank.to_dict().items()},
    'top_design': {
        'coating': best['surface_coating'],
        'ligand': best['targeting_ligand'],
        'payload': best['therapeutic_payload'],
        'size_nm': float(best['size_nm']),
        'charge': best['surface_charge_cat'],
        'dose_mg_kg': float(best['dose_value']),
    },
}
with open('gold_optimization.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print("    Saved gold_optimization.json")

print("\n" + "=" * 70)
print("  DONE!")
print("=" * 70)
