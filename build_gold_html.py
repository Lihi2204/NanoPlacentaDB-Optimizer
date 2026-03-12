#!/usr/bin/env python3
"""
Build interactive Gold NP Optimizer HTML for Shiri
v2 - Fixed: heatmap colors, combination diversity, sensitivity analysis
"""
import pandas as pd, numpy as np, json, re, sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

print("=" * 60)
print("  Building Gold NP Optimizer HTML v2")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# PART 1: DATA LOADING
# ═══════════════════════════════════════════════════════════
print("\n[1/7] Loading data...")
df = pd.read_csv('DB_enriched.csv', encoding='utf-8')
df.replace(['not_mentioned', 'none', '', 'NM'], np.nan, inplace=True)

for col in ['translocation_detected', 'placental_accumulation', 'cytotoxicity_observed']:
    df[col] = df[col].map({'TRUE':1,'FALSE':0,True:1,False:0,'true':1,'false':0,1:1,0:0,'1':1,'0':0})

for col in ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

def parse_gd(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if s.lower() == 'term': return 19.0
    if s.lower() == 'preterm': return 17.0
    m = re.match(r'(?:GD|E|gd|e)?(\d+\.?\d*)', s)
    if m:
        val = float(m.group(1))
        if val <= 21: return val
    return np.nan

df['gd_numeric'] = df['gestational_stage'].apply(parse_gd)
print(f"    {len(df)} records, {df['gd_numeric'].notna().sum()} with parseable GD")

# ═══════════════════════════════════════════════════════════
# PART 2: GD STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n[2/7] Analyzing gestational day effects...")

gd_stats = []
for gd_val in sorted(df['gd_numeric'].dropna().unique()):
    if gd_val > 21: continue
    sub = df[df['gd_numeric'] == gd_val]
    t = sub['translocation_detected'].dropna()
    a = sub['placental_accumulation'].dropna()
    if len(t) >= 3 or len(a) >= 3:
        gd_stats.append({
            'gd': int(gd_val) if gd_val == int(gd_val) else gd_val,
            'n_trans': int(len(t)), 'trans_rate': round(float(t.mean()), 3) if len(t) > 0 else None,
            'n_accum': int(len(a)), 'accum_rate': round(float(a.mean()), 3) if len(a) > 0 else None,
        })

def gd_window(gd):
    if pd.isna(gd): return None
    if gd < 12: return 'early'
    if gd < 14: return '12-13'
    if gd < 15.5: return '14-15'
    if gd < 17.5: return '15.5-17'
    return '17.5+'

df['gd_window'] = df['gd_numeric'].apply(gd_window)
gd_window_stats = []
for window in ['12-13', '14-15', '15.5-17', '17.5+']:
    sub = df[df['gd_window'] == window]
    t = sub['translocation_detected'].dropna()
    a = sub['placental_accumulation'].dropna()
    if len(t) > 0:
        gd_window_stats.append({
            'window': window, 'n': int(len(t)),
            'trans_rate': round(float(t.mean()), 3),
            'accum_rate': round(float(a.mean()), 3) if len(a) > 0 else 0,
        })

for s in gd_window_stats:
    print(f"    GD {s['window']}: n={s['n']}, trans={s['trans_rate']:.0%}, accum={s['accum_rate']:.0%}")

# ═══════════════════════════════════════════════════════════
# PART 3: TRAIN MODELS
# ═══════════════════════════════════════════════════════════
print("\n[3/7] Training models...")

CAT_FEATURES = ['core_material', 'material_class', 'shape', 'surface_coating',
                'surface_charge_cat', 'model_type', 'species', 'exposure_route',
                'targeting_ligand', 'therapeutic_payload']
NUM_FEATURES = ['size_nm', 'zeta_potential_mV', 'dose_value', 'exposure_duration_h', 'gd_numeric']
ML_FEATURES = [c + '_enc' for c in CAT_FEATURES] + NUM_FEATURES

AU_COATINGS = ['PEG','mPEG','COOH','NH2','citrate','PVP','BSA','chitosan',
               'silica','PEG_COOH','PEG_NH2','dextran','transferrin','folate','lipid','DSPE-PEG']
AU_LIGANDS = ['none','insulin','glucose','folate','transferrin','CGKRK',
              'iRGD','RGD','NKGLRNK','RSGVAKS','antibody','aptamer']
AU_PAYLOADS = ['none','anakinra','dexamethasone','siRNA','curcumin']
AU_CHARGES = ['negative','neutral','positive']
AU_SIZES = [5, 10, 15, 20, 30, 50, 100]

label_encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    all_vals = df[col].fillna('_missing_')
    extras = []
    if col == 'surface_coating': extras = AU_COATINGS
    elif col == 'targeting_ligand': extras = AU_LIGANDS
    elif col == 'therapeutic_payload': extras = AU_PAYLOADS
    combined = pd.concat([all_vals, pd.Series(extras + ['_missing_'])])
    le.fit(combined)
    df[col + '_enc'] = le.transform(df[col].fillna('_missing_'))
    label_encoders[col] = le

TARGETS = {
    'translocation_detected': 'Fetal transfer',
    'placental_accumulation': 'Placental accumulation',
    'cytotoxicity_observed': 'Cytotoxicity',
}

model_info = {}
trained_models = {}

for target, desc in TARGETS.items():
    mask = df[target].notna()
    X = df.loc[mask, ML_FEATURES].copy()
    y = df.loc[mask, target].astype(int)
    for col in NUM_FEATURES:
        X[col] = X[col].fillna(X[col].median())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates_models = [
        ('Extra Trees', ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced')),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')),
        ('XGBoost', xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42,
                                       eval_metric='logloss', use_label_encoder=False)),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1,
                                          class_weight='balanced', max_depth=6)),
    ]

    best_auc, best_model, best_name, best_acc, best_f1 = 0, None, '', 0, 0
    for name, model in candidates_models:
        try:
            auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
            f1 = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
            if auc > best_auc:
                best_auc, best_model, best_name = auc, model, name
                best_acc, best_f1 = acc, f1
        except:
            pass

    best_model.fit(X, y)
    trained_models[target] = best_model
    n0, n1 = int(sum(y == 0)), int(sum(y == 1))
    model_info[target] = {
        'name': best_name, 'auc': round(best_auc, 3),
        'accuracy': round(best_acc, 3), 'f1': round(best_f1, 3),
        'n_samples': int(len(X)), 'n_class0': n0, 'n_class1': n1,
    }
    print(f"    {target}: {best_name} AUC={best_auc:.3f} Acc={best_acc:.1%}")

    if hasattr(best_model, 'feature_importances_'):
        fi = sorted(zip(ML_FEATURES, best_model.feature_importances_),
                    key=lambda x: x[1], reverse=True)
        model_info[target]['top_features'] = [
            {'name': f.replace('_enc', ''), 'importance': round(float(imp), 3)}
            for f, imp in fi[:8]
        ]

# ═══════════════════════════════════════════════════════════
# HELPER: predict a batch of candidates
# ═══════════════════════════════════════════════════════════
def predict_batch(cands_list):
    """Takes list of dicts, returns DataFrame with predictions."""
    cdf = pd.DataFrame(cands_list)
    for col in CAT_FEATURES:
        le = label_encoders[col]
        vals = cdf[col].fillna('_missing_')
        known = set(le.classes_)
        vals = vals.apply(lambda x: x if x in known else '_missing_')
        cdf[col + '_enc'] = le.transform(vals)
    Xc = cdf[ML_FEATURES].copy()
    for col in NUM_FEATURES:
        Xc[col] = Xc[col].fillna(0)
    for target in TARGETS:
        cdf[f'{target}_prob'] = trained_models[target].predict_proba(Xc)[:, 1]
    cdf['score'] = (
        cdf['placental_accumulation_prob'] * 2.0
        - cdf['translocation_detected_prob'] * 1.5
        - cdf['cytotoxicity_observed_prob'] * 1.0
    )
    return cdf

def make_cand(coating, ligand, payload, charge, size, gd=15, dose=5):
    return {
        'core_material': 'Au', 'material_class': 'inorganic_metal',
        'shape': 'spherical', 'surface_coating': coating,
        'surface_charge_cat': charge, 'model_type': 'in_vivo_mouse',
        'species': 'mouse', 'exposure_route': 'IV',
        'targeting_ligand': ligand, 'therapeutic_payload': payload,
        'size_nm': size, 'zeta_potential_mV': 0,
        'dose_value': dose, 'exposure_duration_h': 24, 'gd_numeric': gd,
    }

# ═══════════════════════════════════════════════════════════
# PART 4: COMBINATION ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n[4/7] Analyzing combinations...")

# Full candidate grid
candidates = []
for coating in AU_COATINGS:
    for ligand in AU_LIGANDS:
        for payload in AU_PAYLOADS:
            for charge in AU_CHARGES:
                for size in AU_SIZES:
                    candidates.append(make_cand(coating, ligand, payload, charge, size))

cand_df = predict_batch(candidates)

# ── HEATMAP: ALL coatings × ALL ligands ──
print("    Computing heatmap (all coatings × all ligands)...")
heatmap_coats = AU_COATINGS  # all 16
heatmap_ligs = AU_LIGANDS    # all 12
heatmap_data = []
heatmap_trans = []  # separate translocation heatmap
for coat in heatmap_coats:
    row_scores = []
    row_trans = []
    for lig in heatmap_ligs:
        sub = cand_df[(cand_df['surface_coating'] == coat) & (cand_df['targeting_ligand'] == lig)]
        row_scores.append(round(float(sub['score'].mean()), 3))
        row_trans.append(round(float(sub['translocation_detected_prob'].mean()), 3))
    heatmap_data.append(row_scores)
    heatmap_trans.append(row_trans)

hm_all = [v for row in heatmap_data for v in row]
hm_min, hm_max = min(hm_all), max(hm_all)
hm_range = hm_max - hm_min if hm_max > hm_min else 1
print(f"    Heatmap score range: {hm_min:.3f} to {hm_max:.3f} (delta={hm_range:.3f})")

ht_all = [v for row in heatmap_trans for v in row]
ht_min, ht_max = min(ht_all), max(ht_all)
ht_range = ht_max - ht_min if ht_max > ht_min else 1
print(f"    Trans heatmap range: {ht_min:.3f} to {ht_max:.3f} (delta={ht_range:.3f})")

# ── BEST COMBO PER COATING ──
best_per_coating = []
for coat in AU_COATINGS:
    sub = cand_df[cand_df['surface_coating'] == coat].sort_values('score', ascending=False)
    if len(sub) > 0:
        row = sub.iloc[0]
        best_per_coating.append({
            'coating': coat, 'ligand': row['targeting_ligand'],
            'payload': row['therapeutic_payload'], 'size': int(row['size_nm']),
            'charge': row['surface_charge_cat'],
            'accum': round(float(row['placental_accumulation_prob']), 3),
            'trans': round(float(row['translocation_detected_prob']), 3),
            'tox': round(float(row['cytotoxicity_observed_prob']), 3),
            'score': round(float(row['score']), 3),
        })
best_per_coating.sort(key=lambda x: x['score'], reverse=True)

# ── SENSITIVITY ANALYSIS: change one parameter at a time ──
print("    Computing sensitivity analysis...")
# Baseline: PEG, RGD, none, neutral, 15nm
baseline = make_cand('PEG', 'RGD', 'none', 'neutral', 15)

# Sweep coating (all else = baseline)
coat_sweep = []
for coat in AU_COATINGS:
    c = make_cand(coat, 'RGD', 'none', 'neutral', 15)
    coat_sweep.append(c)
coat_sweep_df = predict_batch(coat_sweep)
coat_sensitivity = {}
for _, row in coat_sweep_df.iterrows():
    coat_sensitivity[row['surface_coating']] = {
        'accum': round(float(row['placental_accumulation_prob']), 3),
        'trans': round(float(row['translocation_detected_prob']), 3),
        'tox': round(float(row['cytotoxicity_observed_prob']), 3),
        'score': round(float(row['score']), 3),
    }

# Sweep ligand
lig_sweep = [make_cand('PEG', lig, 'none', 'neutral', 15) for lig in AU_LIGANDS]
lig_sweep_df = predict_batch(lig_sweep)
lig_sensitivity = {}
for _, row in lig_sweep_df.iterrows():
    lig_sensitivity[row['targeting_ligand']] = {
        'accum': round(float(row['placental_accumulation_prob']), 3),
        'trans': round(float(row['translocation_detected_prob']), 3),
        'tox': round(float(row['cytotoxicity_observed_prob']), 3),
        'score': round(float(row['score']), 3),
    }

# Sweep size
size_sweep = [make_cand('PEG', 'RGD', 'none', 'neutral', sz) for sz in AU_SIZES]
size_sweep_df = predict_batch(size_sweep)
size_sensitivity = {}
for _, row in size_sweep_df.iterrows():
    size_sensitivity[str(int(row['size_nm']))] = {
        'accum': round(float(row['placental_accumulation_prob']), 3),
        'trans': round(float(row['translocation_detected_prob']), 3),
        'tox': round(float(row['cytotoxicity_observed_prob']), 3),
        'score': round(float(row['score']), 3),
    }

# Sweep charge
charge_sweep = [make_cand('PEG', 'RGD', 'none', ch, 15) for ch in AU_CHARGES]
charge_sweep_df = predict_batch(charge_sweep)
charge_sensitivity = {}
for _, row in charge_sweep_df.iterrows():
    charge_sensitivity[row['surface_charge_cat']] = {
        'accum': round(float(row['placental_accumulation_prob']), 3),
        'trans': round(float(row['translocation_detected_prob']), 3),
        'tox': round(float(row['cytotoxicity_observed_prob']), 3),
        'score': round(float(row['score']), 3),
    }

# Print sensitivity ranges
for name, data in [('Coating', coat_sensitivity), ('Ligand', lig_sensitivity),
                    ('Size', size_sensitivity), ('Charge', charge_sensitivity)]:
    scores = [v['score'] for v in data.values()]
    trans_vals = [v['trans'] for v in data.values()]
    print(f"    {name}: score {min(scores):.3f}-{max(scores):.3f}, "
          f"trans {min(trans_vals):.1%}-{max(trans_vals):.1%}")

# ── Parameter rankings (from full grid) ──
coat_rank = cand_df.groupby('surface_coating')['score'].mean().sort_values(ascending=False)
lig_rank = cand_df.groupby('targeting_ligand')['score'].mean().sort_values(ascending=False)
size_rank = cand_df.groupby('size_nm')['score'].mean().sort_values(ascending=False)
charge_rank = cand_df.groupby('surface_charge_cat')['score'].mean().sort_values(ascending=False)
payload_rank = cand_df.groupby('therapeutic_payload')['score'].mean().sort_values(ascending=False)

# ═══════════════════════════════════════════════════════════
# PART 5: PREDICTION GRID FOR INTERACTIVE TOOL
# ═══════════════════════════════════════════════════════════
print("\n[5/7] Pre-computing prediction grid...")

pred_grid = {}
grid_cands = []
for ci, coating in enumerate(AU_COATINGS):
    for li, ligand in enumerate(AU_LIGANDS):
        for pi, payload in enumerate(AU_PAYLOADS):
            for chi, charge in enumerate(AU_CHARGES):
                for si, size in enumerate(AU_SIZES):
                    c = make_cand(coating, ligand, payload, charge, size)
                    c['key'] = f"{ci},{li},{pi},{chi},{si}"
                    grid_cands.append(c)

grid_df = predict_batch(grid_cands)
for _, row in grid_df.iterrows():
    pred_grid[row['key']] = [
        int(round(row['placental_accumulation_prob'] * 100)),
        int(round(row['translocation_detected_prob'] * 100)),
        int(round(row['cytotoxicity_observed_prob'] * 100)),
    ]
print(f"    Grid: {len(pred_grid)} combinations")

# GD modifiers
gd_mods = {}
for gd_val in [12, 13, 14, 15, 16, 17, 18]:
    test = [make_cand(c, l, 'none', 'neutral', 15, gd=gd_val)
            for c in ['PEG','mPEG','NH2'] for l in ['none','RGD','glucose']]
    test_df = predict_batch(test)
    gd_mods[str(gd_val)] = [
        round(float(test_df['placental_accumulation_prob'].mean()) * 100),
        round(float(test_df['translocation_detected_prob'].mean()) * 100),
        round(float(test_df['cytotoxicity_observed_prob'].mean()) * 100),
    ]

dose_mods = {}
for dose_val in [0.5, 1, 5, 10, 20]:
    test = [make_cand(c, l, 'none', 'neutral', 15, dose=dose_val)
            for c in ['PEG','mPEG','NH2'] for l in ['none','RGD','glucose']]
    test_df = predict_batch(test)
    dose_mods[str(dose_val)] = [
        round(float(test_df['placental_accumulation_prob'].mean()) * 100),
        round(float(test_df['translocation_detected_prob'].mean()) * 100),
        round(float(test_df['cytotoxicity_observed_prob'].mean()) * 100),
    ]

# Shiri comparison
shiri_compare = [
    {'name': 'mPEG-GNP', 'coating': 'mPEG', 'ligand': 'none', 'actual_pf': 181, 'actual_uptake': 101},
    {'name': 'Insulin-GNP', 'coating': 'mPEG', 'ligand': 'insulin', 'actual_pf': 186, 'actual_uptake': 150},
    {'name': 'Glucose-GNP', 'coating': 'mPEG', 'ligand': 'glucose', 'actual_pf': 360, 'actual_uptake': 131},
]
for exp in shiri_compare:
    mask = ((cand_df['surface_coating'] == exp['coating']) &
            (cand_df['targeting_ligand'] == exp['ligand']) &
            (cand_df['size_nm'] == 20) & (cand_df['surface_charge_cat'] == 'neutral'))
    matches = cand_df[mask]
    if len(matches) > 0:
        row = matches.iloc[0]
        exp['pred_accum'] = round(float(row['placental_accumulation_prob']), 3)
        exp['pred_trans'] = round(float(row['translocation_detected_prob']), 3)
        exp['pred_tox'] = round(float(row['cytotoxicity_observed_prob']), 3)
        exp['score'] = round(float(row['score']), 3)

rec = best_per_coating[0]
shiri_compare.append({
    'name': f"{rec['coating']}+{rec['ligand']}+{rec['payload']} (המלצה)",
    'coating': rec['coating'], 'ligand': rec['ligand'],
    'actual_pf': None, 'actual_uptake': None,
    'pred_accum': rec['accum'], 'pred_trans': rec['trans'],
    'pred_tox': rec['tox'], 'score': rec['score'],
})

# ═══════════════════════════════════════════════════════════
# PART 6: SAVE DATA
# ═══════════════════════════════════════════════════════════
print("\n[6/7] Saving analysis data...")
all_data = {
    'model_info': model_info, 'gd_stats': gd_stats, 'gd_window_stats': gd_window_stats,
    'best_per_coating': best_per_coating, 'shiri_compare': shiri_compare,
    'coat_sensitivity': coat_sensitivity, 'lig_sensitivity': lig_sensitivity,
    'size_sensitivity': size_sensitivity, 'charge_sensitivity': charge_sensitivity,
    'heatmap': {'coatings': heatmap_coats, 'ligands': heatmap_ligs,
                'scores': heatmap_data, 'trans': heatmap_trans,
                'score_min': hm_min, 'score_max': hm_max,
                'trans_min': ht_min, 'trans_max': ht_max},
}
with open('gold_analysis_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════
# PART 7: GENERATE HTML
# ═══════════════════════════════════════════════════════════
print("\n[7/7] Generating HTML...")

pred_json = json.dumps(pred_grid, separators=(',', ':'))

# Build heatmap HTML with proper normalized colors
def score_to_color(val, vmin, vmax):
    """Map value to red→yellow→green. Returns CSS color."""
    if vmax == vmin:
        return 'hsl(60, 70%, 30%)'
    t = (val - vmin) / (vmax - vmin)  # 0=worst, 1=best
    hue = int(t * 120)  # 0=red, 60=yellow, 120=green
    light = 20 + int(t * 15)  # darker for bad, lighter for good
    return f'hsl({hue}, 75%, {light}%)'

def trans_to_color(val, vmin, vmax):
    """For translocation: LOW is good (green), HIGH is bad (red)."""
    if vmax == vmin:
        return 'hsl(60, 70%, 30%)'
    t = 1 - (val - vmin) / (vmax - vmin)  # inverted: low trans = green
    hue = int(t * 120)
    light = 20 + int(t * 15)
    return f'hsl({hue}, 75%, {light}%)'

# Score heatmap rows
heatmap_score_html = ''
for ci, coat in enumerate(heatmap_coats):
    heatmap_score_html += f'<tr><td class="row-header">{coat}</td>'
    for li, lig in enumerate(heatmap_ligs):
        val = heatmap_data[ci][li]
        color = score_to_color(val, hm_min, hm_max)
        heatmap_score_html += f'<td style="background:{color};color:#fff;">{val:.2f}</td>'
    heatmap_score_html += '</tr>\n'

# Translocation heatmap rows
heatmap_trans_html = ''
for ci, coat in enumerate(heatmap_coats):
    heatmap_trans_html += f'<tr><td class="row-header">{coat}</td>'
    for li, lig in enumerate(heatmap_ligs):
        val = heatmap_trans[ci][li]
        color = trans_to_color(val, ht_min, ht_max)
        heatmap_trans_html += f'<td style="background:{color};color:#fff;">{val:.0%}</td>'
    heatmap_trans_html += '</tr>\n'

# Best per coating table HTML
bpc_html = ''
for i, c in enumerate(best_per_coating):
    trans_class = 'good' if c['trans'] < 0.1 else ('warn' if c['trans'] < 0.3 else 'bad')
    tox_class = 'good' if c['tox'] < 0.1 else ('warn' if c['tox'] < 0.3 else 'bad')
    bpc_html += f'''<tr>
        <td class="rank">{i+1}</td>
        <td style="font-weight:bold;">{c['coating']}</td><td>{c['ligand']}</td>
        <td>{c['payload']}</td><td>{c['size']}nm</td><td>{c['charge']}</td>
        <td class="good">{c['accum']:.0%}</td>
        <td class="{trans_class}">{c['trans']:.0%}</td>
        <td class="{tox_class}">{c['tox']:.0%}</td>
        <td style="color:var(--gold);font-weight:bold;">{c['score']:.3f}</td>
    </tr>'''

# Model info boxes HTML
model_boxes_html = ''
for target, info in model_info.items():
    top_feats = ', '.join(f['name'] for f in info.get('top_features', [])[:5])
    model_boxes_html += f'''
    <div class="model-box">
        <b>{TARGETS[target]}:</b> {info['name']}
        <br>
        <span class="model-metric"><b>AUC:</b> {info['auc']}</span>
        <span class="model-metric"><b>דיוק:</b> {info['accuracy']:.0%}</span>
        <span class="model-metric"><b>F1:</b> {info['f1']}</span>
        <span class="model-metric"><b>נתוני אימון:</b> {info['n_samples']} ({info['n_class0']} שלילי, {info['n_class1']} חיובי)</span>
        <br><small style="color:var(--text3);">מאפיינים חשובים: {top_feats}</small>
    </div>'''

# Coating dropdown options
coat_options = ''.join(f'<option value="{i}" {"selected" if c=="PEG" else ""}>{c}</option>'
                       for i, c in enumerate(AU_COATINGS))
lig_options = ''.join(f'<option value="{i}" {"selected" if l=="RGD" else ""}>{l}</option>'
                      for i, l in enumerate(AU_LIGANDS))
pay_options = ''.join(f'<option value="{i}" {"selected" if p=="anakinra" else ""}>{p}</option>'
                      for i, p in enumerate(AU_PAYLOADS))
chg_options = ''.join(f'<option value="{i}" {"selected" if ch=="neutral" else ""}>{ch}</option>'
                      for i, ch in enumerate(AU_CHARGES))

html = f'''<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NanoPlacentaDB - Gold NP Optimizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
    --bg: #0f0f1a; --bg2: #1a1a2e; --bg3: #16213e; --bg4: #1e2a45;
    --gold: #f0c040; --gold2: #c4951a; --gold3: #ffd700;
    --green: #00c853; --red: #ff5252; --orange: #ff9800; --blue: #448aff;
    --text: #e8e8f0; --text2: #a0a0b8; --text3: #707088;
}}
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: var(--bg); color: var(--text);
       line-height: 1.6; direction: rtl; }}
.container {{ max-width: 1300px; margin: 0 auto; padding: 20px; }}
header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1020 100%);
         padding: 30px 0; border-bottom: 2px solid var(--gold2); text-align: center; }}
header h1 {{ font-size: 28px; color: var(--gold); margin-bottom: 8px; }}
header p {{ color: var(--text2); font-size: 15px; }}
.tabs {{ display: flex; justify-content: center; gap: 4px; padding: 16px 0;
         background: var(--bg2); border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
.tab {{ padding: 10px 28px; cursor: pointer; border-radius: 8px 8px 0 0;
        background: var(--bg3); color: var(--text2); font-size: 15px; transition: all 0.3s;
        border: 1px solid transparent; border-bottom: none; }}
.tab:hover {{ background: var(--bg4); color: var(--text); }}
.tab.active {{ background: var(--bg); color: var(--gold); border-color: var(--gold2); font-weight: bold; }}
.tab-content {{ display: none; animation: fadeIn 0.4s; }}
.tab-content.active {{ display: block; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; }} }}
.card {{ background: var(--bg3); border-radius: 12px; padding: 24px; margin-bottom: 24px;
         border: 1px solid #2a2a4a; }}
.card h2 {{ color: var(--gold); font-size: 20px; margin-bottom: 16px;
            padding-bottom: 8px; border-bottom: 1px solid #2a2a4a; }}
.hero {{ background: linear-gradient(135deg, #1a2a1a 0%, #16213e 50%, #2a1a1a 100%);
         border: 2px solid var(--gold); border-radius: 16px; padding: 32px;
         margin-bottom: 32px; text-align: center; }}
.hero h2 {{ color: var(--gold3); font-size: 24px; border: none; margin-bottom: 20px; }}
.hero-design {{ display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
.hero-pill {{ background: var(--bg3); border: 1px solid var(--gold2); border-radius: 20px;
              padding: 8px 20px; font-size: 14px; }}
.hero-pill b {{ color: var(--gold); }}
.hero-outcomes {{ display: flex; justify-content: center; gap: 40px; margin-top: 24px; flex-wrap: wrap; }}
.hero-stat {{ text-align: center; }}
.hero-stat .value {{ font-size: 36px; font-weight: bold; }}
.hero-stat .label {{ font-size: 13px; color: var(--text2); }}
.good {{ color: var(--green); }} .bad {{ color: var(--red); }} .warn {{ color: var(--orange); }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
.chart-box {{ background: var(--bg3); border-radius: 12px; padding: 20px; border: 1px solid #2a2a4a; }}
.chart-box h3 {{ color: var(--gold); font-size: 16px; margin-bottom: 12px; text-align: center; }}
canvas {{ max-height: 320px; }}
.heatmap {{ width: 100%; border-collapse: collapse; font-size: 11px; direction: ltr; }}
.heatmap th {{ background: var(--bg2); color: var(--gold); padding: 6px 3px; text-align: center;
               font-weight: normal; border: 1px solid #2a2a4a; font-size: 10px; }}
.heatmap td {{ padding: 5px 3px; text-align: center; border: 1px solid #1a1a2e;
               font-weight: bold; font-size: 11px; }}
.heatmap .row-header {{ background: var(--bg2); color: var(--gold); text-align: right;
                        padding-right: 8px; direction: ltr; font-size: 11px; min-width: 70px; }}
.combo-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.combo-table th {{ background: var(--bg2); color: var(--gold); padding: 10px 8px;
                   text-align: center; border-bottom: 2px solid var(--gold2); }}
.combo-table td {{ padding: 8px; text-align: center; border-bottom: 1px solid #2a2a4a; }}
.combo-table tr:hover {{ background: var(--bg4); }}
.combo-table .rank {{ color: var(--gold); font-weight: bold; }}
.gd-recommendation {{ background: linear-gradient(135deg, #1a2a2a, #16213e);
                       border: 1px solid var(--blue); border-radius: 12px; padding: 20px; margin-top: 16px; }}
.gd-recommendation h3 {{ color: var(--blue); margin-bottom: 12px; }}
.model-box {{ background: var(--bg2); border-radius: 8px; padding: 16px; margin-top: 12px;
              border-right: 3px solid var(--gold); }}
.model-metric {{ display: inline-block; background: var(--bg3); border-radius: 6px;
                 padding: 4px 12px; margin: 4px; font-size: 13px; }}
.model-metric b {{ color: var(--gold); }}
.predictor-layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.input-panel {{ background: var(--bg3); border-radius: 12px; padding: 24px; border: 1px solid #2a2a4a; }}
.input-panel h2 {{ color: var(--gold); margin-bottom: 16px; }}
.form-group {{ margin-bottom: 16px; }}
.form-group label {{ display: block; color: var(--text2); font-size: 14px; margin-bottom: 4px; }}
.form-group select, .form-group input[type=range] {{
    width: 100%; padding: 8px 12px; background: var(--bg2); color: var(--text);
    border: 1px solid #3a3a5a; border-radius: 6px; font-size: 14px; }}
.form-group select:focus {{ border-color: var(--gold); outline: none; }}
.size-display {{ text-align: center; color: var(--gold); font-size: 20px; font-weight: bold; margin: 4px 0; }}
.result-panel {{ background: var(--bg3); border-radius: 12px; padding: 24px;
                 border: 1px solid #2a2a4a; text-align: center; }}
.result-panel h2 {{ color: var(--gold); margin-bottom: 20px; }}
.gauges {{ display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin: 20px 0; }}
.gauge {{ position: relative; width: 140px; height: 140px; }}
.gauge svg {{ width: 140px; height: 140px; transform: rotate(-90deg); }}
.gauge-circle-bg {{ fill: none; stroke: #2a2a4a; stroke-width: 10; }}
.gauge-circle {{ fill: none; stroke-width: 10; stroke-linecap: round; transition: stroke-dashoffset 0.8s ease; }}
.gauge-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
               font-size: 28px; font-weight: bold; }}
.gauge-label {{ text-align: center; font-size: 13px; color: var(--text2); margin-top: 8px; }}
.composite-score {{ font-size: 48px; font-weight: bold; color: var(--gold); margin: 20px 0 8px; }}
.verdict {{ font-size: 18px; padding: 12px 24px; border-radius: 8px; display: inline-block; margin: 12px 0; }}
.verdict-excellent {{ background: rgba(0,200,83,0.15); color: var(--green); border: 1px solid var(--green); }}
.verdict-good {{ background: rgba(68,138,255,0.15); color: var(--blue); border: 1px solid var(--blue); }}
.verdict-ok {{ background: rgba(255,152,0,0.15); color: var(--orange); border: 1px solid var(--orange); }}
.verdict-bad {{ background: rgba(255,82,82,0.15); color: var(--red); border: 1px solid var(--red); }}
.model-explain {{ background: var(--bg2); border-radius: 12px; padding: 20px; margin-top: 24px;
                   border: 1px solid #2a2a4a; text-align: right; }}
.model-explain h3 {{ color: var(--gold); margin-bottom: 12px; }}
.model-explain p {{ color: var(--text2); font-size: 14px; line-height: 1.8; }}
.metric-badge {{ display: inline-block; background: var(--bg3); padding: 2px 10px;
                 border-radius: 4px; color: var(--gold); font-weight: bold; }}
.legend {{ display: flex; gap: 16px; justify-content: center; margin: 12px 0; font-size: 12px; color: var(--text2); }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{ width: 16px; height: 16px; border-radius: 3px; }}
@media (max-width: 768px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    .predictor-layout {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
    <div class="container">
        <h1>NanoPlacentaDB - Gold NP Experiment Optimizer</h1>
        <p>כלי מבוסס למידת מכונה לאופטימיזציה של ניסויי חלקיקי זהב בשלייה | {len(df)} רשומות, {df['study_id'].nunique()} מחקרים</p>
    </div>
</header>

<div class="tabs">
    <div class="tab active" onclick="switchTab('findings')">ממצאים והמלצות</div>
    <div class="tab" onclick="switchTab('predictor')">כלי חיזוי אינטראקטיבי</div>
</div>

<div class="container">

<!-- ═══════════ TAB 1: FINDINGS ═══════════ -->
<div id="tab-findings" class="tab-content active">

<div class="hero">
    <h2>מערך הניסוי המומלץ לשירי</h2>
    <div class="hero-design">
        <div class="hero-pill"><b>ציפוי:</b> {best_per_coating[0]['coating']}</div>
        <div class="hero-pill"><b>ליגנד מכוון:</b> {best_per_coating[0]['ligand']}</div>
        <div class="hero-pill"><b>מטען טיפולי:</b> {best_per_coating[0]['payload']}</div>
        <div class="hero-pill"><b>גודל:</b> {best_per_coating[0]['size']} nm</div>
        <div class="hero-pill"><b>מטען חשמלי:</b> ניטרלי</div>
        <div class="hero-pill"><b>יום הריון:</b> GD 15-16</div>
    </div>
    <div class="hero-outcomes">
        <div class="hero-stat">
            <div class="value good">{best_per_coating[0]['accum']:.0%}</div>
            <div class="label">הצטברות בשלייה</div>
        </div>
        <div class="hero-stat">
            <div class="value {'good' if best_per_coating[0]['trans'] < 0.1 else 'bad'}">{best_per_coating[0]['trans']:.0%}</div>
            <div class="label">מעבר לעובר</div>
        </div>
        <div class="hero-stat">
            <div class="value {'good' if best_per_coating[0]['tox'] < 0.1 else 'warn'}">{best_per_coating[0]['tox']:.0%}</div>
            <div class="label">רעילות</div>
        </div>
    </div>
</div>

<!-- SENSITIVITY ANALYSIS -->
<div class="card">
    <h2>ניתוח רגישות - מה באמת משפיע?</h2>
    <p style="color:var(--text2); margin-bottom:16px;">
        כל גרף מראה מה קורה כשמשנים פרמטר <b>אחד בלבד</b> (שאר הפרמטרים קבועים: Au, 15nm, neutral, RGD, IV).
        הבר הירוק = הצטברות בשלייה, הבר האדום = מעבר לעובר.
    </p>
    <div class="chart-row">
        <div class="chart-box">
            <h3>השפעת שינוי הציפוי</h3>
            <canvas id="sensCoatChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>השפעת שינוי הליגנד</h3>
            <canvas id="sensLigChart"></canvas>
        </div>
    </div>
    <div class="chart-row">
        <div class="chart-box">
            <h3>השפעת הגודל</h3>
            <canvas id="sensSizeChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>השפעת המטען החשמלי</h3>
            <canvas id="sensChargeChart"></canvas>
        </div>
    </div>
</div>

<!-- GD ANALYSIS -->
<div class="card">
    <h2>באיזה שלב בהריון? (Gestational Day)</h2>
    <p style="color:var(--text2); margin-bottom:16px;">
        ניתוח סטטיסטי של {df['gd_numeric'].notna().sum()} רשומות. בעכברות ההריון נמשך 19-21 ימים.
    </p>
    <div class="chart-row">
        <div class="chart-box">
            <h3>שיעור מעבר לעובר לפי GD</h3>
            <canvas id="gdTransChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>שיעור הצטברות בשלייה לפי GD</h3>
            <canvas id="gdAccumChart"></canvas>
        </div>
    </div>
    <div class="gd-recommendation">
        <h3>המלצה: GD 15-16</h3>
        <p><b>GD 16</b> מציג את שיעור המעבר לעובר <b>הנמוך ביותר</b> (כ-15%) עם הצטברות סבירה (73%).
        <b>GD 14-15</b> נפוץ יותר במחקרי Au NP, עם הצטברות גבוהה (75-100%) אך מעבר גבוה יותר (50%).
        </p>
        <p style="margin-top:8px;">
        <b>לשירי:</b> מומלץ <b>GD 15-16</b>. השלייה בוגרת ולוכדת חלקיקים, מעבר לעובר מינימלי.
        ⚠️ גודל מדגם קטן (n=6-13 לכל GD), מחקרים שונים השתמשו בחומרים שונים.
        </p>
    </div>
</div>

<!-- COMBINATION HEATMAPS -->
<div class="card">
    <h2>ניתוח קומבינציות - ציפוי × ליגנד</h2>
    <p style="color:var(--text2); margin-bottom:8px;">
        כל תא מציג את הציון הממוצע (או שיעור מעבר) עבור שילוב ספציפי של ציפוי וליגנד.
        <b>ירוק = טוב, אדום = רע</b>. השילובים נבדקו דרך מודל ה-ML, לא כל אחד בנפרד.
    </p>

    <h3 style="color:var(--gold);margin:16px 0 8px;">ציון מורכב (ירוק = גבוה = טוב)</h3>
    <div class="legend">
        <div class="legend-item"><div class="legend-swatch" style="background:hsl(0,75%,20%);"></div> נמוך ({hm_min:.2f})</div>
        <div class="legend-item"><div class="legend-swatch" style="background:hsl(60,75%,27%);"></div> בינוני</div>
        <div class="legend-item"><div class="legend-swatch" style="background:hsl(120,75%,35%);"></div> גבוה ({hm_max:.2f})</div>
    </div>
    <div style="overflow-x:auto; margin-bottom:24px;">
        <table class="heatmap">
            <tr><th style="background:var(--bg);">ציפוי \\ ליגנד</th>
            {''.join(f'<th>{lig}</th>' for lig in heatmap_ligs)}</tr>
            {heatmap_score_html}
        </table>
    </div>

    <h3 style="color:var(--gold);margin:16px 0 8px;">שיעור מעבר לעובר (ירוק = נמוך = טוב)</h3>
    <div class="legend">
        <div class="legend-item"><div class="legend-swatch" style="background:hsl(120,75%,35%);"></div> נמוך ({ht_min:.0%})</div>
        <div class="legend-item"><div class="legend-swatch" style="background:hsl(60,75%,27%);"></div> בינוני</div>
        <div class="legend-item"><div class="legend-swatch" style="background:hsl(0,75%,20%);"></div> גבוה ({ht_max:.0%})</div>
    </div>
    <div style="overflow-x:auto;">
        <table class="heatmap">
            <tr><th style="background:var(--bg);">ציפוי \\ ליגנד</th>
            {''.join(f'<th>{lig}</th>' for lig in heatmap_ligs)}</tr>
            {heatmap_trans_html}
        </table>
    </div>
</div>

<!-- BEST PER COATING TABLE -->
<div class="card">
    <h2>השילוב הטוב ביותר לכל סוג ציפוי</h2>
    <p style="color:var(--text2); margin-bottom:16px;">
        לכל ציפוי, המודל מוצא את השילוב האופטימלי של ליגנד, מטען, גודל ומטען חשמלי.
        זה מאפשר להשוות בין ציפויים שונים כשכל אחד מותאם באופן אופטימלי.
    </p>
    <div style="overflow-x:auto;">
        <table class="combo-table">
            <tr><th>#</th><th>ציפוי</th><th>ליגנד</th><th>מטען</th><th>גודל</th>
                <th>מטען חשמלי</th><th>הצטברות</th><th>מעבר</th><th>רעילות</th><th>ציון</th></tr>
            {bpc_html}
        </table>
    </div>
</div>

<!-- SHIRI COMPARISON -->
<div class="card">
    <h2>השוואה לתוצאות שירי הקיימות</h2>
    <p style="color:var(--text2); margin-bottom:16px;">
        P:F ratio = יחס שלייה:עובר. ככל שגבוה יותר - החלקיק נשאר בשלייה ולא עובר.
    </p>
    <div style="max-width:800px; margin:0 auto;">
        <canvas id="shiriChart" style="max-height:350px;"></canvas>
    </div>
</div>

<!-- MODEL INFO -->
<div class="card">
    <h2>על המודל - איך זה עובד?</h2>
    <p style="color:var(--text2); line-height:1.8; margin-bottom:16px;">
        המערכת משתמשת ב<b>למידת מכונה</b> ללמוד דפוסים מ-{len(df)} רשומות.
        <b>שיטת הניקוד:</b> ציון = (הצטברות × 2) − (מעבר × 1.5) − (רעילות × 1.0).
    </p>
    {model_boxes_html}
    <div style="margin-top:16px; padding:16px; background:rgba(255,152,0,0.1); border-radius:8px; border-right:3px solid var(--orange);">
        <b style="color:var(--orange);">מגבלות:</b>
        <ul style="color:var(--text2); font-size:14px; margin-top:8px; padding-right:20px;">
            <li>רשומות Au NP: 59 מתוך 486. המודל לומד גם מחומרים אחרים ומכליל</li>
            <li>accumulation: 94% חיובי בנתוני האימון → המודל מנבא הצטברות גבוהה כמעט תמיד</li>
            <li><b>המבדיל העיקרי בין ציפויים: שיעור מעבר לעובר (translocation)</b></li>
            <li>AUC &gt; 0.80 = טוב, &gt; 0.90 = מצוין | המודלים: 0.82-0.93</li>
            <li>החיזויים הסתברותיים - יש לאמת בניסוי</li>
        </ul>
    </div>
</div>

</div><!-- end findings tab -->

<!-- ═══════════ TAB 2: PREDICTOR ═══════════ -->
<div id="tab-predictor" class="tab-content">

<div class="predictor-layout">
    <div class="input-panel">
        <h2>הגדרת פרמטרים</h2>
        <p style="color:var(--text2); font-size:14px; margin-bottom:16px;">
            בחרי את מאפייני החלקיק - המודל ינבא את התוצאות הצפויות.
        </p>
        <div class="form-group">
            <label>ציפוי (Coating)</label>
            <select id="sel-coating" onchange="predict()">{coat_options}</select>
        </div>
        <div class="form-group">
            <label>ליגנד מכוון (Targeting Ligand)</label>
            <select id="sel-ligand" onchange="predict()">{lig_options}</select>
        </div>
        <div class="form-group">
            <label>מטען טיפולי (Payload)</label>
            <select id="sel-payload" onchange="predict()">{pay_options}</select>
        </div>
        <div class="form-group">
            <label>מטען חשמלי (Charge)</label>
            <select id="sel-charge" onchange="predict()">{chg_options}</select>
        </div>
        <div class="form-group">
            <label>גודל חלקיק (nm)</label>
            <input type="range" id="sel-size" min="0" max="{len(AU_SIZES)-1}" value="2"
                   oninput="updateSizeLabel(); predict();">
            <div class="size-display" id="size-label">{AU_SIZES[2]} nm</div>
        </div>
        <div class="form-group">
            <label>יום הריון (GD)</label>
            <select id="sel-gd" onchange="predict()">
                <option value="12">GD 12</option><option value="13">GD 13</option>
                <option value="14">GD 14</option><option value="15" selected>GD 15 (מומלץ)</option>
                <option value="16">GD 16 (מעבר נמוך ביותר)</option>
                <option value="17">GD 17</option><option value="18">GD 18</option>
            </select>
        </div>
        <div class="form-group">
            <label>מינון (mg/kg)</label>
            <select id="sel-dose" onchange="predict()">
                <option value="0.5">0.5</option><option value="1">1</option>
                <option value="5" selected>5</option><option value="10">10</option>
                <option value="20">20</option>
            </select>
        </div>
    </div>

    <div class="result-panel">
        <h2>תוצאות חזויות</h2>
        <div class="gauges">
            <div>
                <div class="gauge">
                    <svg viewBox="0 0 140 140">
                        <circle class="gauge-circle-bg" cx="70" cy="70" r="60"/>
                        <circle class="gauge-circle" id="gc-accum" cx="70" cy="70" r="60"
                                stroke="var(--green)" stroke-dasharray="377" stroke-dashoffset="377"/>
                    </svg>
                    <span class="gauge-text good" id="gt-accum">--%</span>
                </div>
                <div class="gauge-label">הצטברות בשלייה</div>
            </div>
            <div>
                <div class="gauge">
                    <svg viewBox="0 0 140 140">
                        <circle class="gauge-circle-bg" cx="70" cy="70" r="60"/>
                        <circle class="gauge-circle" id="gc-trans" cx="70" cy="70" r="60"
                                stroke="var(--red)" stroke-dasharray="377" stroke-dashoffset="377"/>
                    </svg>
                    <span class="gauge-text bad" id="gt-trans">--%</span>
                </div>
                <div class="gauge-label">מעבר לעובר</div>
            </div>
            <div>
                <div class="gauge">
                    <svg viewBox="0 0 140 140">
                        <circle class="gauge-circle-bg" cx="70" cy="70" r="60"/>
                        <circle class="gauge-circle" id="gc-tox" cx="70" cy="70" r="60"
                                stroke="var(--orange)" stroke-dasharray="377" stroke-dashoffset="377"/>
                    </svg>
                    <span class="gauge-text warn" id="gt-tox">--%</span>
                </div>
                <div class="gauge-label">רעילות תאית</div>
            </div>
        </div>
        <div class="composite-score" id="comp-score">--</div>
        <div style="color:var(--text2); font-size:14px;">ציון מורכב</div>
        <div class="verdict" id="verdict">בחרי פרמטרים כדי לקבל חיזוי</div>
    </div>
</div>

<div class="model-explain">
    <h3>על מה מתבסס החיזוי?</h3>
    <p>
        <span class="metric-badge">Extra Trees</span> / <span class="metric-badge">Gradient Boosting</span> -
        אלגוריתמי יער החלטות שאומנו על <span class="metric-badge">{len(df)} רשומות</span>.
    </p>
    <p style="margin-top:12px;">
        <b>הצטברות:</b> <span class="metric-badge">AUC={model_info['placental_accumulation']['auc']}</span>
        | <b>מעבר:</b> <span class="metric-badge">AUC={model_info['translocation_detected']['auc']}</span>
        | <b>רעילות:</b> <span class="metric-badge">AUC={model_info['cytotoxicity_observed']['auc']}</span>
    </p>
    <p style="margin-top:8px; color:var(--text3); font-size:13px;">
        AUC: 0.5 = ניחוש, 1.0 = מושלם. מעל 0.80 = טוב.
        הציון = (הצטברות×2) − (מעבר×1.5) − (רעילות×1.0)
    </p>
</div>

</div><!-- end predictor tab -->
</div>

<footer style="text-align:center; padding:20px; color:var(--text3); font-size:12px; border-top:1px solid #2a2a4a; margin-top:40px;">
    NanoPlacentaDB Gold NP Optimizer | {len(df)} records, {df['study_id'].nunique()} studies | ML: Extra Trees / Gradient Boosting
</footer>

<script>
const SIZES = {json.dumps(AU_SIZES)};
const PRED = {pred_json};
const GD_MODS = {json.dumps(gd_mods)};
const DOSE_MODS = {json.dumps(dose_mods)};
const BASE_GD = "15", BASE_DOSE = "5";

function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
}}

function updateSizeLabel() {{
    document.getElementById('size-label').textContent = SIZES[document.getElementById('sel-size').value] + ' nm';
}}

function predict() {{
    const ci = document.getElementById('sel-coating').value;
    const li = document.getElementById('sel-ligand').value;
    const pi = document.getElementById('sel-payload').value;
    const chi = document.getElementById('sel-charge').value;
    const si = document.getElementById('sel-size').value;
    const gd = document.getElementById('sel-gd').value;
    const dose = document.getElementById('sel-dose').value;
    const key = ci+','+li+','+pi+','+chi+','+si;
    let pred = PRED[key];
    if (!pred) {{ document.getElementById('verdict').textContent='שילוב לא נמצא'; return; }}
    let [accum,trans,tox] = pred;
    if (gd!==BASE_GD && GD_MODS[gd] && GD_MODS[BASE_GD]) {{
        const b=GD_MODS[BASE_GD], g=GD_MODS[gd];
        accum=Math.max(0,Math.min(100,accum+(g[0]-b[0])));
        trans=Math.max(0,Math.min(100,trans+(g[1]-b[1])));
        tox=Math.max(0,Math.min(100,tox+(g[2]-b[2])));
    }}
    if (dose!==BASE_DOSE && DOSE_MODS[dose] && DOSE_MODS[BASE_DOSE]) {{
        const b=DOSE_MODS[BASE_DOSE], d=DOSE_MODS[dose];
        accum=Math.max(0,Math.min(100,accum+(d[0]-b[0])));
        trans=Math.max(0,Math.min(100,trans+(d[1]-b[1])));
        tox=Math.max(0,Math.min(100,tox+(d[2]-b[2])));
    }}
    updateGauge('accum',accum); updateGauge('trans',trans); updateGauge('tox',tox);
    const score=(accum/100*2)-(trans/100*1.5)-(tox/100*1.0);
    document.getElementById('comp-score').textContent=score.toFixed(3);
    const v=document.getElementById('verdict');
    if(accum>=80&&trans<=10&&tox<=20){{ v.textContent='שילוב מצוין! הצטברות גבוהה, מעבר מינימלי'; v.className='verdict verdict-excellent'; }}
    else if(accum>=70&&trans<=20){{ v.textContent='שילוב טוב'; v.className='verdict verdict-good'; }}
    else if(trans>=50){{ v.textContent='אזהרה: מעבר גבוה לעובר'; v.className='verdict verdict-bad'; }}
    else if(tox>=50){{ v.textContent='אזהרה: רעילות גבוהה'; v.className='verdict verdict-bad'; }}
    else {{ v.textContent='שילוב סביר'; v.className='verdict verdict-ok'; }}
}}
function updateGauge(id,pct) {{
    const off=377*(1-pct/100);
    document.getElementById('gc-'+id).style.strokeDashoffset=off;
    document.getElementById('gt-'+id).textContent=Math.round(pct)+'%';
}}

Chart.defaults.color='#a0a0b8'; Chart.defaults.borderColor='#2a2a4a';

// ── SENSITIVITY CHARTS ──
const coatSens = {json.dumps(coat_sensitivity)};
const ligSens = {json.dumps(lig_sensitivity)};
const sizeSens = {json.dumps(size_sensitivity)};
const chargeSens = {json.dumps(charge_sensitivity)};

function makeSensChart(canvasId, data, labelKey) {{
    const labels = Object.keys(data);
    const accum = labels.map(k => Math.round(data[k].accum * 100));
    const trans = labels.map(k => Math.round(data[k].trans * 100));
    new Chart(document.getElementById(canvasId), {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [
                {{ label: 'הצטברות (%)', data: accum, backgroundColor: 'rgba(0,200,83,0.7)', borderRadius: 3 }},
                {{ label: 'מעבר לעובר (%)', data: trans, backgroundColor: 'rgba(255,82,82,0.7)', borderRadius: 3 }}
            ]
        }},
        options: {{
            responsive: true,
            indexAxis: labels.length > 6 ? 'y' : 'x',
            plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12 }} }} }},
            scales: {{
                x: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v+'%' }} }},
                y: {{ ticks: {{ font: {{ size: 11 }} }} }}
            }}
        }}
    }});
}}

makeSensChart('sensCoatChart', coatSens);
makeSensChart('sensLigChart', ligSens);

// Size chart - line
const szLabels = Object.keys(sizeSens).sort((a,b) => parseInt(a)-parseInt(b));
new Chart(document.getElementById('sensSizeChart'), {{
    type: 'line',
    data: {{
        labels: szLabels.map(s => s+' nm'),
        datasets: [
            {{ label: 'הצטברות', data: szLabels.map(s => Math.round(sizeSens[s].accum*100)),
               borderColor: 'rgba(0,200,83,0.9)', backgroundColor: 'rgba(0,200,83,0.1)',
               fill: true, tension: 0.3, pointRadius: 5 }},
            {{ label: 'מעבר', data: szLabels.map(s => Math.round(sizeSens[s].trans*100)),
               borderColor: 'rgba(255,82,82,0.9)', backgroundColor: 'rgba(255,82,82,0.1)',
               fill: true, tension: 0.3, pointRadius: 5 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12 }} }} }},
        scales: {{ y: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v+'%' }} }} }}
    }}
}});

// Charge chart
makeSensChart('sensChargeChart', chargeSens);

// ── GD CHARTS ──
const gdStats = {json.dumps(gd_stats)};
const gdT = gdStats.filter(d => d.trans_rate!==null && d.n_trans>=3 && d.gd>=10);
new Chart(document.getElementById('gdTransChart'), {{
    type: 'bar',
    data: {{
        labels: gdT.map(d => 'GD '+d.gd),
        datasets: [{{ label: 'מעבר לעובר (%)', data: gdT.map(d => Math.round(d.trans_rate*100)),
            backgroundColor: gdT.map(d => d.trans_rate<0.3?'rgba(0,200,83,0.7)':d.trans_rate<0.6?'rgba(255,152,0,0.7)':'rgba(255,82,82,0.7)'),
            borderRadius: 4 }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend:{{display:false}}, tooltip:{{callbacks:{{afterLabel:ctx=>'n='+gdT[ctx.dataIndex].n_trans}}}} }},
        scales: {{ y:{{beginAtZero:true,max:100,ticks:{{callback:v=>v+'%'}}}} }}
    }}
}});

const gdA = gdStats.filter(d => d.accum_rate!==null && d.n_accum>=3 && d.gd>=10);
new Chart(document.getElementById('gdAccumChart'), {{
    type: 'bar',
    data: {{
        labels: gdA.map(d => 'GD '+d.gd),
        datasets: [{{ label: 'הצטברות (%)', data: gdA.map(d => Math.round(d.accum_rate*100)),
            backgroundColor: gdA.map(d => d.accum_rate>=0.8?'rgba(0,200,83,0.7)':'rgba(255,152,0,0.7)'),
            borderRadius: 4 }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend:{{display:false}}, tooltip:{{callbacks:{{afterLabel:ctx=>'n='+gdA[ctx.dataIndex].n_accum}}}} }},
        scales: {{ y:{{beginAtZero:true,max:100,ticks:{{callback:v=>v+'%'}}}} }}
    }}
}});

// ── SHIRI COMPARISON ──
const shiriData = {json.dumps(shiri_compare, ensure_ascii=False)};
new Chart(document.getElementById('shiriChart'), {{
    type: 'bar',
    data: {{
        labels: shiriData.map(d => d.name),
        datasets: [
            {{ label: 'מעבר חזוי (%)', data: shiriData.map(d => Math.round((d.pred_trans||0)*100)),
               backgroundColor: 'rgba(255,82,82,0.7)', borderRadius: 4 }},
            {{ label: 'הצטברות חזויה (%)', data: shiriData.map(d => Math.round((d.pred_accum||0)*100)),
               backgroundColor: 'rgba(0,200,83,0.7)', borderRadius: 4 }},
            {{ label: 'P:F Ratio', data: shiriData.map(d => d.actual_pf||0),
               backgroundColor: 'rgba(240,192,64,0.7)', borderRadius: 4, yAxisID: 'y1' }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
            y: {{ beginAtZero:true, max:100, position:'right', ticks:{{callback:v=>v+'%'}},
                  title:{{display:true,text:'הסתברות (%)'}} }},
            y1: {{ beginAtZero:true, position:'left', grid:{{display:false}},
                   title:{{display:true,text:'P:F Ratio'}} }}
        }}
    }}
}});

predict();
</script>
</body>
</html>
'''

with open('GoldNP_Optimizer.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"    Generated GoldNP_Optimizer.html ({os.path.getsize('GoldNP_Optimizer.html')/1024:.0f} KB)")
print(f"    Pred grid: {len(pred_grid)} combos")
print("\n" + "=" * 60)
print("  DONE!")
print("=" * 60)
