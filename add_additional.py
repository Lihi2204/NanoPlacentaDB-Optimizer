#!/usr/bin/env python3
"""Add additional relevant papers to DB_enriched.csv"""
import json, csv, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NM = 'not_mentioned'

def extract_surname(authors_str):
    if not authors_str:
        return 'Unknown'
    first = authors_str.split(';')[0].strip()
    if ',' in first:
        return first.split(',')[0].strip()
    parts = first.split()
    if not parts:
        return 'Unknown'
    if len(parts[0].replace('.', '')) <= 2:
        return parts[-1]
    return parts[0]

def detect_material(text):
    t = text.lower()
    if re.search(r'\bgold\b|aunp|au np|au-np|gnp|\bau\b.{0,10}nano|nanorod.{0,10}gold|gold.{0,10}nanorod|nanostar|nanocluster.{0,10}gold|gold.{0,10}nanocluster', t):
        return 'Au', 'inorganic_metal'
    if re.search(r'\bsilver\b|agnp|ag np', t):
        return 'Ag', 'inorganic_metal'
    if re.search(r'liposom', t):
        return 'liposome', 'lipid_based'
    if re.search(r'lipid nanoparticle|lipid-nanoparticle|\blnp\b', t):
        return 'LNP', 'lipid_based'
    if re.search(r'\bplga\b', t):
        return 'PLGA', 'polymer'
    if re.search(r'polystyrene', t):
        return 'polystyrene', 'polymer'
    if re.search(r'tio2|titanium', t):
        return 'TiO2', 'inorganic_metal_oxide'
    if re.search(r'sio2|silica.nano', t):
        return 'SiO2', 'inorganic_metal_oxide'
    if re.search(r'nanoparticle|nano.particle', t):
        return 'NP_unspecified', 'unspecified'
    return NM, NM

def extract_row(paper):
    text = (paper.get('title', '') or '') + ' ' + (paper.get('abstract', '') or '')
    t = text.lower()
    surname = extract_surname(paper.get('authors', ''))
    year = paper.get('year', '')
    sid = f"{surname}_{year}" if surname != 'Unknown' else f"PMID{paper.get('pmid', '')}_{year}"
    core_mat, mat_class = detect_material(text)

    # Model type
    model = NM
    if re.search(r'in.vivo|mice|mouse|rat|murine', t):
        model = 'in_vivo'
    elif re.search(r'in.vitro|cell.culture|transwell|bewo|jeg', t):
        model = 'in_vitro'
    elif re.search(r'ex.vivo|perfusion', t):
        model = 'ex_vivo'

    # Species
    species = NM
    if re.search(r'\bmouse\b|\bmice\b', t):
        species = 'mouse'
    elif re.search(r'\brat\b', t):
        species = 'rat'
    elif re.search(r'\bhuman\b|bewo|jeg|patient', t):
        species = 'human'

    # Size
    size = NM
    m = re.search(r'(\d+(?:\.\d+)?)\s*nm', text)
    if m:
        v = float(m.group(1))
        if 0.5 <= v <= 10000:
            size = str(v)

    # Charge
    charge_cat, zeta = NM, NM
    m = re.search(r'zeta.{0,20}([+-]?\d+(?:\.\d+)?)\s*mv', t)
    if m:
        val = float(m.group(1))
        charge_cat = 'positive' if val > 10 else ('negative' if val < -10 else 'neutral')
        zeta = str(val)

    # Route
    route = NM
    if re.search(r'intraven|i\.v\.|tail vein', t):
        route = 'IV'
    elif re.search(r'\boral\b|gavage', t):
        route = 'oral'

    # Translocation
    transloc = NM
    if re.search(r'translocat|cross.{0,10}placent|fetal.{0,10}transfer', t):
        transloc = 'FALSE' if re.search(r'no.{0,10}translocat|did not cross', t) else 'TRUE'

    # Cytotoxicity
    cyto = NM
    if re.search(r'no.{0,10}cytotox|non.toxic|biocompat|safe', t):
        cyto = 'FALSE'
    elif re.search(r'cytotox|toxic|cell death', t):
        cyto = 'TRUE'

    # Coating
    coating = NM
    if re.search(r'\bpeg\b|pegylat|mpeg', t):
        coating = 'PEG'
    elif re.search(r'\bcooh\b|carboxyl', t):
        coating = 'COOH'
    elif re.search(r'citrate', t):
        coating = 'citrate'

    # Detection
    detection = NM
    if re.search(r'icp.ms|icp.oes', t):
        detection = 'ICP-MS'
    elif re.search(r'fluorescen|confocal', t):
        detection = 'fluorescence'
    elif re.search(r'\btem\b', t):
        detection = 'TEM'

    # Gestational stage
    gest = NM
    m2 = re.search(r'(?:gd|gestational day|e)\s*(\d+)', t)
    if m2:
        gest = f"GD{m2.group(1)}"
    elif re.search(r'\bterm\b.{0,10}placent', t):
        gest = 'term'

    # Targeting
    targeting = 'none'
    if re.search(r'cgkrk', t):
        targeting = 'CGKRK'
    elif re.search(r'irgd', t):
        targeting = 'iRGD'
    elif re.search(r'folate|folic', t):
        targeting = 'folate'

    # Payload
    payload = 'none'
    if re.search(r'sirna', t):
        payload = 'siRNA'
    elif re.search(r'mrna\b', t):
        payload = 'mRNA'
    elif re.search(r'doxorubicin', t):
        payload = 'doxorubicin'

    # Indirect effects
    indirect = NM
    if re.search(r'fetal growth restrict|fgr|iugr', t):
        indirect = 'FGR'
    elif re.search(r'preeclamp', t):
        indirect = 'preeclampsia'

    return {
        'study_id': sid,
        'doi': paper.get('doi', '') or NM,
        'pmid': paper.get('pmid', '') or NM,
        'pmcid': paper.get('pmcid', '') or NM,
        'year': year or NM,
        'source': 'pubmed_enrichment',
        'confidence': 'medium' if paper.get('pmcid') else 'low',
        'notes': (paper.get('title', '') or '')[:120],
        'core_material': core_mat,
        'material_class': mat_class,
        'size_nm': size,
        'size_method': NM,
        'size_in_medium_nm': NM,
        'shape': NM,
        'surface_coating': coating,
        'targeting_ligand': targeting,
        'therapeutic_payload': payload,
        'surface_charge_cat': charge_cat,
        'zeta_potential_mV': zeta,
        'model_type': model,
        'cell_line_maternal': NM,
        'cell_line_fetal': NM,
        'species': species,
        'gestational_stage': gest,
        'exposure_route': route,
        'dose_value': NM,
        'dose_unit': NM,
        'exposure_duration_h': NM,
        'translocation_detected': transloc,
        'translocation_pct': NM,
        'placental_accumulation': NM,
        'placental_uptake_value': NM,
        'placental_uptake_unit': NM,
        'placenta_fetus_ratio': NM,
        'extraembryonic_uptake': NM,
        'fetal_distribution': NM,
        'maternal_organ_distribution': NM,
        'cytotoxicity_observed': cyto,
        'cytotoxicity_type': NM,
        'indirect_effects': indirect,
        'detection_method': detection,
    }


# Load additional relevant papers
with open('additional_relevant.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

# Read existing enriched DB
existing_pmids = set()
existing_ids = set()
with open('DB_enriched.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    db_cols = reader.fieldnames
    existing_rows = list(reader)
    for row in existing_rows:
        existing_pmids.add(row.get('pmid', '').strip())
        existing_ids.add(row.get('study_id', ''))

print(f"Current DB: {len(existing_rows)} rows")

new_rows = []
for p in papers:
    pmid = p.get('pmid', '')
    if pmid in existing_pmids:
        continue
    row = extract_row(p)
    sid = row['study_id']
    if sid in existing_ids:
        for suffix in 'bcdefghijklmnopqrstuvwxyz':
            candidate = f"{sid}{suffix}"
            if candidate not in existing_ids:
                row['study_id'] = candidate
                existing_ids.add(candidate)
                break
    else:
        existing_ids.add(sid)
    new_rows.append(row)
    existing_pmids.add(pmid)

print(f"New rows to add: {len(new_rows)}")

all_rows = existing_rows + new_rows
for row in all_rows:
    for c in db_cols:
        if c not in row:
            row[c] = 'not_mentioned'

with open('DB_enriched.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=db_cols)
    writer.writeheader()
    writer.writerows(all_rows)

unique_studies = set(r['study_id'] for r in all_rows)
au_count = sum(1 for r in all_rows if r.get('core_material', '') == 'Au')
lipo_count = sum(1 for r in all_rows if r.get('core_material', '') in ('liposome', 'LNP', 'lipid_NP'))

print(f"\nFinal DB_enriched.csv:")
print(f"  Total rows: {len(all_rows)}")
print(f"  Unique studies: {len(unique_studies)}")
print(f"  Gold NP rows: {au_count}")
print(f"  Liposome/LNP rows: {lipo_count}")
