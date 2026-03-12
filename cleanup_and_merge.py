#!/usr/bin/env python3
"""
Clean up extracted papers: reclassify NP_unspecified, remove false positives,
merge with existing DB.csv to create final DB_enriched.csv
"""

import json, csv, re, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load paper metadata for abstract access
papers_by_pmid = {}
for fname in ['new_papers_enriched.json', 'new_papers_found.json']:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            for p in json.load(f):
                papers_by_pmid[p.get('pmid', '')] = p
    except:
        pass

# ── Better material classification ─────────────────────────────────
def reclassify_material(text):
    """More thorough material classification for NP_unspecified papers."""
    t = text.lower()

    # Zein protein NP
    if re.search(r'zein', t):
        return 'zein', 'protein_based'
    # Albumin NP
    if re.search(r'albumin.{0,10}nano|nab.{0,3}paclitaxel', t):
        return 'albumin_NP', 'protein_based'
    # Silk
    if re.search(r'silk.{0,10}nano|silk fibroin', t):
        return 'silk', 'protein_based'
    # PAMAM dendrimer
    if re.search(r'pamam|dendrimer', t):
        return 'dendrimer', 'polymer'
    # Nanoemulsion
    if re.search(r'nanoemulsion|nano.emulsion', t):
        return 'nanoemulsion', 'lipid_based'
    # Nanocomplex / nanoconstruct (general)
    if re.search(r'nanocomplex|nano.complex', t):
        return 'nanocomplex', 'polymer'
    # siRNA nanoparticle
    if re.search(r'sirna.{0,20}nano|nano.{0,20}sirna|lipid.{0,10}sirna', t):
        return 'siRNA_NP', 'lipid_based'
    # Polymeric micelle
    if re.search(r'micelle', t):
        return 'micelle', 'polymer'
    # Hydroxyapatite
    if re.search(r'hydroxyapatit|nano.apatit', t):
        return 'hydroxyapatite', 'inorganic_ceramic'
    # Selenium NP
    if re.search(r'selenium.{0,10}nano|se.{0,5}nano|nano.{0,10}selenium', t):
        return 'Se', 'inorganic_metal'
    # Copper NP
    if re.search(r'copper.{0,10}nano|cu.{0,5}nano|cuo', t):
        return 'Cu', 'inorganic_metal'
    # Calcium phosphate
    if re.search(r'calcium phosphat|cap.{0,5}nano', t):
        return 'CaP', 'inorganic_ceramic'

    return None, None

# ── False positive detection ───────────────────────────────────────
def is_false_positive(text, title):
    """Detect papers that only incidentally mention nanoparticles."""
    t = text.lower()
    title_l = title.lower()

    # If NP/nano is NOT in the title and is only mentioned <2 times in abstract, likely false positive
    np_in_title = bool(re.search(r'nano|liposom|lnp|nanoparticle', title_l))
    np_mentions = len(re.findall(r'nano|liposom|lnp', t))

    # Papers about NTA (nanoparticle tracking analysis) as a METHOD, not studying NPs
    if re.search(r'nanoparticle tracking analysis|nta\)', t) and np_mentions <= 2 and not np_in_title:
        return True

    # Papers about exosomes/EVs where NP is just the methodology
    if re.search(r'exosom|extracellular vesicle', title_l) and not np_in_title:
        if np_mentions <= 3:
            return True

    # Papers about gene expression/signaling with incidental NP mention
    if not np_in_title and np_mentions <= 1:
        return True

    # Papers about environmental NP exposure that are purely toxicology without specific NP characterization
    # Keep these as they're still about NPs + placenta

    return False

# ── Process new_rows.csv ───────────────────────────────────────────
print("Processing extracted rows...")

cleaned_rows = []
removed = 0
reclassified = 0

with open('new_rows.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for row in reader:
        pmid = row.get('pmid', '')
        paper = papers_by_pmid.get(pmid, {})
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        text = title + ' ' + abstract

        # Check for false positive
        if is_false_positive(text, title):
            removed += 1
            continue

        # Reclassify NP_unspecified
        if row['core_material'] in ('NP_unspecified', 'not_mentioned'):
            new_mat, new_class = reclassify_material(text)
            if new_mat:
                row['core_material'] = new_mat
                row['material_class'] = new_class
                reclassified += 1

        cleaned_rows.append(row)

print(f"  Removed {removed} false positives")
print(f"  Reclassified {reclassified} NP_unspecified")
print(f"  Remaining: {len(cleaned_rows)} rows")

# ── Read existing DB ───────────────────────────────────────────────
existing_rows = []
existing_pmids = set()
with open('DB.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    db_cols = reader.fieldnames
    for row in reader:
        existing_rows.append(row)
        existing_pmids.add(row.get('pmid', '').strip())

# Deduplicate against existing PMIDs
final_new = [r for r in cleaned_rows if r.get('pmid', '') not in existing_pmids]
print(f"  After PMID dedup vs existing DB: {len(final_new)} new rows")

# Fix study_id collisions
existing_ids = set(r.get('study_id','') for r in existing_rows)
for row in final_new:
    sid = row['study_id']
    if sid in existing_ids:
        base = sid
        for suffix in 'bcdefghijklmnopqrstuvwxyz':
            candidate = f'{base}{suffix}'
            if candidate not in existing_ids:
                row['study_id'] = candidate
                existing_ids.add(candidate)
                break
    else:
        existing_ids.add(sid)

# ── Merge ──────────────────────────────────────────────────────────
all_rows = existing_rows + final_new

# Ensure all rows have all columns
for row in all_rows:
    for c in db_cols:
        if c not in row or row[c] == '':
            row[c] = row.get(c, 'not_mentioned')

# Write merged DB
with open('DB_enriched.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=db_cols)
    writer.writeheader()
    writer.writerows(all_rows)

# ── Final statistics ───────────────────────────────────────────────
unique_studies = set(r['study_id'] for r in all_rows)
unique_pmids = set(r['pmid'] for r in all_rows if r.get('pmid'))
print(f"\n=== FINAL DB_enriched.csv ===")
print(f"Total rows: {len(all_rows)}")
print(f"Unique studies: {len(unique_studies)}")
print(f"Unique PMIDs: {len(unique_pmids)}")
print(f"  Existing: {len(existing_rows)} rows")
print(f"  New: {len(final_new)} rows")

from collections import Counter

# Material breakdown
mats = Counter(r.get('core_material','') for r in all_rows)
print(f"\nMaterial distribution:")
for m, c in mats.most_common(25):
    print(f"  {m}: {c}")

# Model types
models = Counter(r.get('model_type','') for r in all_rows)
print(f"\nModel types:")
for m, c in models.most_common():
    print(f"  {m}: {c}")

# Species
species = Counter(r.get('species','') for r in all_rows)
print(f"\nSpecies:")
for s, c in species.most_common(10):
    print(f"  {s}: {c}")

# Year range
years = sorted(set(r.get('year','') for r in all_rows if r.get('year','').isdigit()))
print(f"\nYear range: {years[0]} - {years[-1]}")

# Gold + Liposome specific counts
au_count = sum(1 for r in all_rows if r.get('core_material','') == 'Au')
lipo_count = sum(1 for r in all_rows if r.get('core_material','') in ('liposome', 'LNP', 'lipid_NP'))
print(f"\nGold NP rows: {au_count}")
print(f"Liposome/LNP rows: {lipo_count}")

# Confidence
confs = Counter(r.get('confidence','') for r in all_rows)
print(f"\nConfidence levels:")
for c, n in confs.most_common():
    print(f"  {c}: {n}")
