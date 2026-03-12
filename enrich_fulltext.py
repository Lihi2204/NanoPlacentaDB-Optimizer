#!/usr/bin/env python3
"""
Fetch PMC full text for Au and liposome papers with PMCIDs
and re-extract fields with more detail.
"""
import csv, re, json, sys, io, time, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NM = 'not_mentioned'
BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

# Read enriched DB and find Au/Lipo papers with PMCIDs and low field coverage
rows = []
with open('DB_enriched.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    rows = list(reader)

# Find papers that could benefit from full text
targets = []
for i, row in enumerate(rows):
    pmcid = row.get('pmcid', '').strip()
    if not pmcid or pmcid == NM:
        continue
    mat = row.get('core_material', '')
    if mat not in ('Au', 'liposome', 'LNP', 'lipid_NP'):
        continue
    # Check how many fields are not_mentioned
    nm_count = sum(1 for c in cols if row.get(c, '') == NM)
    if nm_count >= 15:  # At least 15 missing fields - worth fetching
        targets.append((i, row, pmcid))

print(f"Found {len(targets)} Au/Lipo papers with PMCIDs needing enrichment")

# Fetch and re-extract (limit to 40 to stay within API limits)
targets = targets[:40]
enriched_count = 0

for idx, (row_idx, row, pmcid) in enumerate(targets):
    # Normalize PMCID
    pmcid_clean = pmcid.replace('PMC', '').strip()

    url = f"{BASE}/efetch.fcgi?db=pmc&id={pmcid_clean}&rettype=xml&retmode=xml"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=20)
        xml = resp.read().decode('utf-8', errors='replace')

        # Quick check: verify this is the right paper
        title_in_xml = ''
        m = re.search(r'<article-title>(.*?)</article-title>', xml, re.S)
        if m:
            title_in_xml = re.sub(r'<[^>]+>', '', m.group(1)).strip().lower()

        row_title = row.get('notes', '').lower()[:50]
        # Simple check - at least some title words match
        title_words = set(w for w in row_title.split() if len(w) > 4)
        xml_words = set(w for w in title_in_xml.split() if len(w) > 4)
        if title_words and xml_words:
            overlap = len(title_words & xml_words) / max(len(title_words), 1)
            if overlap < 0.3:
                print(f"  SKIP {pmcid} - title mismatch (overlap={overlap:.1%})")
                time.sleep(0.4)
                continue

        # Extract text from XML
        # Get methods section
        methods_text = ''
        m = re.search(r'<sec[^>]*>.*?(?:method|material|experiment).*?</sec>', xml, re.S | re.I)
        if m:
            methods_text = re.sub(r'<[^>]+>', ' ', m.group(0))

        # Get results section
        results_text = ''
        m = re.search(r'<sec[^>]*>.*?(?:result|finding).*?</sec>', xml, re.S | re.I)
        if m:
            results_text = re.sub(r'<[^>]+>', ' ', m.group(0))

        # Full body text
        body = re.sub(r'<[^>]+>', ' ', xml)
        fulltext = methods_text + ' ' + results_text + ' ' + body[:20000]
        ft = fulltext.lower()

        updated = False

        # Extract size_nm if missing
        if row.get('size_nm', NM) == NM:
            for pat in [
                r'(?:diameter|size|hydrodynamic|DLS|dh)\s*(?:of|=|:|was)\s*(?:approximately\s*|~)?(\d+(?:\.\d+)?)\s*(?:\xb1\s*\d+(?:\.\d+)?)?\s*nm',
                r'(\d+(?:\.\d+)?)\s*(?:\xb1\s*\d+(?:\.\d+)?)?\s*nm\s*(?:in diameter|in size|diameter)',
            ]:
                m = re.search(pat, fulltext, re.I)
                if m:
                    v = float(m.group(1))
                    if 0.5 <= v <= 10000:
                        rows[row_idx]['size_nm'] = str(v)
                        updated = True
                        break

        # Size method
        if row.get('size_method', NM) == NM:
            methods = []
            if re.search(r'\bdls\b|dynamic light scattering', ft): methods.append('DLS')
            if re.search(r'\btem\b|transmission electron', ft): methods.append('TEM')
            if re.search(r'\bnta\b|nanoparticle tracking', ft): methods.append('NTA')
            if methods:
                rows[row_idx]['size_method'] = '_'.join(methods)
                updated = True

        # Zeta potential
        if row.get('zeta_potential_mV', NM) == NM:
            m = re.search(r'zeta.{0,30}([+-]?\s*\d+(?:\.\d+)?)\s*(?:\xb1\s*\d+(?:\.\d+)?)?\s*mv', ft)
            if m:
                val = m.group(1).replace(' ', '')
                rows[row_idx]['zeta_potential_mV'] = val
                try:
                    v = float(val)
                    rows[row_idx]['surface_charge_cat'] = 'positive' if v > 10 else ('negative' if v < -10 else 'neutral')
                except:
                    pass
                updated = True

        # Dose
        if row.get('dose_value', NM) == NM:
            m = re.search(r'(\d+(?:\.\d+)?)\s*(?:\xb1\s*\d+(?:\.\d+)?)?\s*mg\s*/\s*kg', ft)
            if m:
                rows[row_idx]['dose_value'] = m.group(1)
                rows[row_idx]['dose_unit'] = 'mg_kg'
                updated = True
            else:
                m = re.search(r'(\d+(?:\.\d+)?)\s*(?:\xb1\s*\d+(?:\.\d+)?)?\s*(?:ug|µg)\s*/\s*(?:ml|mL)', ft)
                if m:
                    rows[row_idx]['dose_value'] = m.group(1)
                    rows[row_idx]['dose_unit'] = 'ug_mL'
                    updated = True

        # Gestational stage
        if row.get('gestational_stage', NM) == NM:
            m = re.search(r'(?:gd|gestational day|embryonic day|e)\s*(\d+(?:\.\d+)?)', ft)
            if m:
                rows[row_idx]['gestational_stage'] = f"GD{m.group(1)}"
                updated = True
            elif re.search(r'\bterm\b.{0,10}placent', ft):
                rows[row_idx]['gestational_stage'] = 'term'
                updated = True

        # Exposure route
        if row.get('exposure_route', NM) == NM:
            if re.search(r'intraven|i\.v\.|tail vein|intravenous', ft):
                rows[row_idx]['exposure_route'] = 'IV'
                updated = True
            elif re.search(r'\boral\b|gavage', ft):
                rows[row_idx]['exposure_route'] = 'oral'
                updated = True

        # Duration
        if row.get('exposure_duration_h', NM) == NM:
            m = re.search(r'(\d+)\s*(?:h|hr|hour)(?:s)?(?:\s|$|\.)', ft)
            if m:
                rows[row_idx]['exposure_duration_h'] = m.group(1)
                updated = True

        # Translocation percentage
        if row.get('translocation_pct', NM) == NM:
            m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:translocat|transfer|transport|cross)', ft)
            if not m:
                m = re.search(r'(?:translocat|transfer).{0,30}(\d+(?:\.\d+)?)\s*%', ft)
            if m:
                rows[row_idx]['translocation_pct'] = m.group(1)
                rows[row_idx]['translocation_detected'] = 'TRUE'
                updated = True

        # Placental uptake
        if row.get('placental_uptake_value', NM) == NM:
            m = re.search(r'placent.{0,20}(\d+(?:\.\d+)?)\s*(?:ug|µg|ng)\s*/\s*g', ft)
            if m:
                rows[row_idx]['placental_uptake_value'] = m.group(1)
                rows[row_idx]['placental_uptake_unit'] = 'ug_g'
                rows[row_idx]['placental_accumulation'] = 'TRUE'
                updated = True

        # Cell lines
        if row.get('cell_line_maternal', NM) == NM:
            if re.search(r'bewo', ft):
                rows[row_idx]['cell_line_maternal'] = 'BeWo'
                updated = True
            elif re.search(r'jeg.?3', ft):
                rows[row_idx]['cell_line_maternal'] = 'JEG-3'
                updated = True
            elif re.search(r'htr.?8', ft):
                rows[row_idx]['cell_line_maternal'] = 'HTR-8/SVneo'
                updated = True

        # Detection method enrichment
        if row.get('detection_method', NM) == NM:
            methods = []
            if re.search(r'icp.ms|icp.oes', ft): methods.append('ICP-MS')
            if re.search(r'fluorescen|confocal', ft): methods.append('fluorescence')
            if re.search(r'histolog|h.e|immunohistochem', ft): methods.append('histology')
            if re.search(r'dark.field', ft): methods.append('dark_field')
            if re.search(r'\bct\b.{0,10}imag|micro.ct', ft): methods.append('CT')
            if methods:
                rows[row_idx]['detection_method'] = '_'.join(methods)
                updated = True

        # Shape
        if row.get('shape', NM) == NM:
            if re.search(r'spheric|sphere', ft):
                rows[row_idx]['shape'] = 'spherical'
                updated = True
            elif re.search(r'\brod\b|nanorod', ft):
                rows[row_idx]['shape'] = 'rod'
                updated = True

        # Coating enrichment
        if row.get('surface_coating', NM) == NM:
            coatings = []
            if re.search(r'\bpeg\b|pegylat|mpeg', ft): coatings.append('PEG')
            if re.search(r'\bcooh\b|carboxyl', ft): coatings.append('COOH')
            if re.search(r'citrate', ft): coatings.append('citrate')
            if re.search(r'chitosan', ft): coatings.append('chitosan')
            if coatings:
                rows[row_idx]['surface_coating'] = '_'.join(coatings)
                updated = True

        # Maternal organs
        if row.get('maternal_organ_distribution', NM) == NM:
            organs = []
            if re.search(r'liver|hepat', ft) and re.search(r'distribut|accumul|biodistribut|organ', ft):
                organs.append('liver')
            if re.search(r'spleen', ft) and re.search(r'distribut|accumul', ft):
                organs.append('spleen')
            if re.search(r'kidney|renal', ft) and re.search(r'distribut|accumul', ft):
                organs.append('kidney')
            if organs:
                rows[row_idx]['maternal_organ_distribution'] = '_'.join(organs)
                updated = True

        if updated:
            rows[row_idx]['confidence'] = 'medium'  # upgrade from low
            enriched_count += 1
            print(f"  Enriched {pmcid} ({row.get('study_id','')}) - {row.get('core_material','')}")

        time.sleep(0.4)

    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode()
        print(f"  Error for {pmcid}: {err}")
        time.sleep(0.4)

    if (idx + 1) % 10 == 0:
        print(f"  Progress: {idx+1}/{len(targets)}")

print(f"\nEnriched {enriched_count}/{len(targets)} papers from PMC full text")

# Write updated DB
with open('DB_enriched.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print("Updated DB_enriched.csv")
