#!/usr/bin/env python3
"""
Extract all 41 DB fields from paper abstracts for NanoPlacentaDB enrichment.
Processes papers from new_papers_enriched.json and new_papers_found.json.
Filters for actual placenta/pregnancy + nanoparticle papers.
Outputs new_rows.csv with extracted data.
"""

import json, csv, re, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Column definitions ──────────────────────────────────────────────
COLUMNS = [
    'study_id','doi','pmid','pmcid','year','source','confidence','notes',
    'core_material','material_class','size_nm','size_method','size_in_medium_nm',
    'shape','surface_coating','targeting_ligand','therapeutic_payload',
    'surface_charge_cat','zeta_potential_mV','model_type','cell_line_maternal',
    'cell_line_fetal','species','gestational_stage','exposure_route',
    'dose_value','dose_unit','exposure_duration_h','translocation_detected',
    'translocation_pct','placental_accumulation','placental_uptake_value',
    'placental_uptake_unit','placenta_fetus_ratio','extraembryonic_uptake',
    'fetal_distribution','maternal_organ_distribution','cytotoxicity_observed',
    'cytotoxicity_type','indirect_effects','detection_method'
]

NM = "not_mentioned"

# ── Helper: extract first author surname ────────────────────────────
def extract_surname(authors_str):
    if not authors_str:
        return "Unknown"
    first = authors_str.split(';')[0].strip()
    # Handle "Surname, I." or "Surname I." or "I. Surname"
    if ',' in first:
        return first.split(',')[0].strip()
    parts = first.split()
    if not parts:
        return "Unknown"
    # If first part is initial (1-2 chars with optional dot), surname is last
    if len(parts[0].replace('.','')) <= 2:
        return parts[-1]
    return parts[0]

# ── Keyword patterns ────────────────────────────────────────────────
PLACENTA_RE = re.compile(
    r'placent|pregnan|gestat|trophoblast|fetal|fetus|maternal[\s-]fetal|'
    r'transplacent|embryo|preeclamp|pre[\s-]eclamp|chorion|decidua|'
    r'uterine|intrauterine|amnio|yolk.sac|umbilical', re.I)

NP_RE = re.compile(
    r'nanoparticle|nano[\s-]particle|gold[\s-]nano|AuNP|GNP|liposom|'
    r'lipid[\s-]nano|LNP|nanocarrier|nano[\s-]carrier|nanomed|PLGA|'
    r'polymer[\s-]nano|silver[\s-]nano|AgNP|quantum[\s-]dot|nanotube|'
    r'nanocrystal|nano[\s-]formul|nanostructur|nanosphere|dendrim|micelle|'
    r'nanocomposit|nano[\s-]emulsion|exosom|vesicle|nanogel|'
    r'iron[\s-]oxide|SPION|IONP|silica[\s-]nano|SiO2|TiO2|ZnO|CeO2|'
    r'carbon[\s-]nano|fullerene|graphene|nano[\s-]rod|nano[\s-]tube', re.I)

# ── Material detection ──────────────────────────────────────────────
def detect_material(text):
    """Return (core_material, material_class) from text."""
    t = text.lower()

    # Gold
    if re.search(r'\bgold\b|aunp|au np|au-np|gnp|\bau\b.{0,10}nano', t):
        return 'Au', 'inorganic_metal'
    # Silver
    if re.search(r'\bsilver\b|agnp|ag np|ag-np', t):
        return 'Ag', 'inorganic_metal'
    # Liposome
    if re.search(r'liposom', t):
        return 'liposome', 'lipid_based'
    # LNP
    if re.search(r'lipid nanoparticle|lipid-nanoparticle|\blnp\b', t):
        return 'LNP', 'lipid_based'
    # PLGA
    if re.search(r'\bplga\b|poly.lactic.co.glycol', t):
        return 'PLGA', 'polymer'
    # Polystyrene
    if re.search(r'polystyrene|\bps\b.{0,5}nano|\bps-np\b', t):
        return 'polystyrene', 'polymer'
    # TiO2
    if re.search(r'tio2|titanium.dioxide|titanium.nano', t):
        return 'TiO2', 'inorganic_metal_oxide'
    # SiO2
    if re.search(r'sio2|silica.nano|mesoporous.silica', t):
        return 'SiO2', 'inorganic_metal_oxide'
    # ZnO
    if re.search(r'zno|zinc.oxide.nano', t):
        return 'ZnO', 'inorganic_metal_oxide'
    # Iron oxide / SPION
    if re.search(r'iron.oxide|spion|ionp|fe3o4|fe2o3|magnetite|superparamagnetic', t):
        return 'iron_oxide', 'inorganic_metal_oxide'
    # CeO2
    if re.search(r'ceo2|cerium|nanoceria', t):
        return 'CeO2', 'inorganic_metal_oxide'
    # Carbon-based
    if re.search(r'carbon.nano|swcnt|mwcnt|nanotube|fullerene|c60|graphene', t):
        return 'carbon', 'carbon_based'
    # Quantum dots
    if re.search(r'quantum.dot|\bqd\b|qdot|cdse|cdte', t):
        return 'quantum_dot', 'inorganic_semiconductor'
    # Dendrimer
    if re.search(r'dendrim|pamam', t):
        return 'dendrimer', 'polymer'
    # Chitosan
    if re.search(r'chitosan', t):
        return 'chitosan', 'polymer'
    # Exosome/extracellular vesicle
    if re.search(r'exosom|extracellular.vesicle', t):
        return 'exosome', 'biological'
    # Generic polymer NP
    if re.search(r'polymer.{0,5}nano|polymeric.nano', t):
        return 'polymer_NP', 'polymer'
    # Generic lipid
    if re.search(r'lipid|lipoplex|solid.lipid', t):
        return 'lipid_NP', 'lipid_based'
    # Albumin
    if re.search(r'albumin.nano|nab-|abraxane', t):
        return 'albumin', 'protein_based'
    # Generic NP
    if re.search(r'nanoparticle|nano.particle', t):
        return 'NP_unspecified', 'unspecified'
    return NM, NM

# ── Size extraction ─────────────────────────────────────────────────
def extract_size(text):
    """Extract size in nm from text."""
    # Look for patterns like "X nm", "X-Y nm", "diameter of X nm"
    patterns = [
        r'(?:diameter|size|hydrodynamic diameter|mean diameter|average diameter|DLS|dh)\s*(?:of|=|:|\s)\s*(?:approximately\s*|~\s*|about\s*)?(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*nm',
        r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*nm\s*(?:in diameter|in size|diameter|size|particles|nanoparticle)',
        r'(\d+(?:\.\d+)?)\s*nm\s*(?:Au|gold|silver|Ag|liposom|LNP|PLGA|nanoparticle|NP)',
        r'(?:Au|gold|silver|liposom|LNP|PLGA|NP)\s*(?:of|with)?\s*(\d+(?:\.\d+)?)\s*nm',
        r'(\d{1,4}(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*nm',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = float(m.group(1))
            if 0.5 <= val <= 10000:  # reasonable NP size range
                return str(val)
    return NM

def extract_size_method(text):
    t = text.lower()
    methods = []
    if re.search(r'\bdls\b|dynamic light scattering', t):
        methods.append('DLS')
    if re.search(r'\btem\b|transmission electron microscop', t):
        methods.append('TEM')
    if re.search(r'\bnta\b|nanoparticle tracking', t):
        methods.append('NTA')
    if re.search(r'\bsem\b|scanning electron microscop', t):
        methods.append('SEM')
    if re.search(r'\bafm\b|atomic force microscop', t):
        methods.append('AFM')
    return '_'.join(methods) if methods else NM

# ── Shape extraction ────────────────────────────────────────────────
def extract_shape(text):
    t = text.lower()
    if re.search(r'spheric|sphere', t): return 'spherical'
    if re.search(r'\brod\b|nanorod', t): return 'rod'
    if re.search(r'cube|nanocube', t): return 'cubic'
    if re.search(r'star|nanostar', t): return 'star'
    if re.search(r'wire|nanowire', t): return 'wire'
    if re.search(r'plate|nanoplate|disc|nanodisk', t): return 'plate'
    if re.search(r'triangle|nanoprism', t): return 'triangular'
    return NM

# ── Surface coating ─────────────────────────────────────────────────
def extract_coating(text):
    t = text.lower()
    coatings = []
    if re.search(r'\bpeg\b|polyethylene glycol|pegylat|mpeg', t): coatings.append('PEG')
    if re.search(r'\bcooh\b|carboxyl', t): coatings.append('COOH')
    if re.search(r'\bnh2\b|amine|amino', t): coatings.append('NH2')
    if re.search(r'citrate', t): coatings.append('citrate')
    if re.search(r'chitosan', t): coatings.append('chitosan')
    if re.search(r'pvp|polyvinylpyrrolidone', t): coatings.append('PVP')
    if re.search(r'pva|polyvinyl alcohol', t): coatings.append('PVA')
    if re.search(r'bsa|bovine serum albumin', t): coatings.append('BSA')
    if re.search(r'transferrin', t): coatings.append('transferrin')
    if re.search(r'folate|folic acid', t): coatings.append('folate')
    if re.search(r'antibod|immunoglobulin|\bIgG\b', t): coatings.append('antibody')
    if re.search(r'dextran', t): coatings.append('dextran')
    if re.search(r'starch', t): coatings.append('starch')
    if re.search(r'silica|sio2 coat', t): coatings.append('silica')
    if re.search(r'lipid coat|lipid shell|phospholipid coat', t): coatings.append('lipid')
    return '_'.join(coatings) if coatings else NM

# ── Targeting ligand ────────────────────────────────────────────────
def extract_targeting(text):
    t = text.lower()
    ligands = []
    if re.search(r'cgkrk', t): ligands.append('CGKRK')
    if re.search(r'irgd', t): ligands.append('iRGD')
    if re.search(r'rgd\b', t): ligands.append('RGD')
    if re.search(r'folate|folic acid', t): ligands.append('folate')
    if re.search(r'transferrin', t): ligands.append('transferrin')
    if re.search(r'insulin\b', t): ligands.append('insulin')
    if re.search(r'glucose', t) and re.search(r'target|ligand|conjugat|decorat', t): ligands.append('glucose')
    if re.search(r'aptamer', t): ligands.append('aptamer')
    if re.search(r'antibod|scfv|fab\b|nanobod', t) and re.search(r'target|conjugat|decorat', t): ligands.append('antibody')
    if re.search(r'peptide.{0,15}target|target.{0,15}peptide|homing peptide', t): ligands.append('peptide')
    if re.search(r'hyaluronic acid|ha-', t) and re.search(r'target', t): ligands.append('hyaluronic_acid')
    return '_'.join(ligands) if ligands else 'none'

# ── Therapeutic payload ─────────────────────────────────────────────
def extract_payload(text):
    t = text.lower()
    payloads = []
    if re.search(r'doxorubicin|dox\b', t): payloads.append('doxorubicin')
    if re.search(r'paclitaxel|taxol', t): payloads.append('paclitaxel')
    if re.search(r'sirna|si-rna|small interfering rna', t): payloads.append('siRNA')
    if re.search(r'mrna\b', t): payloads.append('mRNA')
    if re.search(r'mirna|microrna', t): payloads.append('miRNA')
    if re.search(r'crispr|cas9|cas-9', t): payloads.append('CRISPR')
    if re.search(r'insulin\b', t) and re.search(r'deliver|encapsulat|load|payload|cargo', t): payloads.append('insulin')
    if re.search(r'metformin', t): payloads.append('metformin')
    if re.search(r'dexamethasone', t): payloads.append('dexamethasone')
    if re.search(r'curcumin', t): payloads.append('curcumin')
    if re.search(r'anakinra|il-1r', t): payloads.append('anakinra')
    if re.search(r'igf|growth factor', t) and re.search(r'deliver|encapsulat', t): payloads.append('growth_factor')
    if re.search(r'antioxidant', t) and re.search(r'deliver|encapsulat|load', t): payloads.append('antioxidant')
    if re.search(r'aspirin|acetylsalicylic', t): payloads.append('aspirin')
    if re.search(r'heparin', t): payloads.append('heparin')
    if re.search(r'progesterone', t): payloads.append('progesterone')
    if re.search(r'oxytocin', t): payloads.append('oxytocin')
    if re.search(r'plasmid|dna\b|pdna', t) and re.search(r'deliver|encapsulat|transfect', t): payloads.append('pDNA')
    return '_'.join(payloads) if payloads else 'none'

# ── Charge detection ────────────────────────────────────────────────
def extract_charge(text):
    t = text.lower()
    # Look for zeta potential value
    m = re.search(r'zeta.{0,20}(?:potential)?.{0,10}(?:of|=|:|\s)\s*([+-]?\s*\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*mv', t)
    if m:
        val_str = m.group(1).replace(' ', '')
        try:
            val = float(val_str)
            cat = 'positive' if val > 10 else ('negative' if val < -10 else 'neutral')
            return cat, str(val)
        except:
            pass

    # Text-based charge detection
    if re.search(r'positive.{0,10}charge|cation|positively charged', t):
        return 'positive', NM
    if re.search(r'negative.{0,10}charge|anion|negatively charged', t):
        return 'negative', NM
    if re.search(r'neutral.{0,10}charge|uncharged|zwitterion', t):
        return 'neutral', NM
    return NM, NM

# ── Model type detection ───────────────────────────────────────────
def extract_model(text):
    t = text.lower()
    models = []
    if re.search(r'in.vivo|mice|mouse|rat|rabbit|animal model|murine|rodent|pregnant.{0,20}(?:mice|mouse|rat|rabbit)', t):
        models.append('in_vivo')
    if re.search(r'in.vitro|cell.culture|cultured cells|well.plate|transwell', t):
        models.append('in_vitro')
    if re.search(r'ex.vivo|perfusion|placental perfusion|tissue explant|explant culture', t):
        models.append('ex_vivo')
    if re.search(r'in.silico|computational|simulation|molecular dynamics|docking', t):
        models.append('in_silico')
    if re.search(r'bewo|jeg|jar\b|httr|swan|htr-8', t):
        models.append('in_vitro')
    return '_'.join(sorted(set(models))) if models else NM

# ── Cell line detection ─────────────────────────────────────────────
def extract_cell_lines(text):
    t = text.lower()
    maternal = []
    fetal = []
    if re.search(r'bewo', t): maternal.append('BeWo')
    if re.search(r'jeg-?3', t): maternal.append('JEG-3')
    if re.search(r'jar\b', t): maternal.append('JAR')
    if re.search(r'htr-?8|swan[\s-]?71', t): maternal.append('HTR-8/SVneo')
    if re.search(r'huvec', t): fetal.append('HUVEC')
    if re.search(r'hpec|a2 cell|hfpec', t): fetal.append('HPEC-A2')
    if re.search(r'human placental|primary trophoblast', t): maternal.append('primary_trophoblast')
    mat = '_'.join(maternal) if maternal else NM
    fet = '_'.join(fetal) if fetal else NM
    return mat, fet

# ── Species detection ───────────────────────────────────────────────
def extract_species(text):
    t = text.lower()
    species = []
    if re.search(r'\bmouse\b|\bmice\b|\bmurine\b', t): species.append('mouse')
    if re.search(r'\brat\b|\brats\b', t): species.append('rat')
    if re.search(r'\bhuman\b|patient|women|clinical|bewo|jeg|htr-8', t): species.append('human')
    if re.search(r'\brabbit\b', t): species.append('rabbit')
    if re.search(r'\bsheep\b|\bovine\b|\bewe\b', t): species.append('sheep')
    if re.search(r'\bzebrafish\b', t): species.append('zebrafish')
    if re.search(r'\bguinea pig\b', t): species.append('guinea_pig')
    return '_'.join(species) if species else NM

# ── Gestational stage ──────────────────────────────────────────────
def extract_gestational_stage(text):
    t = text.lower()
    # Look for GD/ED patterns
    m = re.search(r'(?:gd|gestational day|embryonic day|ed|e)\s*(\d+(?:\.\d+)?)', t)
    if m:
        return f"GD{m.group(1)}"
    if re.search(r'first trimester|1st trimester', t): return 'first_trimester'
    if re.search(r'second trimester|2nd trimester', t): return 'second_trimester'
    if re.search(r'third trimester|3rd trimester', t): return 'third_trimester'
    if re.search(r'\bterm\b.{0,10}placent|term pregnan|at term|full.term', t): return 'term'
    if re.search(r'preterm|pre-term', t): return 'preterm'
    if re.search(r'early pregnan', t): return 'early_pregnancy'
    if re.search(r'late pregnan|late gestation', t): return 'late_pregnancy'
    if re.search(r'mid.pregnan|mid.gestation', t): return 'mid_pregnancy'
    return NM

# ── Exposure route ──────────────────────────────────────────────────
def extract_route(text):
    t = text.lower()
    if re.search(r'intraven|i\.v\.|iv inject|iv admin|tail vein|intravenous', t): return 'IV'
    if re.search(r'\boral\b|gavage|ingestion|drinking water', t): return 'oral'
    if re.search(r'inhalat|intranasal|pulmonary|aerosol|nebuliz', t): return 'inhalation'
    if re.search(r'intraperitoneal|i\.p\.', t): return 'IP'
    if re.search(r'subcutaneous|s\.c\.', t): return 'SC'
    if re.search(r'intramuscular|i\.m\.', t): return 'IM'
    if re.search(r'transdermal|topical|dermal', t): return 'dermal'
    if re.search(r'apical|basolateral|transwell', t): return 'apical_in_vitro'
    if re.search(r'maternal.{0,10}circuit|perfusion', t): return 'maternal_circuit'
    return NM

# ── Dose extraction ─────────────────────────────────────────────────
def extract_dose(text):
    t = text.lower()
    # mg/kg patterns
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*mg\s*/\s*kg', t)
    if m:
        return m.group(1), 'mg_kg'
    # ug/mL patterns
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*(?:ug|µg|μg)\s*/\s*(?:ml|mL)', t)
    if m:
        return m.group(1), 'ug_mL'
    # mg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*mg\s*/\s*(?:ml|mL)', t)
    if m:
        return m.group(1), 'mg_mL'
    # ug/kg
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*(?:ug|µg|μg)\s*/\s*kg', t)
    if m:
        return m.group(1), 'ug_kg'
    # nM
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*nm(?:ol)?(?:/l)?', t)
    if m:
        val = float(m.group(1))
        if val > 500:  # likely size in nm, not concentration
            pass
        else:
            return m.group(1), 'nM'
    # ppm
    m = re.search(r'(\d+(?:\.\d+)?)\s*ppm', t)
    if m:
        return m.group(1), 'ppm'
    return NM, NM

# ── Duration extraction ─────────────────────────────────────────────
def extract_duration(text):
    t = text.lower()
    # hours
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:h|hr|hour)(?:s)?(?:\s|$|\.)', t)
    if m:
        return m.group(1)
    # days -> hours
    m = re.search(r'(\d+)\s*(?:d|day)(?:s)?(?:\s+(?:exposure|treatment|incubat))', t)
    if m:
        return str(int(m.group(1)) * 24)
    # minutes -> hours
    m = re.search(r'(\d+)\s*min(?:ute)?(?:s)?(?:\s+(?:exposure|treatment|incubat))', t)
    if m:
        return str(round(int(m.group(1)) / 60, 2))
    return NM

# ── Translocation detection ────────────────────────────────────────
def extract_translocation(text):
    t = text.lower()
    detected = NM
    pct = NM

    if re.search(r'translocat|cross.{0,10}placent|fetal.{0,10}transfer|placental.{0,10}transfer|maternal.{0,10}fetal.{0,10}transfer|penetrat.{0,10}placent', t):
        if re.search(r'no translocat|did not cross|unable to cross|no.{0,10}transfer|prevent.{0,10}transfer|block.{0,10}transfer|not detect|no evidence|negligible', t):
            detected = 'FALSE'
        else:
            detected = 'TRUE'

    # Percentage
    m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:translocat|transfer|cross|transloc|transport)', t)
    if m:
        pct = m.group(1)
        detected = 'TRUE'
    m = re.search(r'(?:translocat|transfer).{0,30}(\d+(?:\.\d+)?)\s*%', t)
    if m:
        pct = m.group(1)
        detected = 'TRUE'

    return detected, pct

# ── Placental accumulation ──────────────────────────────────────────
def extract_accumulation(text):
    t = text.lower()
    if re.search(r'placental.{0,15}accumul|accumulat.{0,15}placent|placental.{0,10}uptake|retain.{0,15}placent|placental.{0,10}deposition|deposit.{0,15}placent|tropism.{0,10}placent|placental.{0,10}retention', t):
        return 'TRUE'
    if re.search(r'no.{0,10}accumul.{0,10}placent|no.{0,10}placental.{0,10}uptake|not.{0,10}retain.{0,10}placent', t):
        return 'FALSE'
    return NM

# ── Cytotoxicity ────────────────────────────────────────────────────
def extract_cytotoxicity(text):
    t = text.lower()
    observed = NM
    tox_type = NM

    if re.search(r'cytotox|toxicit|cell death|apoptosis|necrosis|viabilit|mtt|wst|ldhrelease', t):
        if re.search(r'no.{0,10}cytotox|non.{0,5}toxic|no.{0,10}toxicit|biocompat|no.{0,10}significant.{0,10}toxicit|low.{0,5}toxicit|safe|no adverse|no.{0,5}cell death', t):
            observed = 'FALSE'
            tox_type = 'none'
        else:
            observed = 'TRUE'
            types = []
            if re.search(r'apoptosis|apoptotic', t): types.append('apoptosis')
            if re.search(r'necrosis|necrotic', t): types.append('necrosis')
            if re.search(r'oxidative stress|ros\b|reactive oxygen', t): types.append('oxidative_stress')
            if re.search(r'inflammat|inflammatory|il-6|il-1|tnf|cytokine', t): types.append('inflammation')
            if re.search(r'dna damage|genotox', t): types.append('genotoxicity')
            tox_type = '_'.join(types) if types else 'cytotoxicity'

    return observed, tox_type

# ── Detection method ────────────────────────────────────────────────
def extract_detection(text):
    t = text.lower()
    methods = []
    if re.search(r'icp[\s-]?ms|icp[\s-]?oes|icp[\s-]?aes', t): methods.append('ICP-MS')
    if re.search(r'la[\s-]icp', t): methods.append('LA-ICP-MS')
    if re.search(r'fluorescen|confocal|facs|flow cytom', t): methods.append('fluorescence')
    if re.search(r'\btem\b|transmission electron', t): methods.append('TEM')
    if re.search(r'\bsem\b|scanning electron', t): methods.append('SEM')
    if re.search(r'dark.field|darkfield', t): methods.append('dark_field')
    if re.search(r'ct\s*imag|micro.ct|computed tomograph|ct scan', t): methods.append('CT')
    if re.search(r'mri\b|magnetic resonance imag', t): methods.append('MRI')
    if re.search(r'pet\b|positron emission', t): methods.append('PET')
    if re.search(r'spect\b', t): methods.append('SPECT')
    if re.search(r'raman|sers', t): methods.append('Raman')
    if re.search(r'elisa\b', t): methods.append('ELISA')
    if re.search(r'western blot', t): methods.append('western_blot')
    if re.search(r'qpcr|rt-pcr|real-time pcr', t): methods.append('qPCR')
    if re.search(r'histolog|h&e|immunohistochem|ihc\b', t): methods.append('histology')
    if re.search(r'uv[\s-]?vis|spectrophotom', t): methods.append('UV-Vis')
    if re.search(r'mass spec|ms/ms|lc-ms|hplc', t): methods.append('mass_spec')
    return '_'.join(methods) if methods else NM

# ── Indirect effects ────────────────────────────────────────────────
def extract_indirect_effects(text):
    t = text.lower()
    effects = []
    if re.search(r'fetal growth restrict|fgr\b|iugr\b|growth restrict', t): effects.append('FGR')
    if re.search(r'preterm.{0,10}(?:birth|deliver|labor)', t): effects.append('preterm_birth')
    if re.search(r'preeclamp|pre-eclamp|hypertens.{0,10}pregnan', t): effects.append('preeclampsia')
    if re.search(r'resorption|abortion|miscarr|fetal loss|fetal death', t): effects.append('fetal_loss')
    if re.search(r'placental insufficien|placental dysfunction', t): effects.append('placental_dysfunction')
    if re.search(r'inflammation|inflammat.{0,10}placent', t): effects.append('inflammation')
    if re.search(r'oxidative stress.{0,15}placent|placental.{0,15}oxidative', t): effects.append('oxidative_stress')
    if re.search(r'birth weight|low birth weight|small for gestational|sga\b', t): effects.append('low_birth_weight')
    if re.search(r'malformation|teratogen|birth defect|congenital', t): effects.append('malformation')
    return '_'.join(effects) if effects else NM

# ── Maternal organ distribution ─────────────────────────────────────
def extract_maternal_organs(text):
    t = text.lower()
    organs = []
    if re.search(r'liver|hepat', t) and re.search(r'distribut|accumul|organ|biodistribut', t): organs.append('liver')
    if re.search(r'spleen|splenic', t) and re.search(r'distribut|accumul|organ|biodistribut', t): organs.append('spleen')
    if re.search(r'kidney|renal', t) and re.search(r'distribut|accumul|organ|biodistribut', t): organs.append('kidney')
    if re.search(r'lung|pulmonary', t) and re.search(r'distribut|accumul|organ|biodistribut', t): organs.append('lung')
    if re.search(r'brain', t) and re.search(r'distribut|accumul|organ|biodistribut|cross.*barrier', t): organs.append('brain')
    if re.search(r'heart|cardiac', t) and re.search(r'distribut|accumul|organ|biodistribut', t): organs.append('heart')
    return '_'.join(organs) if organs else NM

# ── Generate notes ──────────────────────────────────────────────────
def generate_notes(paper, text):
    """Generate brief descriptive notes."""
    parts = []
    title = paper.get('title', '')
    # Truncate title for notes
    if len(title) > 100:
        title = title[:97] + '...'
    parts.append(title)
    return '. '.join(parts)

# ══════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def extract_row(paper):
    """Extract one DB row from a paper dict with abstract."""
    text = (paper.get('title', '') or '') + ' ' + (paper.get('abstract', '') or '')

    # Metadata
    surname = extract_surname(paper.get('authors', ''))
    year = paper.get('year', '')
    study_id = f"{surname}_{year}" if surname != "Unknown" else f"PMID{paper.get('pmid','')}_{year}"

    # Material
    core_mat, mat_class = detect_material(text)

    # Physicochemical
    size = extract_size(text)
    size_method = extract_size_method(text)
    shape = extract_shape(text)
    coating = extract_coating(text)
    targeting = extract_targeting(text)
    payload = extract_payload(text)
    charge_cat, zeta = extract_charge(text)

    # Model
    model = extract_model(text)
    cell_mat, cell_fet = extract_cell_lines(text)
    species = extract_species(text)
    gest_stage = extract_gestational_stage(text)
    route = extract_route(text)
    dose_val, dose_unit = extract_dose(text)
    duration = extract_duration(text)

    # Outcomes
    transloc_det, transloc_pct = extract_translocation(text)
    accum = extract_accumulation(text)
    cyto_obs, cyto_type = extract_cytotoxicity(text)
    detection = extract_detection(text)
    indirect = extract_indirect_effects(text)
    mat_organs = extract_maternal_organs(text)

    # Confidence: abstract_only = low
    has_pmc = bool(paper.get('pmcid', ''))
    confidence = 'medium' if has_pmc else 'low'

    notes = generate_notes(paper, text)

    row = {
        'study_id': study_id,
        'doi': paper.get('doi', '') or NM,
        'pmid': paper.get('pmid', '') or NM,
        'pmcid': paper.get('pmcid', '') or NM,
        'year': year or NM,
        'source': 'pubmed_enrichment',
        'confidence': confidence,
        'notes': notes,
        'core_material': core_mat,
        'material_class': mat_class,
        'size_nm': size,
        'size_method': size_method,
        'size_in_medium_nm': NM,  # rarely in abstracts
        'shape': shape,
        'surface_coating': coating,
        'targeting_ligand': targeting,
        'therapeutic_payload': payload,
        'surface_charge_cat': charge_cat,
        'zeta_potential_mV': zeta,
        'model_type': model,
        'cell_line_maternal': cell_mat,
        'cell_line_fetal': cell_fet,
        'species': species,
        'gestational_stage': gest_stage,
        'exposure_route': route,
        'dose_value': dose_val,
        'dose_unit': dose_unit,
        'exposure_duration_h': duration,
        'translocation_detected': transloc_det,
        'translocation_pct': transloc_pct,
        'placental_accumulation': accum,
        'placental_uptake_value': NM,  # specific values rarely in abstracts
        'placental_uptake_unit': NM,
        'placenta_fetus_ratio': NM,  # rarely in abstracts
        'extraembryonic_uptake': NM,
        'fetal_distribution': NM,
        'maternal_organ_distribution': mat_organs,
        'cytotoxicity_observed': cyto_obs,
        'cytotoxicity_type': cyto_type,
        'indirect_effects': indirect,
        'detection_method': detection
    }
    return row

# ── Load data ───────────────────────────────────────────────────────
print("Loading paper data...")

# Read existing PMIDs
existing_pmids = set()
with open('DB.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('pmid'):
            existing_pmids.add(row['pmid'].strip())
print(f"  Existing PMIDs in DB: {len(existing_pmids)}")

# Merge all paper metadata
with open('new_papers_enriched.json', 'r', encoding='utf-8') as f:
    enriched = json.load(f)
try:
    with open('new_papers_found.json', 'r', encoding='utf-8') as f:
        found = json.load(f)
except:
    found = []

all_papers = {}
for p in found:
    pmid = p.get('pmid', '')
    if pmid and pmid not in existing_pmids:
        all_papers[pmid] = p
for p in enriched:
    pmid = p.get('pmid', '')
    if pmid and pmid not in existing_pmids:
        all_papers[pmid] = p

print(f"  Total new papers with metadata: {len(all_papers)}")

# Filter for ACTUAL placenta/pregnancy + NP papers
relevant = {}
for pmid, p in all_papers.items():
    text = (p.get('title', '') or '') + ' ' + (p.get('abstract', '') or '')
    has_placenta = bool(PLACENTA_RE.search(text))
    has_np = bool(NP_RE.search(text))
    if has_placenta and has_np:
        relevant[pmid] = p

print(f"  Papers about BOTH placenta AND nanoparticles: {len(relevant)}")

# ── Extract all rows ────────────────────────────────────────────────
print("\nExtracting fields from abstracts...")
rows = []
study_id_counts = {}

for pmid, paper in sorted(relevant.items(), key=lambda x: x[1].get('year', ''), reverse=True):
    row = extract_row(paper)

    # Handle duplicate study_ids (same author, same year)
    sid = row['study_id']
    if sid in study_id_counts:
        study_id_counts[sid] += 1
        # Distinguish by adding letter
        row['study_id'] = f"{sid}{chr(96 + study_id_counts[sid])}"
    else:
        study_id_counts[sid] = 1

    rows.append(row)

print(f"  Extracted {len(rows)} rows")

# ── Write output ────────────────────────────────────────────────────
output_file = 'new_rows.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} rows to {output_file}")

# ── Statistics ──────────────────────────────────────────────────────
print("\n=== Extraction Statistics ===")

# Material distribution
from collections import Counter
materials = Counter(r['core_material'] for r in rows)
print(f"\nMaterial distribution:")
for mat, count in materials.most_common(15):
    print(f"  {mat}: {count}")

# Fields with data (not NM)
print(f"\nField coverage (% of rows with data):")
for col in COLUMNS:
    filled = sum(1 for r in rows if r[col] != NM and r[col] != '' and r[col] != 'none')
    pct = filled / len(rows) * 100 if rows else 0
    if pct > 0:
        print(f"  {col}: {filled}/{len(rows)} ({pct:.1f}%)")

# Year distribution
years = Counter(r['year'] for r in rows if r['year'] != NM)
print(f"\nYear distribution:")
for yr in sorted(years.keys()):
    print(f"  {yr}: {years[yr]}")
