#!/usr/bin/env python3
"""
Build NanoPlacentaDB HTML from DB_enriched.csv using TEMPLATE.html style.
Only DATA tab (Database Explorer) - no dashboard.
"""
import csv, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Read DB
rows = []
with open('DB_enriched.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for row in reader:
        rec = {}
        for c in cols:
            val = row.get(c, '')
            # Convert types
            if val == '':
                rec[c] = ''
            elif val == 'not_mentioned':
                rec[c] = ''
            elif val in ('TRUE', 'True', 'true'):
                rec[c] = True
            elif val in ('FALSE', 'False', 'false'):
                rec[c] = False
            elif c in ('year', 'pmid') and val.isdigit():
                rec[c] = int(val)
            elif c in ('size_nm', 'size_in_medium_nm', 'zeta_potential_mV', 'dose_value',
                        'placental_uptake_value', 'placenta_fetus_ratio'):
                try:
                    rec[c] = float(val)
                except:
                    rec[c] = val
            else:
                rec[c] = val
        rows.append(rec)

print(f"Loaded {len(rows)} rows")

# Stats
unique_studies = len(set(r['study_id'] for r in rows))
unique_pmids = len(set(str(r['pmid']) for r in rows if r.get('pmid')))
materials = len(set(r['core_material'] for r in rows if r.get('core_material')))
au_count = sum(1 for r in rows if r.get('core_material') == 'Au')
lipo_count = sum(1 for r in rows if r.get('core_material') in ('liposome', 'LNP', 'lipid_NP'))
years = [r['year'] for r in rows if isinstance(r.get('year'), int)]
year_range = f"{min(years)}-{max(years)}" if years else ""

# Serialize data
db_json = json.dumps(rows, ensure_ascii=False)

html = f'''<!DOCTYPE html>
<html lang="he" dir="ltr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NanoPlacentaDB - Data Explorer</title>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --card-hover: #222636;
    --border: #2a2e3d;
    --text: #e4e4e7;
    --text-dim: #9ca3af;
    --accent: #8b5cf6;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --accent4: #ef4444;
    --accent5: #10b981;
    --gold: #fbbf24;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%); padding: 2rem 2rem 1.5rem; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #c4b5fd, #67e8f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.25rem; }}
  .header p {{ color: var(--text-dim); font-size: 0.9rem; }}
  .header .meta {{ display: flex; gap: 2rem; margin-top: 0.75rem; flex-wrap: wrap; }}
  .header .meta span {{ font-size: 0.8rem; color: var(--text-dim); }}
  .header .meta span strong {{ color: var(--accent2); }}

  /* Summary Cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; transition: transform 0.2s, border-color 0.2s; }}
  .card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
  .card .number {{ font-size: 1.8rem; font-weight: 800; line-height: 1; }}
  .card .label {{ font-size: 0.72rem; color: var(--text-dim); margin-top: 0.2rem; }}
  .card.purple .number {{ color: var(--accent); }}
  .card.cyan .number {{ color: var(--accent2); }}
  .card.amber .number {{ color: var(--accent3); }}
  .card.red .number {{ color: var(--accent4); }}
  .card.green .number {{ color: var(--accent5); }}
  .card.gold .number {{ color: var(--gold); }}

  /* ====== DB EXPLORER ====== */
  .container {{ max-width: 1800px; margin: 0 auto; padding: 1.5rem; }}

  .db-toolbar {{
    display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;
    padding: 1rem 1.25rem; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px 12px 0 0; border-bottom: none;
  }}
  .db-search {{
    flex: 1; min-width: 200px; padding: 0.55rem 1rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: 8px; color: var(--text);
    font-size: 0.85rem; outline: none; transition: border-color 0.2s;
  }}
  .db-search:focus {{ border-color: var(--accent); }}
  .db-search::placeholder {{ color: #555; }}

  .db-select {{
    padding: 0.55rem 0.75rem; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); font-size: 0.8rem; cursor: pointer; outline: none;
  }}
  .db-select:focus {{ border-color: var(--accent); }}

  .db-btn {{
    padding: 0.55rem 1rem; background: var(--accent); border: none;
    border-radius: 8px; color: white; font-size: 0.8rem; font-weight: 600;
    cursor: pointer; transition: opacity 0.2s; white-space: nowrap;
  }}
  .db-btn:hover {{ opacity: 0.85; }}
  .db-btn.secondary {{ background: var(--bg); border: 1px solid var(--border); color: var(--text-dim); }}
  .db-btn.secondary:hover {{ border-color: var(--accent); color: var(--text); }}

  .db-count {{ font-size: 0.8rem; color: var(--text-dim); white-space: nowrap; }}

  .db-table-wrap {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 0 0 12px 12px; overflow: hidden;
  }}
  .db-scroll {{ overflow: auto; max-height: 75vh; }}

  .db-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  .db-table th {{
    background: #14161e; color: var(--text-dim); font-weight: 600;
    text-align: left; padding: 0.55rem 0.65rem; white-space: nowrap;
    position: sticky; top: 0; z-index: 10; cursor: pointer;
    border-bottom: 2px solid var(--border); user-select: none;
  }}
  .db-table th:hover {{ color: var(--accent2); }}
  .db-table th.sorted-asc::after {{ content: ' \\2191'; color: var(--accent); }}
  .db-table th.sorted-desc::after {{ content: ' \\2193'; color: var(--accent); }}
  .db-table td {{
    padding: 0.45rem 0.65rem; border-bottom: 1px solid #1f2233;
    max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .db-table tr:hover td {{ background: #1e2235; }}

  .badge {{ display: inline-block; padding: 0.12rem 0.45rem; border-radius: 10px; font-size: 0.68rem; font-weight: 600; }}
  .badge-yes {{ background: #052e16; color: #4ade80; }}
  .badge-no {{ background: #450a0a; color: #fca5a5; }}
  .badge-na {{ background: #1c1917; color: #78716c; }}
  .badge-material {{ background: #1e1b4b; color: #c4b5fd; }}
  .badge-model {{ background: #042f2e; color: #5eead4; }}
  .badge-source {{ background: #172554; color: #93c5fd; }}
  .badge-au {{ background: #422006; color: #fbbf24; }}
  .badge-lipo {{ background: #052e16; color: #86efac; }}

  .doi-link {{ color: var(--accent2); text-decoration: none; font-size: 0.75rem; }}
  .doi-link:hover {{ text-decoration: underline; }}

  /* Row detail panel */
  .detail-overlay {{
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center;
  }}
  .detail-overlay.show {{ display: flex; }}
  .detail-panel {{
    background: var(--card); border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem; max-width: 700px; width: 90%; max-height: 85vh; overflow-y: auto;
  }}
  .detail-panel h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: var(--accent2); }}
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }}
  .detail-field {{ padding: 0.4rem 0; }}
  .detail-field .dlabel {{ font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }}
  .detail-field .dvalue {{ font-size: 0.9rem; color: var(--text); word-break: break-word; }}
  .detail-field.full {{ grid-column: 1 / -1; }}
  .detail-close {{ float: right; background: none; border: none; color: var(--text-dim); font-size: 1.5rem; cursor: pointer; }}
  .detail-close:hover {{ color: var(--accent4); }}

  .footer {{ text-align: center; padding: 2rem; color: var(--text-dim); font-size: 0.75rem; border-top: 1px solid var(--border); }}

  @media (max-width: 768px) {{
    .cards {{ grid-template-columns: repeat(2, 1fr); }}
    .detail-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>NanoPlacentaDB</h1>
  <p>Structured Database of Nanoparticle Placental Transfer Studies</p>
  <div class="meta">
    <span>Researcher: <strong>Shiri Katzir</strong>, Bar-Ilan University</span>
    <span>Supervisors: <strong>Prof. R. Popovtzer</strong> (BINA) + <strong>Dr. A. Tzur</strong> (Sheba)</span>
    <span>Updated: <strong>2026-03-10</strong></span>
  </div>
</div>

<div class="container">

  <!-- Summary Cards -->
  <div class="cards">
    <div class="card purple"><div class="number">{len(rows)}</div><div class="label">Total Records</div></div>
    <div class="card cyan"><div class="number">{unique_studies}</div><div class="label">Unique Studies</div></div>
    <div class="card amber"><div class="number">{materials}</div><div class="label">Material Types</div></div>
    <div class="card gold"><div class="number">{au_count}</div><div class="label">Gold NP Records</div></div>
    <div class="card green"><div class="number">{lipo_count}</div><div class="label">Liposome/LNP Records</div></div>
    <div class="card red"><div class="number">{year_range}</div><div class="label">Year Range</div></div>
  </div>

  <!-- Toolbar -->
  <div class="db-toolbar">
    <input type="text" class="db-search" id="dbSearch" placeholder="Search by study, material, notes, coating..." oninput="applyFilters()">
    <select class="db-select" id="filterMaterial" onchange="applyFilters()">
      <option value="">All Materials</option>
    </select>
    <select class="db-select" id="filterModel" onchange="applyFilters()">
      <option value="">All Models</option>
    </select>
    <select class="db-select" id="filterSpecies" onchange="applyFilters()">
      <option value="">All Species</option>
    </select>
    <select class="db-select" id="filterTrans" onchange="applyFilters()">
      <option value="">Translocation: All</option>
      <option value="true">Translocation: YES</option>
      <option value="false">Translocation: NO</option>
    </select>
    <select class="db-select" id="filterTox" onchange="applyFilters()">
      <option value="">Cytotoxicity: All</option>
      <option value="true">Cytotoxic: YES</option>
      <option value="false">Cytotoxic: NO</option>
    </select>
    <select class="db-select" id="filterSource" onchange="applyFilters()">
      <option value="">All Sources</option>
    </select>
    <select class="db-select" id="filterConf" onchange="applyFilters()">
      <option value="">All Confidence</option>
      <option value="high">High</option>
      <option value="medium">Medium</option>
      <option value="low">Low</option>
    </select>
    <button class="db-btn secondary" onclick="resetFilters()">Reset</button>
    <button class="db-btn" onclick="exportCSV()">Export CSV</button>
    <span class="db-count"><span id="dbCount">{len(rows)}</span> / {len(rows)} records</span>
  </div>

  <!-- Table -->
  <div class="db-table-wrap">
    <div class="db-scroll">
      <table class="db-table" id="dbTable">
        <thead>
          <tr>
            <th data-col="study_id" onclick="sortDB('study_id')">Study</th>
            <th data-col="year" onclick="sortDB('year')">Year</th>
            <th data-col="core_material" onclick="sortDB('core_material')">Material</th>
            <th data-col="material_class" onclick="sortDB('material_class')">Class</th>
            <th data-col="size_nm" onclick="sortDB('size_nm')">Size (nm)</th>
            <th data-col="size_method" onclick="sortDB('size_method')">Size Method</th>
            <th data-col="shape" onclick="sortDB('shape')">Shape</th>
            <th data-col="surface_coating" onclick="sortDB('surface_coating')">Coating</th>
            <th data-col="targeting_ligand" onclick="sortDB('targeting_ligand')">Ligand</th>
            <th data-col="therapeutic_payload" onclick="sortDB('therapeutic_payload')">Payload</th>
            <th data-col="surface_charge_cat" onclick="sortDB('surface_charge_cat')">Charge</th>
            <th data-col="zeta_potential_mV" onclick="sortDB('zeta_potential_mV')">Zeta (mV)</th>
            <th data-col="model_type" onclick="sortDB('model_type')">Model</th>
            <th data-col="species" onclick="sortDB('species')">Species</th>
            <th data-col="gestational_stage" onclick="sortDB('gestational_stage')">Gest. Stage</th>
            <th data-col="exposure_route" onclick="sortDB('exposure_route')">Route</th>
            <th data-col="dose_value" onclick="sortDB('dose_value')">Dose</th>
            <th data-col="dose_unit" onclick="sortDB('dose_unit')">Unit</th>
            <th data-col="exposure_duration_h" onclick="sortDB('exposure_duration_h')">Duration (h)</th>
            <th data-col="translocation_detected" onclick="sortDB('translocation_detected')">Transloc.</th>
            <th data-col="translocation_pct" onclick="sortDB('translocation_pct')">%</th>
            <th data-col="placental_accumulation" onclick="sortDB('placental_accumulation')">Accum.</th>
            <th data-col="placenta_fetus_ratio" onclick="sortDB('placenta_fetus_ratio')">P:F Ratio</th>
            <th data-col="cytotoxicity_observed" onclick="sortDB('cytotoxicity_observed')">Cytotox.</th>
            <th data-col="cytotoxicity_type" onclick="sortDB('cytotoxicity_type')">Tox. Type</th>
            <th data-col="indirect_effects" onclick="sortDB('indirect_effects')">Indirect Effects</th>
            <th data-col="detection_method" onclick="sortDB('detection_method')">Detection</th>
            <th data-col="confidence" onclick="sortDB('confidence')">Conf.</th>
            <th data-col="source" onclick="sortDB('source')">Source</th>
            <th>DOI</th>
          </tr>
        </thead>
        <tbody id="dbBody"></tbody>
      </table>
    </div>
  </div>

</div>

<!-- Detail Overlay -->
<div class="detail-overlay" id="detailOverlay" onclick="if(event.target===this)closeDetail()">
  <div class="detail-panel" id="detailPanel"></div>
</div>

<div class="footer">
  NanoPlacentaDB &copy; 2025-2026 &bull; Shiri Katzir, Bar-Ilan University &bull; {len(rows)} records from {unique_studies} studies &bull; Generated by Claude Code
</div>

<script>
const DB = {db_json};

// ====== DB EXPLORER ======
let currentSort = {{ col: null, dir: 1 }};
let filteredDB = [...DB];

function populateDropdowns() {{
  const add = (id, field) => {{
    const sel = document.getElementById(id);
    const vals = [...new Set(DB.map(r => r[field]).filter(v => v && v !== ''))].sort();
    vals.forEach(v => {{ const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }});
  }};
  add('filterMaterial', 'core_material');
  add('filterModel', 'model_type');
  add('filterSpecies', 'species');
  add('filterSource', 'source');
}}

function applyFilters() {{
  const q = document.getElementById('dbSearch').value.toLowerCase();
  const mat = document.getElementById('filterMaterial').value;
  const mod = document.getElementById('filterModel').value;
  const sp = document.getElementById('filterSpecies').value;
  const tr = document.getElementById('filterTrans').value;
  const tx = document.getElementById('filterTox').value;
  const src = document.getElementById('filterSource').value;
  const conf = document.getElementById('filterConf').value;

  filteredDB = DB.filter(r => {{
    if (mat && r.core_material !== mat) return false;
    if (mod && r.model_type !== mod) return false;
    if (sp && r.species !== sp) return false;
    if (src && r.source !== src) return false;
    if (conf && r.confidence !== conf) return false;
    if (tr === 'true' && r.translocation_detected !== true) return false;
    if (tr === 'false' && r.translocation_detected !== false) return false;
    if (tx === 'true' && r.cytotoxicity_observed !== true) return false;
    if (tx === 'false' && r.cytotoxicity_observed !== false) return false;
    if (q) {{
      const text = [r.study_id, r.core_material, r.surface_coating, r.targeting_ligand,
        r.model_type, r.species, r.notes, r.detection_method, r.doi, r.cytotoxicity_type,
        r.gestational_stage, r.exposure_route, r.material_class, r.therapeutic_payload,
        r.indirect_effects, r.source, r.confidence].join(' ').toLowerCase();
      if (!text.includes(q)) return false;
    }}
    return true;
  }});

  if (currentSort.col) sortFilteredDB();
  renderDB();
}}

function resetFilters() {{
  document.getElementById('dbSearch').value = '';
  document.getElementById('filterMaterial').value = '';
  document.getElementById('filterModel').value = '';
  document.getElementById('filterSpecies').value = '';
  document.getElementById('filterTrans').value = '';
  document.getElementById('filterTox').value = '';
  document.getElementById('filterSource').value = '';
  document.getElementById('filterConf').value = '';
  currentSort = {{ col: null, dir: 1 }};
  document.querySelectorAll('.db-table th').forEach(th => th.className = '');
  filteredDB = [...DB];
  renderDB();
}}

function sortDB(col) {{
  if (currentSort.col === col) currentSort.dir *= -1;
  else {{ currentSort.col = col; currentSort.dir = 1; }}
  document.querySelectorAll('.db-table th').forEach(th => th.className = '');
  const th = document.querySelector(`.db-table th[data-col="${{col}}"]`);
  if (th) th.className = currentSort.dir === 1 ? 'sorted-asc' : 'sorted-desc';
  sortFilteredDB();
  renderDB();
}}

function sortFilteredDB() {{
  const col = currentSort.col;
  const dir = currentSort.dir;
  filteredDB.sort((a, b) => {{
    let va = a[col], vb = b[col];
    if (va === '' || va === null || va === undefined) va = null;
    if (vb === '' || vb === null || vb === undefined) vb = null;
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
    if (typeof va === 'boolean' && typeof vb === 'boolean') return ((va ? 1 : 0) - (vb ? 1 : 0)) * dir;
    return String(va).localeCompare(String(vb)) * dir;
  }});
}}

function boolBadge(val) {{
  if (val === true) return '<span class="badge badge-yes">YES</span>';
  if (val === false) return '<span class="badge badge-no">NO</span>';
  return '<span class="badge badge-na">-</span>';
}}

function matBadge(val) {{
  if (!val) return '';
  if (val === 'Au') return '<span class="badge badge-au">Au</span>';
  if (val === 'liposome' || val === 'LNP' || val === 'lipid_NP') return '<span class="badge badge-lipo">' + val + '</span>';
  return '<span class="badge badge-material">' + val + '</span>';
}}

function renderDB() {{
  const tbody = document.getElementById('dbBody');
  tbody.innerHTML = '';
  filteredDB.forEach((r, i) => {{
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.onclick = () => showDetail(r);

    const ligand = r.targeting_ligand && r.targeting_ligand !== 'none' ? r.targeting_ligand : '';
    const payload = r.therapeutic_payload && r.therapeutic_payload !== 'none' ? r.therapeutic_payload : '';
    const doiShort = r.doi ? `<a class="doi-link" href="https://doi.org/${{r.doi}}" target="_blank" onclick="event.stopPropagation()">${{r.doi.length > 20 ? r.doi.substring(0,20)+'...' : r.doi}}</a>` : '';
    const doseStr = r.dose_value ? String(r.dose_value) : '';

    tr.innerHTML = `
      <td><strong>${{r.study_id}}</strong></td>
      <td>${{r.year||''}}</td>
      <td>${{matBadge(r.core_material)}}</td>
      <td>${{r.material_class||''}}</td>
      <td>${{r.size_nm||''}}</td>
      <td>${{r.size_method||''}}</td>
      <td>${{r.shape||''}}</td>
      <td>${{r.surface_coating||''}}</td>
      <td>${{ligand}}</td>
      <td>${{payload}}</td>
      <td>${{r.surface_charge_cat||''}}</td>
      <td>${{r.zeta_potential_mV||''}}</td>
      <td><span class="badge badge-model">${{(r.model_type||'').replace('in_vitro_','').replace('ex_vivo_','').replace('in_vivo_','')}}</span></td>
      <td>${{r.species||''}}</td>
      <td>${{r.gestational_stage||''}}</td>
      <td>${{r.exposure_route||''}}</td>
      <td>${{doseStr}}</td>
      <td>${{r.dose_unit||''}}</td>
      <td>${{r.exposure_duration_h||''}}</td>
      <td>${{boolBadge(r.translocation_detected)}}</td>
      <td>${{r.translocation_pct||''}}</td>
      <td>${{boolBadge(r.placental_accumulation)}}</td>
      <td>${{r.placenta_fetus_ratio||''}}</td>
      <td>${{boolBadge(r.cytotoxicity_observed)}}</td>
      <td>${{r.cytotoxicity_type||''}}</td>
      <td>${{r.indirect_effects||''}}</td>
      <td>${{r.detection_method||''}}</td>
      <td>${{r.confidence||''}}</td>
      <td><span class="badge badge-source">${{r.source||''}}</span></td>
      <td>${{doiShort}}</td>
    `;
    tbody.appendChild(tr);
  }});
  document.getElementById('dbCount').textContent = filteredDB.length;
}}

function showDetail(r) {{
  const ALL_FIELDS = [
    ['study_id','Study ID'], ['doi','DOI'], ['pmid','PMID'], ['pmcid','PMCID'],
    ['year','Year'], ['source','Source'], ['confidence','Confidence'],
    ['core_material','Core Material'], ['material_class','Material Class'],
    ['size_nm','Size (nm)'], ['size_method','Size Method'], ['size_in_medium_nm','Size in Medium (nm)'],
    ['shape','Shape'], ['surface_coating','Surface Coating'], ['targeting_ligand','Targeting Ligand'],
    ['therapeutic_payload','Therapeutic Payload'], ['surface_charge_cat','Charge Category'],
    ['zeta_potential_mV','Zeta Potential (mV)'], ['model_type','Model Type'],
    ['cell_line_maternal','Maternal Cell Line'], ['cell_line_fetal','Fetal Cell Line'],
    ['species','Species'], ['gestational_stage','Gestational Stage'], ['exposure_route','Exposure Route'],
    ['dose_value','Dose Value'], ['dose_unit','Dose Unit'], ['exposure_duration_h','Duration (h)'],
    ['translocation_detected','Translocation Detected'], ['translocation_pct','Translocation %'],
    ['placental_accumulation','Placental Accumulation'], ['placental_uptake_value','Placental Uptake Value'],
    ['placental_uptake_unit','Placental Uptake Unit'], ['placenta_fetus_ratio','P:F Ratio'],
    ['extraembryonic_uptake','Extraembryonic Uptake'], ['fetal_distribution','Fetal Distribution'],
    ['maternal_organ_distribution','Maternal Organ Dist.'], ['cytotoxicity_observed','Cytotoxicity'],
    ['cytotoxicity_type','Cytotoxicity Type'], ['indirect_effects','Indirect Effects'],
    ['detection_method','Detection Method']
  ];

  let html = `<button class="detail-close" onclick="closeDetail()">&times;</button>`;
  html += `<h2>${{r.study_id}} &mdash; ${{r.core_material}} ${{r.surface_coating||''}}</h2>`;

  if (r.notes) {{
    html += `<div style="background:var(--bg);padding:0.75rem 1rem;border-radius:8px;margin-bottom:1rem;font-size:0.85rem;color:var(--text-dim);line-height:1.5">${{r.notes}}</div>`;
  }}

  html += '<div class="detail-grid">';
  ALL_FIELDS.forEach(([key, label]) => {{
    let val = r[key];
    if (val === true) val = '<span class="badge badge-yes">YES</span>';
    else if (val === false) val = '<span class="badge badge-no">NO</span>';
    else if (val === null || val === undefined || val === '') val = '<span style="color:#555">-</span>';
    else if (key === 'doi' && val) val = `<a class="doi-link" href="https://doi.org/${{val}}" target="_blank">${{val}}</a>`;
    else if (key === 'pmid' && val) val = `<a class="doi-link" href="https://pubmed.ncbi.nlm.nih.gov/${{val}}" target="_blank">${{val}}</a>`;
    else if (key === 'pmcid' && val) val = `<a class="doi-link" href="https://pmc.ncbi.nlm.nih.gov/articles/${{val}}" target="_blank">${{val}}</a>`;

    const isLong = key === 'notes' || key === 'fetal_distribution' || key === 'maternal_organ_distribution' || key === 'indirect_effects';
    html += `<div class="detail-field ${{isLong?'full':''}}""><div class="dlabel">${{label}}</div><div class="dvalue">${{val}}</div></div>`;
  }});
  html += '</div>';

  document.getElementById('detailPanel').innerHTML = html;
  document.getElementById('detailOverlay').classList.add('show');
}}

function closeDetail() {{
  document.getElementById('detailOverlay').classList.remove('show');
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDetail(); }});

function exportCSV() {{
  const headers = ['study_id','doi','pmid','pmcid','year','source','confidence','notes','core_material','material_class','size_nm','size_method','size_in_medium_nm','shape','surface_coating','targeting_ligand','therapeutic_payload','surface_charge_cat','zeta_potential_mV','model_type','cell_line_maternal','cell_line_fetal','species','gestational_stage','exposure_route','dose_value','dose_unit','exposure_duration_h','translocation_detected','translocation_pct','placental_accumulation','placental_uptake_value','placental_uptake_unit','placenta_fetus_ratio','extraembryonic_uptake','fetal_distribution','maternal_organ_distribution','cytotoxicity_observed','cytotoxicity_type','indirect_effects','detection_method'];

  let csv = '\\uFEFF' + headers.join(',') + '\\n';
  filteredDB.forEach(r => {{
    csv += headers.map(h => {{
      let v = r[h];
      if (v === null || v === undefined) v = '';
      v = String(v);
      if (v.includes(',') || v.includes('"') || v.includes('\\n')) v = '"' + v.replace(/"/g, '""') + '"';
      return v;
    }}).join(',') + '\\n';
  }});

  const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `nanoplacentadb_export_${{filteredDB.length}}rows.csv`;
  a.click();
}}

// ====== INIT ======
populateDropdowns();
renderDB();
</script>
</body>
</html>'''

with open('NanoPlacentaDB_Data.html', 'w', encoding='utf-8') as f:
    f.write(html)

import os
size_kb = os.path.getsize('NanoPlacentaDB_Data.html') / 1024
print(f"Wrote NanoPlacentaDB_Data.html ({size_kb:.0f} KB)")
print(f"  {len(rows)} records, {unique_studies} studies")
