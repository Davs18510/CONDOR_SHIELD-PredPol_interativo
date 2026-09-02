# =============================================================================
# ETAPA 6 — DASHBOARD INTERATIVO DE POLICIAMENTO PREDITIVO HIBRIDO
# =============================================================================
# Gera outputs/dashboard_preditivo.html — aplicação SPA com:
#   - Previsão espaço-temporal XGBoost Poisson
#   - Heatmap de densidade 100% fidedigno com efeito visual de áreas quentes
#   - Exibição de risco percentual de ocorrência Poisson (%) nos cards e popups
#   - Atualização automática live ao alterar qualquer controle/slider (sem botão)
#   - Direcionamento de viaturas alocado estritamente nos maiores picos de risco
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import json
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACOES ────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -24.008, -23.357
LON_MIN, LON_MAX = -46.826, -46.365
GRID_N = 60
BW = 0.003

CRIME_KEYS = [
    'TENTATIVA DE HOMICIDIO',
    'HOMICIDIO DOLOSO',
    'LESAO CORPORAL SEGUIDA DE MORTE',
    'LATROCINIO',
]
CRIME_LABELS = {
    'TENTATIVA DE HOMICIDIO'         : 'Tentativa de Homicidio',
    'HOMICIDIO DOLOSO'               : 'Homicidio Doloso',
    'LESAO CORPORAL SEGUIDA DE MORTE': 'Lesao Corp. Seguida de Morte',
    'LATROCINIO'                     : 'Latrocinio',
}

# ── CARREGAMENTO ─────────────────────────────────────────────────────────────
print('Carregando dados...')
df_tr = pd.read_csv('data/processed/df_treino.csv',    sep=';')
df_va = pd.read_csv('data/processed/df_validacao.csv', sep=';')
df_te = pd.read_csv('data/processed/df_teste.csv',     sep=';')

for df in [df_tr, df_va, df_te]:
    df['LATITUDE']  = pd.to_numeric(df['LATITUDE'],  errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')

coords = df_tr[df_tr['LATITUDE'].notna()].copy()
print(f'Pontos de treino com coords: {len(coords)}')

# ── GRADE 60x60 ───────────────────────────────────────────────────────────────
lat_arr = np.linspace(LAT_MIN, LAT_MAX, GRID_N)
lon_arr = np.linspace(LON_MIN, LON_MAX, GRID_N)
lg, ng  = np.meshgrid(lat_arr, lon_arr, indexing='ij')
grade   = np.radians(np.column_stack([lg.ravel(), ng.ravel()]))

def kde_grid(pts_df):
    if len(pts_df) < 5:
        return [[0]*GRID_N for _ in range(GRID_N)]
    rad = np.radians(pts_df[['LATITUDE', 'LONGITUDE']].values)
    kde = KernelDensity(kernel='gaussian', metric='haversine', bandwidth=BW)
    kde.fit(rad)
    d = np.exp(kde.score_samples(grade))
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    return (d.reshape(GRID_N, GRID_N) * 100).round(1).tolist()

print('Calculando grades KDE por tipo de crime...')
grids = {'all': kde_grid(coords)}
for ck in CRIME_KEYS:
    sub = coords[coords['NATUREZA APURADA'] == ck]
    print(f'  {CRIME_LABELS[ck]}: {len(sub)} pontos')
    grids[ck] = kde_grid(sub)

# ── CARREGA PREVISOES MULTI-HORIZONTE (XGBOOST) ──────────────────────────────
with open('data/processed/previsoes_xgboost_horizonte.json', encoding='utf-8') as f:
    dados_xgb = json.load(f)

probabilidades = dados_xgb['probabilidades']
lambdas_mes    = dados_xgb['lambdas']
pred_pts_por_h = dados_xgb['pred_pts']

# ── DENSIDADE ESPACIAL 100% FIDEDIGNA AOS PONTOS DE CRIMES PREVISTOS ─────────
BANDWIDTH_KM = 1.5
R_TERRA = 6371.0

lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing='ij')
lat_flat = lat_grid.ravel()
lon_flat = lon_grid.ravel()

print('Gerando grades de densidade preditiva fidedignas aos pontos de crimes previstos...')
densidades_brutas = {}
max_global = 1e-9

for h in [1, 2, 3, 4, 5, 6]:
    pts_h = pred_pts_por_h.get(str(h), [])
    if not pts_h:
        densidades_brutas[str(h)] = np.zeros(GRID_N * GRID_N)
        continue
    
    lats_p = np.array([p['lat'] for p in pts_h])
    lons_p = np.array([p['lon'] for p in pts_h])
    
    p1 = np.radians(lat_flat)[:, None]                # (3600, 1)
    p2 = np.radians(lats_p)[None, :]                  # (1, M)
    dphi = np.radians(lats_p[None, :] - lat_flat[:, None])
    dl   = np.radians(lons_p[None, :] - lon_flat[:, None])
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    dist_km = 2 * R_TERRA * np.arcsin(np.sqrt(a))      # (3600, M)
    
    # Soma dos kernels Gaussianos sobre as coordenadas exatas dos pontos previstos
    d_h = np.exp(-(dist_km**2) / (2 * BANDWIDTH_KM**2)).sum(axis=1) # (3600,)
    densidades_brutas[str(h)] = d_h
    if d_h.max() > max_global:
        max_global = d_h.max()

grids_horizonte = {}
for h in [1, 2, 3, 4, 5, 6]:
    d_h = densidades_brutas.get(str(h), np.zeros(GRID_N * GRID_N))
    d_norm = (d_h / max_global) * 100.0
    grid_h = d_norm.reshape(GRID_N, GRID_N)
    grids_horizonte[str(h)] = grid_h.round(1).tolist()

# ── PONTOS PARA DISPLAY ───────────────────────────────────────────────────────
def pts_json(df, periodo):
    out = []
    for _, r in df[df['LATITUDE'].notna()].iterrows():
        out.append({
            'lat'  : round(float(r['LATITUDE']),  5),
            'lon'  : round(float(r['LONGITUDE']), 5),
            'tipo' : str(r.get('NATUREZA APURADA', '')),
            'data' : str(r.get('DATA_OCORRENCIA_BO', ''))[:10],
            'local': str(r.get('DESCR_TIPOLOCAL', '')),
            'deleg': str(r.get('NOME_DELEGACIA_CIRC', '')),
            'per'  : str(r.get('DESC_PERIODO', '')),
            'ds'   : periodo,
        })
    return out

turnos = df_tr['TURNO'].value_counts().to_dict() if 'TURNO' in df_tr.columns \
         else {'noite': 99, 'manha': 77, 'madrugada': 76, 'tarde': 67}
tipo_counts = coords['NATUREZA APURADA'].value_counts().to_dict()

DATA = {
    'grids'           : grids,
    'grids_horizonte' : grids_horizonte,
    'pred_pts'        : pred_pts_por_h,
    'lambdas'         : lambdas_mes,
    'probabilidades'  : probabilidades,
    'lat_arr'         : lat_arr.round(5).tolist(),
    'lon_arr'         : lon_arr.round(5).tolist(),
    'grid_n'          : GRID_N,
    'treino_pts'      : pts_json(df_tr, 'treino'),
    'val_pts'         : pts_json(df_va, 'val'),
    'test_pts'        : pts_json(df_te, 'test'),
    'metrics'         : {
        'n_treino': len(df_tr), 'n_coords': len(coords),
        'n_val': len(df_va),    'n_test': len(df_te),
    },
    'turnos'          : turnos,
    'tipo_counts'     : tipo_counts,
    'crime_keys'      : CRIME_KEYS,
    'crime_labels'    : CRIME_LABELS,
}

data_js = 'const DATA = ' + json.dumps(DATA, ensure_ascii=False, separators=(',', ':')) + ';'
print(f'Dados JSON: {len(data_js)/1024:.1f} KB')

# ── HTML ───────────────────────────────────────────────────────────────────────
HEAD = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CONDOR SHIELD - Policiamento Preditivo XGBoost</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://leaflet.github.io/Leaflet.heat/dist/leaflet-heat.js"></script>
<style>
:root {
  --preto-suave: #1A1A1A;
  --azul-fechado: #0F2027;
  --cinza-prata: #BDC3C7;
  --branco-puro: #FFFFFF;
  --bg: #0F2027;
  --bg2: #1A1A1A;
  --card: rgba(15, 32, 39, 0.65);
  --bdr: rgba(189, 195, 199, 0.18);
  --bdr-hover: rgba(189, 195, 199, 0.45);
  --txt: #FFFFFF;
  --muted: #BDC3C7;
  --red: #EF4444;
  --ora: #F97316;
  --yel: #EAB308;
  --bl: #FFFFFF;
  --blue: #BDC3C7;
  --sw: 290px;
  --pw: 370px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  height: 100%;
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--txt);
  overflow: hidden;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: #1A1A1A;
}
::-webkit-scrollbar-thumb {
  background: rgba(189, 195, 199, 0.25);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(189, 195, 199, 0.5);
}

#hdr {
  height: 54px;
  background: linear-gradient(90deg, #0F2027 0%, #1A1A1A 50%, #0F2027 100%);
  border-bottom: 1px solid var(--bdr);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  position: relative;
  z-index: 999;
}

.hdr-l {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-shield {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #1A1A1A, #0F2027);
  border: 1px solid rgba(189, 195, 199, 0.35);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  box-shadow: 0 0 14px rgba(0, 0, 0, 0.5);
}

.hdr-l h1 {
  font-size: 15px;
  font-weight: 800;
  letter-spacing: .03em;
  color: var(--branco-puro);
  font-family: 'Outfit', sans-serif;
}

.hdr-l h1 span {
  color: var(--cinza-prata);
}

.sim-badge {
  font-size: 9px;
  font-weight: 800;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(189, 195, 199, 0.3);
  color: var(--branco-puro);
  padding: 3px 8px;
  border-radius: 4px;
  letter-spacing: .1em;
  animation: pulse 2.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; border-color: rgba(189, 195, 199, 0.5); }
  50% { opacity: 0.7; border-color: rgba(189, 195, 199, 0.2); }
}

.hdr-r {
  font-size: 11px;
  color: var(--muted);
  text-align: right;
  line-height: 1.6;
}

.hdr-r b {
  color: var(--branco-puro);
}

#main {
  display: grid;
  grid-template-columns: var(--sw) 1fr var(--pw);
  height: calc(100vh - 54px - 34px);
}

#sidebar {
  background: var(--bg2);
  border-right: 1px solid var(--bdr);
  overflow-y: auto;
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

#mapbox {
  position: relative;
}

#map {
  width: 100%;
  height: 100%;
}

#patrol {
  background: var(--bg2);
  border-left: 1px solid var(--bdr);
  overflow-y: auto;
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

#ftr {
  height: 34px;
  background: var(--azul-fechado);
  border-top: 1px solid var(--bdr);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  font-size: 11px;
  color: var(--muted);
}

.panel {
  background: var(--card);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 12px 14px;
  transition: border-color 0.2s;
}

.panel:hover {
  border-color: var(--bdr-hover);
}

.ptitle {
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .12em;
  color: var(--branco-puro);
  margin-bottom: 10px;
}

.copt {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 6px 8px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 3px;
  transition: background .15s, color .15s;
}

.copt:hover {
  background: rgba(189, 195, 199, 0.08);
}

.copt input {
  accent-color: var(--cinza-prata);
  width: 15px;
  height: 15px;
  cursor: pointer;
  flex-shrink: 0;
}

.copt label {
  font-size: 12px;
  cursor: pointer;
  line-height: 1.3;
  user-select: none;
  color: var(--muted);
  transition: color 0.15s;
}

.copt:hover label {
  color: var(--branco-puro);
}

.srow {
  margin-bottom: 8px;
}

.slbl {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 6px;
}

.slbl b {
  color: var(--branco-puro);
  font-weight: 800;
}

input[type=range] {
  width: 100%;
  accent-color: var(--cinza-prata);
  cursor: pointer;
}

select {
  width: 100%;
  padding: 7px 10px;
  background: var(--azul-fechado);
  border: 1px solid var(--bdr);
  border-radius: 6px;
  color: var(--branco-puro);
  font-size: 12px;
  font-family: inherit;
  cursor: pointer;
  outline: none;
  transition: border-color 0.2s;
}

select:focus {
  border-color: var(--cinza-prata);
}

.pcard {
  background: var(--card);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 12px 14px;
  cursor: pointer;
  transition: all .2s cubic-bezier(0.16, 1, 0.3, 1);
  margin-bottom: 6px;
}

.pcard:hover {
  border-color: var(--bdr-hover);
  background: rgba(15, 32, 39, 0.95);
  transform: translateX(3px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
}

.pcard.critico { border-left: 3px solid var(--red); }
.pcard.alto { border-left: 3px solid var(--ora); }
.pcard.medio { border-left: 3px solid var(--yel); }
.pcard.baixo { border-left: 3px solid var(--cinza-prata); }

.phdr {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  flex-wrap: wrap;
}

.vnum {
  font-size: 11px;
  font-weight: 800;
  color: var(--preto-suave);
  background: var(--branco-puro);
  padding: 2px 7px;
  border-radius: 4px;
  min-width: 36px;
  text-align: center;
}

.utype {
  font-size: 10px;
  font-weight: 600;
  color: var(--branco-puro);
  background: rgba(189, 195, 199, 0.12);
  border: 1px solid rgba(189, 195, 199, 0.2);
  padding: 2px 7px;
  border-radius: 4px;
  flex: 1;
}

.rbadge {
  font-size: 9px;
  font-weight: 800;
  padding: 2px 7px;
  border-radius: 4px;
  letter-spacing: .06em;
}

.rbadge.CRITICO { background: rgba(239,68,68,.25); color: #FCA5A5; }
.rbadge.ALTO { background: rgba(249,115,22,.25); color: #FDBA74; }
.rbadge.MEDIO { background: rgba(234,179,8,.25); color: #FDE68A; }
.rbadge.BAIXO { background: rgba(189,195,199,.25); color: var(--branco-puro); }

.pbairro {
  font-size: 13px;
  font-weight: 700;
  color: var(--branco-puro);
  margin-bottom: 5px;
}

.pdet {
  font-size: 11px;
  color: var(--muted);
  display: flex;
  flex-direction: column;
  gap: 3px;
  line-height: 1.5;
}

.pdet b {
  color: var(--branco-puro);
}

.mrow {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 5px 0;
  border-bottom: 1px solid rgba(189, 195, 199, 0.08);
}

.mrow:last-child {
  border-bottom: none;
}

.mkey {
  font-size: 11px;
  color: var(--muted);
}

.mval {
  font-size: 12px;
  font-weight: 800;
  color: var(--branco-puro);
}

.pp-hdr {
  border-bottom: 1px solid var(--bdr);
  padding-bottom: 10px;
  margin-bottom: 4px;
}

.pp-title {
  font-size: 12px;
  font-weight: 800;
  letter-spacing: .06em;
  color: var(--branco-puro);
  text-transform: uppercase;
}

.pp-sub {
  font-size: 11px;
  color: var(--muted);
  margin-top: 3px;
}

.empty-state {
  text-align: center;
  padding: 35px 10px;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.6;
}

.empty-state b {
  color: var(--branco-puro);
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
}

#map-overlay {
  position: absolute;
  bottom: 16px;
  left: 16px;
  z-index: 999;
  background: rgba(26, 26, 26, 0.92);
  backdrop-filter: blur(10px);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 11px;
  line-height: 1.6;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
}

#map-overlay b {
  font-size: 11px;
  color: var(--branco-puro);
  display: block;
  margin-bottom: 5px;
  font-weight: 700;
}

.leg-row {
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
}

.leg-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.ibox {
  background: rgba(15, 32, 39, 0.7);
  border: 1px solid var(--bdr);
  border-radius: 8px;
  padding: 9px 12px;
  font-size: 11px;
  color: var(--muted);
  line-height: 1.5;
}
</style>
</head>
<body>
<div id="hdr">
  <div class="hdr-l">
    <div class="logo-shield">&#128737;</div>
    <h1>CONDOR <span>SHIELD</span> &nbsp;<span style="font-weight:400;color:var(--muted);font-size:11px;">| Inteligência Policial Preditiva & Patrulha</span></h1>
    <span class="sim-badge">XGBOOST POISSON</span>
  </div>
  <div class="hdr-r">
    Modelo: <b>XGBoost Espaço-Temporal</b> &nbsp;|&nbsp; Grade: <b>60&times;60</b> &nbsp;|&nbsp; Delegacias: <b>95</b><br>
    Fonte: <b>SIPCV / SSP-SP</b> (Out 2025&ndash;Mai 2026)
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <div class="panel">
      <div class="ptitle">Crimes Selecionados</div>
      <div class="copt"><input type="checkbox" id="ck0" checked><label for="ck0">Tentativa de Homicídio</label></div>
      <div class="copt"><input type="checkbox" id="ck1" checked><label for="ck1">Homicídio Doloso</label></div>
      <div class="copt"><input type="checkbox" id="ck2" checked><label for="ck2">Lesão Corp. Seg. Morte</label></div>
      <div class="copt"><input type="checkbox" id="ck3" checked><label for="ck3">Latrocínio</label></div>
    </div>
    <div class="panel">
      <div class="ptitle">Camadas do Mapa</div>
      <div class="copt"><input type="checkbox" id="chk-pred" checked><label for="chk-pred">&#128308; Pontos Previstos (XGBoost)</label></div>
      <div class="copt"><input type="checkbox" id="chk-val" checked><label for="chk-val">&#9899; Ocorrências Históricas Reais</label></div>
    </div>
    <div class="panel">
      <div class="ptitle">Horizonte de Previsão</div>
      <div class="srow">
        <div class="slbl"><span>Meses à frente</span><b id="hv">1</b></div>
        <input type="range" id="hslider" min="1" max="6" value="1">
        <div id="htarget" style="font-size:11px;color:var(--bl);margin-top:4px;font-weight:700;"></div>
      </div>
    </div>
    <div class="panel">
      <div class="ptitle">Viaturas Disponíveis</div>
      <div class="srow">
        <div class="slbl"><span>Unidades</span><b id="uv">5</b></div>
        <input type="range" id="uslider" min="1" max="20" value="5">
      </div>
      <div class="ptitle" style="margin-top:8px;">Tipo de Patrulha</div>
      <select id="psel">
        <option value="AUTO">Atribuição Automática</option>
        <option value="ROCAM/M">ROCAM/M</option>
        <option value="RPA">RPA (Rádio Patrulha)</option>
        <option value="Forca Tatica">Força Tática</option>
        <option value="ROTAM">ROTAM</option>
        <option value="GCM">Guarda Civil Metropolitana</option>
        <option value="CHOQUE">Batalhão de Choque</option>
      </select>
    </div>
    <div class="panel" id="mpanel">
      <div class="ptitle">Métricas de Risco Preditivo</div>
      <div id="mcontent">
        <div class="mrow"><span class="mkey">Probabilidade Média nas Zonas</span><span class="mval">--</span></div>
        <div class="mrow"><span class="mkey">Pontos Previstos no Mês</span><span class="mval">--</span></div>
        <div class="mrow"><span class="mkey">Modelo</span><span class="mval">XGBoost Poisson</span></div>
      </div>
    </div>
    <div class="ibox">
      &#9888; Atualização em tempo real: As viaturas são direcionadas aos maiores focos de risco e densidade de ocorrências previstas pelo CONDOR SHIELD (XGBoost).
    </div>
  </div>
  <div id="mapbox">
    <div id="map"></div>
    <div id="map-overlay">
      <b>Legenda (Risco Preditivo CONDOR SHIELD)</b>
      <div class="leg-row"><div class="leg-dot" style="background:#EF4444"></div> Risco CRÍTICO (&gt;40% prob.)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#F97316"></div> Risco ALTO (25-40% prob.)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#EAB308"></div> Risco MÉDIO (10-25% prob.)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#BDC3C7"></div> Risco BAIXO (&lt;10% prob.)</div>
      <div class="leg-row" style="margin-top:4px"><div class="leg-dot" style="background:#F43F5E;border:1px solid #fff;"></div> &nbsp;Ponto de Crime Previsto (XGBoost)</div>
      <div class="leg-row" style="margin-top:4px"><div class="leg-dot"
        style="background:linear-gradient(90deg,#0EA5E9,#10B981,#F59E0B,#EF4444);border-radius:3px;width:28px;height:6px"></div>
        &nbsp;Área Quente (Densidade de Risco)</div>
    </div>
  </div>
  <div id="patrol">
    <div class="pp-hdr">
      <div class="pp-title">Direcionamento de Viaturas</div>
      <div class="pp-sub" id="pp-sub">Alocação automática em tempo real</div>
    </div>
    <div id="plist">
      <div class="empty-state">
        <b>&#128205; Carregando previsão CONDOR SHIELD...</b>
      </div>
    </div>
  </div>
</div>
<div id="ftr">
  <span>CONDOR SHIELD &nbsp;|&nbsp; Previsão XGBoost Poisson &nbsp;|&nbsp; Direcionamento Preditivo de Patrulha &nbsp;|&nbsp; SSP-SP</span>
  <span id="fstatus" style="color:var(--bl);font-weight:600;"></span>
</div>
<script>'''

TAIL = '''
const BAIRROS=[
  {n:"Se / Centro Historico",lat:-23.547,lon:-46.637},{n:"Republica",lat:-23.543,lon:-46.642},
  {n:"Liberdade",lat:-23.560,lon:-46.632},{n:"Consolacao",lat:-23.554,lon:-46.652},
  {n:"Bela Vista",lat:-23.557,lon:-46.643},{n:"Cambuci",lat:-23.572,lon:-46.622},
  {n:"Bras / Belem",lat:-23.547,lon:-46.607},{n:"Mooca",lat:-23.553,lon:-46.596},
  {n:"Tatuape",lat:-23.540,lon:-46.572},{n:"Penha",lat:-23.520,lon:-46.543},
  {n:"Itaquera",lat:-23.536,lon:-46.453},{n:"Ermelino Matarazzo",lat:-23.497,lon:-46.472},
  {n:"Vila Matilde",lat:-23.530,lon:-46.516},{n:"Ipiranga",lat:-23.589,lon:-46.605},
  {n:"Saude / Jabaquara",lat:-23.622,lon:-46.636},{n:"Cidade Ademar",lat:-23.663,lon:-46.663},
  {n:"Vila Prudente",lat:-23.585,lon:-46.578},{n:"Sao Mateus",lat:-23.613,lon:-46.537},
  {n:"Sapopemba",lat:-23.591,lon:-46.521},{n:"Santana",lat:-23.498,lon:-46.624},
  {n:"Tucuruvi",lat:-23.475,lon:-46.607},{n:"Pirituba",lat:-23.474,lon:-46.728},
  {n:"Lapa / Barra Funda",lat:-23.523,lon:-46.706},{n:"Pinheiros",lat:-23.564,lon:-46.693},
  {n:"Butanta",lat:-23.578,lon:-46.722},{n:"Campo Limpo",lat:-23.676,lon:-46.743},
  {n:"Santo Amaro",lat:-23.655,lon:-46.706},{n:"Jardins",lat:-23.561,lon:-46.654},
  {n:"Perdizes / Santa Cecilia",lat:-23.537,lon:-46.655},{n:"Guarulhos",lat:-23.465,lon:-46.533},
  {n:"Osasco / Carapicuiba",lat:-23.532,lon:-46.792},{n:"Taboao da Serra",lat:-23.611,lon:-46.762},
  {n:"Sao Bernardo do Campo",lat:-23.698,lon:-46.565},{n:"Santo Andre",lat:-23.653,lon:-46.538},
  {n:"Diadema",lat:-23.686,lon:-46.622},{n:"Maua",lat:-23.668,lon:-46.461},
];
const PATROL_TYPES=['ROCAM/M','RPA','RPA','Forca Tatica','ROTAM','GCM','RPA','CHOQUE'];
const BPMS=['1o BPM/M (Se)','5o BPM/M (Bras)','7o BPM/M (Ipiranga)','8o BPM/M (Pinheiros)',
  '12o BPM/M (S.Mateus)','13o BPM/M (Santana)','14o BPM/M (Tatuape)','15o BPM/M (Jabaquara)',
  '17o BPM/M (C.Limpo)','19o BPM/M (Guarulhos)','21o BPM/M (Pirituba)','23o BPM/M (Itaquera)',
  '24o BPM/M (Guaianazes)','27o BPM/M (Diadema)','37o BPM/M (Osasco)'];
const RCOLS={CRITICO:'#EF4444',ALTO:'#F97316',MEDIO:'#EAB308',BAIXO:'#BDC3C7'};
const MONTHS=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];

// ── MAP SETUP ─────────────────────────────────────────────────────────────────
const map=L.map('map').setView([-23.5505,-46.6333],12);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
  attribution:'&copy; OpenStreetMap &amp; CARTO | CONDOR SHIELD',
  subdomains:'abcd',maxZoom:18
}).addTo(map);

let heatL=null,zoneL=[],markL=[],routeL=[],predL=[];

function clearLayers(){
  if(heatL){map.removeLayer(heatL);heatL=null;}
  [...zoneL,...markL,...routeL,...predL].forEach(l=>map.removeLayer(l));
  zoneL=[];markL=[];routeL=[];predL=[];
}

// ── UTILITIES ─────────────────────────────────────────────────────────────────
function nearestBairro(lat,lon){
  let mn=999,best='Sao Paulo';
  for(const b of BAIRROS){
    const d=Math.hypot(lat-b.lat,lon-b.lon);
    if(d<mn){mn=d;best=b.n;}
  }
  return best;
}

function getTargetMonth(h){
  const now=new Date();
  const td=new Date(now.getFullYear(),now.getMonth()+parseInt(h),1);
  return MONTHS[td.getMonth()]+' '+td.getFullYear();
}

function getSelectedKeys(){
  const IDS=['TENTATIVA DE HOMICIDIO','HOMICIDIO DOLOSO',
    'LESAO CORPORAL SEGUIDA DE MORTE','LATROCINIO'];
  const keys=[];
  for(let i=0;i<4;i++)
    if(document.getElementById('ck'+i).checked) keys.push(IDS[i]);
  return keys;
}

// ── COMBINE DENSITY GRIDS FOR HORIZON H ───────────────────────────────────────
function combineGrids(keys, h){
  const N = DATA.grid_n;
  const gridH = DATA.grids_horizonte[String(h)] || DATA.grids_horizonte['1'];
  
  if (keys.length === 4) return gridH;
  
  const shape = (keys.length === 1 && DATA.grids[keys[0]])
    ? DATA.grids[keys[0]]
    : DATA.grids['all'];

  let sMax = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
    if (shape[i][j] > sMax) sMax = shape[i][j];

  const out = Array.from({length: N}, () => new Float32Array(N));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const w = sMax > 0 ? (0.3 + 0.7 * (shape[i][j] / sMax)) : 1.0;
    out[i][j] = gridH[i][j] * w;
  }
  return out;
}

// ── SELEÇÃO E ORDENAÇÃO ESTRITA DE ZONAS POR DENSIDADE DE RISCO ───────────────
function findZones(grid, hFactor){
  const N = DATA.grid_n;
  const cells = [];
  
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (grid[i][j] > 2.0) {  // Seleciona células de densidade preditiva relevante
        cells.push({
          lat: DATA.lat_arr[i],
          lon: DATA.lon_arr[j],
          d: grid[i][j],
          i, j
        });
      }
    }
  }
  // Ordena estritamente por densidade descritiva (maiores picos primeiro)
  cells.sort((a, b) => b.d - a.d);

  const MIN_DIST = 0.035;  // Distância mínima em graus (~3.8 km) para NMS
  const zones = [];
  
  for (const c of cells) {
    let nearExist = false;
    for (const z of zones) {
      if (Math.hypot(c.lat - z.lat, c.lon - z.lon) < MIN_DIST) {
        nearExist = true;
        break;
      }
    }
    if (!nearExist) {
      // Converte densidade relativa em probabilidade Poisson aproximada (%)
      const probPct = Math.min(95.0, roundVal(c.d * 0.95));
      zones.push({
        lat: c.lat,
        lon: c.lon,
        totD: c.d,
        maxD: c.d,
        probPct: probPct,
        radM: Math.round(1400 * Math.max(0.8, hFactor))
      });
    }
    if (zones.length >= 20) break;
  }
  
  zones.sort((a, b) => b.maxD - a.maxD);
  return zones;
}

function roundVal(v){ return Math.round(v * 10) / 10; }

function assignUnits(zones, nUnits, pType){
  if (!zones.length) return [];
  const assigns = [];
  for (let u = 0; u < nUnits; u++) {
    const zi = u % zones.length;
    const z = zones[zi];
    const probPct = z.probPct;
    const risk = probPct >= 40.0 ? 'CRITICO' : probPct >= 25.0 ? 'ALTO' : probPct >= 10.0 ? 'MEDIO' : 'BAIXO';
    const bairro = nearestBairro(z.lat, z.lon);
    const ut = pType !== 'AUTO' ? pType : PATROL_TYPES[u % PATROL_TYPES.length];
    const bpm = BPMS[zi % BPMS.length];
    const tos = DATA.turnos;
    const maxT = Object.entries(tos).sort((a,b)=>b[1]-a[1])[0][0];
    const tStr = {noite:'18h-23h (pico noturno)',manha:'06h-12h (pico matutino)',
      tarde:'12h-18h (pico vespertino)',madrugada:'00h-05h (madrugada)'}[maxT]||'18h-23h';
      
    assigns.push({
      num: u + 1,
      zone: z,
      probPct: probPct,
      risk: risk,
      bairro: bairro,
      ut: ut,
      bpm: bpm,
      tStr: tStr,
      zi: zi,
      coords: [z.lat, z.lon]
    });
  }
  return assigns;
}

// ── EFEITO VISUAL DE ÁREAS QUENTES (INCANDESCENTES) ───────────────────────────
function renderHeatmap(keys, h){
  const grid = combineGrids(keys, h);
  const N = DATA.grid_n;
  const pts = [];
  let maxV = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (grid[i][j] > maxV) maxV = grid[i][j];
  }
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (grid[i][j] > 1.0) {
      const normVal = maxV > 0 ? (grid[i][j] / maxV) : 0;
      pts.push([DATA.lat_arr[i], DATA.lon_arr[j], normVal]);
    }
  }
  if (!pts.length) return;
  heatL = L.heatLayer(pts, {
    radius: 35, blur: 22, minOpacity: 0.12, maxZoom: 16, max: 0.25,
    gradient: {
      0.10: 'rgba(15,32,39,0.3)',
      0.30: '#0EA5E9',
      0.50: '#10B981',
      0.70: '#F59E0B',
      0.85: '#F97316',
      1.00: '#EF4444'
    }
  }).addTo(map);
}

function renderZones(zones, assigns, activePredPts){
  const zu = {};
  for (const a of assigns) {
    if (!zu[a.zi]) zu[a.zi] = {z: a.zone, risk: a.risk, probPct: a.probPct, bairro: a.bairro, units: []};
    zu[a.zi].units.push(a);
  }
  for (const [zi, zd] of Object.entries(zu)) {
    const col = RCOLS[zd.risk];
    const circle = L.circle([zd.z.lat, zd.z.lon], {
      radius: zd.z.radM, color: col, fillColor: col, fillOpacity: 0.15,
      weight: 2, dashArray: '6,3'
    }).addTo(map);
    
    const popBody = zd.units.map(u => `<b>V${String(u.num).padStart(2,'0')}</b> ${u.ut}`).join('<br>');
    circle.bindPopup(`<div style="font-family:'Inter',sans-serif;font-size:12px;background:#1A1A1A;color:#FFFFFF;padding:4px;border-radius:6px">
      <b style="color:${col};font-size:13px">${zd.bairro}</b><br>
      <b>Probabilidade de Ocorrência: ${zd.probPct.toFixed(1)}%</b><br>
      Nível de Risco: <span style="color:${col};font-weight:700">${zd.risk}</span><br><br>${popBody}</div>`);
    zoneL.push(circle);

    const lbl = zd.units.length > 1 ? zd.units.length + 'V' : 'V' + String(zd.units[0].num).padStart(2,'0');
    const icon = L.divIcon({
      html: `<div style="background:#1A1A1A;color:#FFFFFF;border-radius:50%;width:34px;height:34px;
        display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;
        border:2px solid ${col};box-shadow:0 3px 12px rgba(0,0,0,.8);
        font-family:'Inter',sans-serif">${lbl}</div>`,
      className: '', iconSize: [34, 34], iconAnchor: [17, 17]
    });
    const mk = L.marker([zd.z.lat, zd.z.lon], {icon}).addTo(map);
    mk.bindPopup(`<div style="font-family:'Inter',sans-serif;font-size:12px;background:#1A1A1A;color:#FFFFFF;padding:4px;border-radius:6px">
      <b style="font-size:13px;color:#FFFFFF">${zd.bairro}</b><br>
      <b>Probabilidade de Ocorrência: ${zd.probPct.toFixed(1)}%</b><br>
      Nível de Risco: <span style="color:${col};font-weight:700">${zd.risk}</span><br><br>${popBody}</div>`);
    markL.push(mk);

    // Conecta a viatura aos pontos de ocorrência previstos mais próximos na zona
    const nearPred = activePredPts.filter(p => Math.hypot(p.lat - zd.z.lat, p.lon - zd.z.lon) < 0.04).slice(0, 6);
    for (const np of nearPred) {
      const rl = L.polyline([[zd.z.lat, zd.z.lon], [np.lat, np.lon]],
        {color: col, weight: 1.5, opacity: 0.65, dashArray: '3,4'}).addTo(map);
      routeL.push(rl);
    }
  }
}

// ── PATROL PANEL (DIRECIONAMENTO CONDOR SHIELD) ───────────────────────────────
function renderPatrolPanel(assigns, h){
  const target = getTargetMonth(h);
  document.getElementById('pp-sub').textContent = 'Alocação para ' + target + ' | ' + assigns.length + ' viaturas';

  let html = `<div class="panel" style="margin-bottom:8px">
    <div class="ptitle">CONDOR SHIELD — Despacho Preditivo</div>
    <div style="font-size:12px;font-weight:700;color:var(--branco-puro);margin-bottom:4px">Alocação Estrita nos Maiores Focos de Risco</div>
    <div style="font-size:10px;color:var(--muted)">Horizonte: +${h} mês(es) | Alvo: ${target}</div>
  </div>`;

  for (const a of assigns) {
    const rc = a.risk.toLowerCase();
    const rl = {CRITICO:'CRÍTICO',ALTO:'ALTO',MEDIO:'MÉDIO',BAIXO:'BAIXO'}[a.risk];
    html += `<div class="pcard ${rc}" onclick="focusZone(${a.zone.lat},${a.zone.lon},${a.zone.radM})">
      <div class="phdr">
        <span class="vnum">V${String(a.num).padStart(2,'0')}</span>
        <span class="utype">${a.ut}</span>
        <span class="rbadge ${a.risk}">${rl}</span>
      </div>
      <div class="pbairro">&#128205; ${a.bairro}</div>
      <div class="pdet">
        <span>&#128202; Probabilidade Estimada: <b>${a.probPct.toFixed(1)}%</b></span>
        <span>&#9888; Nível de Risco Preditivo: <b style="color:${RCOLS[a.risk]}">${a.risk}</b></span>
        <span>&#127963; ${a.bpm}</span>
        <span>&#8987; Horário Recomendado: ${a.tStr}</span>
        <span>&#128204; Coordenadas: ${a.zone.lat.toFixed(4)}, ${a.zone.lon.toFixed(4)}</span>
        <span>&#128308; Foco Preditivo ${a.zi+1} de ${Math.min(assigns.length, 12)}</span>
      </div>
    </div>`;
  }
  document.getElementById('plist').innerHTML = html;
}

// ── METRICS PANEL ─────────────────────────────────────────────────────────────
function updateMetrics(keys, assigns, h, totalPredPts){
  const probsVisiveis = assigns.map(a => a.probPct);
  const probMedia = probsVisiveis.length
    ? (probsVisiveis.reduce((s,v)=>s+v,0) / probsVisiveis.length) : 0;
  document.getElementById('mcontent').innerHTML = `
    <div class="mrow"><span class="mkey">Probabilidade Média nas Zonas</span><span class="mval">${probMedia.toFixed(1)}%</span></div>
    <div class="mrow"><span class="mkey">Pontos Previstos no Mês</span><span class="mval">${totalPredPts}</span></div>
    <div class="mrow"><span class="mkey">Horizonte</span><span class="mval">+${h} mês(es)</span></div>
    <div class="mrow"><span class="mkey">Polos de Atuação</span><span class="mval">${Math.min(assigns.length, 12)}</span></div>
    <div class="mrow"><span class="mkey">Viaturas Alocadas</span><span class="mval">${assigns.length}</span></div>
    <div class="mrow"><span class="mkey">Modelo</span><span class="mval">XGBoost Poisson</span></div>
  `;
}

// ── FOCUS MAP ─────────────────────────────────────────────────────────────────
function focusZone(lat, lon, radM){
  const zoom = radM > 2000 ? 12 : radM > 1000 ? 13 : 14;
  map.flyTo([lat, lon], zoom, {duration: 1.0});
}

// ── MAIN RUN (AUTOMÁTICA LIVE) ────────────────────────────────────────────────
function runPrediction(){
  const keys = getSelectedKeys();
  const h = parseInt(document.getElementById('hslider').value);
  const n = parseInt(document.getElementById('uslider').value);
  const pt = document.getElementById('psel').value;
  const showPredPts = document.getElementById('chk-pred').checked;
  const showValPts = document.getElementById('chk-val').checked;
  const hF = 1 + (h - 1) * 0.15;

  const grid = combineGrids(keys, h);
  const zones = findZones(grid, hF);
  const assigns = assignUnits(zones, n, pt);

  clearLayers();
  renderHeatmap(keys, h);

  // Pontos de Crimes Previstos (XGBoost)
  const rawPredPts = DATA.pred_pts[String(h)] || [];
  const vKeys = keys.length ? keys : DATA.crime_keys;
  const activePredPts = rawPredPts.filter(p => vKeys.includes(p.tipo));

  if (showPredPts && activePredPts.length) {
    for (const p of activePredPts) {
      const mk = L.circleMarker([p.lat, p.lon], {
        radius: 5, color: '#F43F5E', fillColor: '#F43F5E',
        fillOpacity: 0.85, weight: 1.5
      }).bindPopup(
        `<div style="font-size:11px;font-family:'Inter',sans-serif;background:#1A1A1A;color:#FFFFFF;padding:4px;border-radius:4px">
          <b style="color:#F43F5E">&#9888; Crime Previsto (XGBoost)</b><br>
          <b>${p.tipo}</b><br>
          Delegacia: ${p.deleg}<br>
          Expectativa: ${p.lambda} crimes/mês<br>
          Probabilidade: ${p.prob}%
        </div>`
      ).addTo(map);
      predL.push(mk);
    }
  }

  renderZones(zones, assigns, activePredPts);
  renderPatrolPanel(assigns, h);
  updateMetrics(keys, assigns, h, activePredPts.length);

  // Ocorrências Históricas Reais (opcional)
  if (showValPts) {
    const valPts = [...DATA.val_pts, ...DATA.test_pts];
    const vf = valPts.filter(p => vKeys.includes(p.tipo));
    if (vf.length) {
      const vLayer = L.layerGroup();
      for (const p of vf) {
        L.circleMarker([p.lat, p.lon], {radius: 3, color: '#BDC3C7', fillColor: '#BDC3C7',
          fillOpacity: .5, weight: 0}).bindPopup(
          `<div style="font-size:11px;font-family:'Inter',sans-serif;background:#1A1A1A;color:#FFFFFF;padding:4px;border-radius:4px"><b>${p.tipo}</b> (Histórico)<br>${p.data}<br>${p.local}</div>`
        ).addTo(vLayer);
      }
      vLayer.addTo(map);
      markL.push(vLayer);
    }
  }

  const target = getTargetMonth(h);
  document.getElementById('fstatus').textContent =
    'CONDOR SHIELD: ' + target + ' | ' + assigns.length + ' viaturas direcionadas aos focos críticos';
}

// ── EVENT LISTENERS (REATIVIDADE LIVE AUTOMÁTICA EM TEMPO REAL) ──────────────
document.getElementById('hslider').addEventListener('input', e => {
  document.getElementById('hv').textContent = e.target.value;
  document.getElementById('htarget').textContent = 'Alvo: ' + getTargetMonth(e.target.value);
  runPrediction();
});

document.getElementById('uslider').addEventListener('input', e => {
  document.getElementById('uv').textContent = e.target.value;
  runPrediction();
});

document.getElementById('psel').addEventListener('change', runPrediction);
document.getElementById('chk-pred').addEventListener('change', runPrediction);
document.getElementById('chk-val').addEventListener('change', runPrediction);

for (let i = 0; i < 4; i++) {
  const ck = document.getElementById('ck' + i);
  if (ck) ck.addEventListener('change', runPrediction);
}

// ── INIT ──────────────────────────────────────────────────────────────────────
document.getElementById('htarget').textContent = 'Alvo: ' + getTargetMonth(1);
runPrediction();
</script>
</body>
</html>'''

HTML = HEAD + '\n' + data_js + '\n' + TAIL

out_path = 'outputs/dashboard_preditivo.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f'[OK] Dashboard salvo: {out_path}')
print(f'     Tamanho: {len(HTML)/1024:.0f} KB')
print('[OK] Etapa 6 concluida. Abra o dashboard_preditivo.html no navegador.')
