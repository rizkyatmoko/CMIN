"""Execute the validation notebook's analysis cells (V2-V6) locally against REAL price,
weather and policy data but SYNTHETIC model outputs.

Two things are being checked.

  1. Do the cells run at all?  A NameError or a shape bug found here costs nothing; found
     after a Colab run it costs the whole run.

  2. Can the tests detect a signal that is really there?  Arm 'planted' gets a latent that
     genuinely encodes the factor and an attribution whose weather channel genuinely spikes
     near extreme-rain days.  Arm 'collapsed' gets the pathology the June run actually has.
     If the planted arm does not come out positive, the test is underpowered and the
     experiment is not worth running.
"""
import json, os, sys, numpy as np, pandas as pd

ROOT = r'C:\Users\HP\Documents\Claude'
NB   = os.path.join(ROOT, 'cmin_protocol', 'CMIN_validation.ipynb')
VAL_DIR = os.path.join(ROOT, 'cmin_validation', '_dryrun_out')
os.makedirs(VAL_DIR, exist_ok=True)
PROTO_DIR = VAL_DIR   # V6 prints it; the real notebook gets it from cell 1
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- real data
CACHE = os.path.join(ROOT, 'cmin_validation', '_cache_df.parquet')
if os.path.exists(CACHE):
    df = pd.read_parquet(CACHE).sort_values('tanggal')
else:
    tr = pd.read_excel(os.path.join(ROOT, 'dataset_train.xlsx'))
    te = pd.read_excel(os.path.join(ROOT, 'dataset_test.xlsx'))
    for d in (tr, te):
        d['tanggal'] = pd.to_datetime(d['tanggal'])
    df = pd.concat([tr, te], ignore_index=True).sort_values('tanggal')
    na = pd.read_csv(os.path.join(ROOT, 'CMIM', 'CMIN 2',
                                  'nasa_power_jawa_timur_merged.csv'),
                     usecols=['wilayah', 'date', 'PRECTOTCORR'], parse_dates=['date'])
    na = na.rename(columns={'wilayah': 'kab_kota', 'date': 'tanggal',
                            'PRECTOTCORR': 'nasa_PRECTOTCORR'})
    df = df.merge(na, on=['kab_kota', 'tanggal'], how='left')
    df[['tanggal', 'kab_kota', 'pasar', 'harga', 'Ramadan', 'Panen tahun lalu',
        'nasa_PRECTOTCORR', 'Harga BBM (Pertalite)', 'Harga BBM (Pertamax)',
        'Harga BBM (Solar Subsidi)']].to_parquet(CACHE, index=False)
    df = pd.read_parquet(CACHE)
policy_events_all = pd.read_csv(os.path.join(ROOT, r'CMIM\CMIN 2\policy_event_features.csv'),
                                parse_dates=['event_date'])
dates = pd.DatetimeIndex(np.sort(df['tanggal'].unique()))
TEST_START = pd.Timestamp('2025-02-28')
T = len(dates)

# ---------------------------------------------------------------- fake the run context
L, H = 30, 30
ALL_ORIGINS = list(range(L - 1, T - H))
ORIG_DATE = pd.DatetimeIndex([dates[t] for t in ALL_ORIGINS])
IS_TEST = np.array([dates[t + 1] >= TEST_START for t in ALL_ORIGINS])
n = len(ALL_ORIGINS)
DOMS = ['price', 'weather', 'supply', 'economy']

rain = df.groupby('tanggal')['nasa_PRECTOTCORR'].mean().reindex(dates)
istr = dates < TEST_START
peak_rain_flag = (rain >= np.nanquantile(rain[istr], 0.75)).astype(int).to_numpy()
hv = df.groupby('tanggal')['Panen tahun lalu'].mean().reindex(dates)
harvest_flag = (hv >= np.nanquantile(hv[istr], 0.75)).astype(int).to_numpy()
ramadan_flag = df.groupby('tanggal')['Ramadan'].max().reindex(dates).fillna(0).astype(int).to_numpy()

# the true factor, used only to build the planted latent
city_lp = (df.groupby(['tanggal', 'kab_kota'])['harga']
             .apply(lambda s: np.log(s.dropna()).mean()).unstack('kab_kota')
             .sort_index().reindex(dates).ffill(limit=7))
Rr = city_lp.shift(-14) - city_lp
Ztr = ((Rr - Rr[istr].mean()) / Rr[istr].std(ddof=0))
w, V = np.linalg.eigh(np.cov(Ztr[istr & Rr.notna().all(axis=1)].to_numpy(), rowvar=False))
Ftrue = pd.Series(Ztr.to_numpy() @ V[:, -1], index=dates).reindex(ORIG_DATE).fillna(0).to_numpy()

rain_o = rain.reindex(ORIG_DATE).to_numpy()
rain_hi = rain_o >= np.nanquantile(rain[istr], 0.95)

EXP1_ARMS = [('planted@R', 'latent that really encodes the factor'),
             ('noise@R',   'latent that does not')]
EXP2_ARMS = [('planted@S', 'attribution that really tracks rain'),
             ('collapsed@S', 'one scalar wearing four labels (the June pathology)')]
NEED = ['planted@R', 'noise@R', 'planted@S', 'collapsed@S']
AVAIL = {k: list(range(6)) for k in NEED}

lat, rows = {}, []
for nm in NEED:
    for sd in AVAIL[nm]:
        planted = nm.startswith('planted')
        base = rng.normal(size=(n, 64))
        if planted:
            base[:, 0] = Ftrue + rng.normal(scale=0.8, size=n)     # weak but real
        lat[f'{nm}|{sd}|lat'] = base
        lat[f'{nm}|{sd}|enc'] = rng.normal(size=(n, 64))
        A = rng.normal(scale=0.4, size=(n, 4))
        if nm.startswith('collapsed'):
            s1 = rng.normal(size=n)
            A = np.stack([-s1, s1, s1, s1], 1) + rng.normal(scale=0.01, size=(n, 4))
        elif planted:
            A[:, 1] += 3.0 * rain_hi                               # weather channel only
            A[:, 3] += 2.0 * ramadan_flag[[t for t in ALL_ORIGINS]]
        P = np.exp(A) / np.exp(A).sum(1, keepdims=True)
        d = pd.DataFrame({'rung': nm, 'seed': sd, 'origin_date': ORIG_DATE, 'is_test': IS_TEST})
        for j, dm in enumerate(DOMS):
            d['attr_' + dm] = P[:, j]
            d['ctxw_' + dm] = P[:, j]
        for k in ('dens', 'ent', 'alpha', 'temp', 'gate'):
            d[k] = rng.normal(0.5, 0.05, n) + (0.1 * peak_rain_flag[[t for t in ALL_ORIGINS]]
                                               if k == 'dens' else 0)
        rows.append(d)
DAILY = pd.concat(rows, ignore_index=True)
np.savez_compressed(os.path.join(VAL_DIR, 'latents.npz'), **lat)
EX = {(nm, sd): None for nm in NEED for sd in AVAIL[nm]}

# ---------------------------------------------------------------- run the real cells
cells = json.load(open(NB, encoding='utf-8'))['cells']
code = [''.join(c['source']) for c in cells]
V = {t.split('.')[0].replace('# === ', ''): i
     for i, t in enumerate(code) if t.startswith('# === V')}
print('found cells:', V, '\n')
g = dict(globals())
for tag in ('V2', 'V3', 'V4', 'V5', 'V6'):
    print('\n' + '#' * 78 + f'\n#  {tag}\n' + '#' * 78)
    try:
        exec(compile(code[V[tag]], f'<{tag}>', 'exec'), g)
    except Exception as e:
        import traceback
        print(f'\n!!! {tag} FAILED: {type(e).__name__}: {e}')
        traceback.print_exc(limit=6)
        sys.exit(1)
print('\n\nALL CELLS RAN.')
