"""Run CMIN_final.ipynb's analysis cells (F0, F2a, F3-F12) locally.

Real: the price panel, NASA POWER rain, Ramadan and fuel columns, the policy event file.
Synthetic: the protocol arrays (Y, obs, mu, sd, S, A_phys) and every extraction file F2b
would write.  F1 (training) and F2b (inference) need the checkpoints and are skipped.

What this proves: every analysis cell runs on files of the exact shape F2b writes, and the
global-scale guard holds through all of them.  What it does not prove: anything about the
real results.
"""
import json, os, sys, shutil, time
import numpy as np, pandas as pd

ROOT = r'C:\Users\HP\Documents\Claude'
NB = os.path.join(ROOT, 'cmin_protocol', 'CMIN_final.ipynb')
PROTO_DIR = os.path.join(ROOT, 'cmin_validation', '_dry_final')
shutil.rmtree(PROTO_DIR, ignore_errors=True)
os.makedirs(PROTO_DIR)
rng = np.random.default_rng(1)
DRY_RUN = True
OUT_DIR = os.path.dirname(PROTO_DIR)   # F0 looks here for cmin_original_checkpoints.zip
DEVICE = 'cpu'

# ---------------------------------------------------------------- real data
CACHE = os.path.join(ROOT, 'cmin_validation', '_cache_df.parquet')
if not os.path.exists(CACHE):
    tr = pd.read_excel(os.path.join(ROOT, 'dataset_train.xlsx'))
    te = pd.read_excel(os.path.join(ROOT, 'dataset_test.xlsx'))
    d_ = pd.concat([tr, te], ignore_index=True); d_['tanggal'] = pd.to_datetime(d_['tanggal'])
    na = pd.read_csv(os.path.join(ROOT, 'CMIM', 'CMIN 2', 'nasa_power_jawa_timur_merged.csv'),
                     usecols=['wilayah', 'date', 'PRECTOTCORR'], parse_dates=['date'])
    na = na.rename(columns={'wilayah': 'kab_kota', 'date': 'tanggal', 'PRECTOTCORR': 'nasa_PRECTOTCORR'})
    d_ = d_.merge(na, on=['kab_kota', 'tanggal'], how='left')
    d_[['tanggal', 'kab_kota', 'pasar', 'harga', 'Ramadan', 'Panen tahun lalu', 'nasa_PRECTOTCORR',
        'Harga BBM (Pertalite)', 'Harga BBM (Pertamax)', 'Harga BBM (Solar Subsidi)']].to_parquet(CACHE, index=False)
df = pd.read_parquet(CACHE).sort_values('tanggal')
policy_events_all = pd.read_csv(os.path.join(ROOT, 'CMIM', 'CMIN 2', 'policy_event_features.csv'),
                                parse_dates=['event_date'])
dates = pd.DatetimeIndex(np.sort(df.tanggal.unique()))
TEST_START = pd.Timestamp('2025-02-28')
T = len(dates)
cities = sorted(df.kab_kota.unique()); R = len(cities)
markets = df.drop_duplicates(['kab_kota', 'pasar'])[['kab_kota', 'pasar']].reset_index(drop=True)
N = len(markets)

# ---------------------------------------------------------------- synthetic protocol arrays
CFG = dict(H=30, L=30, HORIZONS=[1, 3, 7, 14, 30], d=64, d_ctx=32, n_heads=4)
SCFG = dict(seeds=10, epochs=50)
HEPOCHS = 50
Y = np.cumsum(rng.normal(0, 0.05, (T, N)), 0); Y = (Y - Y.mean(0)) / Y.std(0)
obs = (rng.random((T, N)) > 0.02).astype(np.float32)
mu = np.log(rng.uniform(30000, 60000, N)); sd = rng.uniform(0.2, 0.4, N)
S = np.zeros((N, R), np.float32)
for i, c in enumerate(markets.kab_kota):
    S[i, cities.index(c)] = 1
A_phys = rng.random((R, R)); np.fill_diagonal(A_phys, 0); A_phys /= A_phys.sum(1, keepdims=True)


def RP(z):
    return np.exp(z * sd[None, :, None] + mu[None, :, None])


origins = list(range(CFG['L'] - 1, T - CFG['H']))
TEST_ORIGINS = [t for t in origins if dates[t + 1] >= TEST_START]
pool = [t for t in origins if dates[t + 1] < TEST_START]
nval = max(30, int(0.1 * len(pool)))
train_o, val_o = pool[:-nval], pool[-nval:]
rain = df.groupby('tanggal').nasa_PRECTOTCORR.mean().reindex(dates)
ist = dates < TEST_START
peak_rain_flag = (rain >= np.nanquantile(rain[ist], 0.75)).astype(int).to_numpy()
hv = df.groupby('tanggal')['Panen tahun lalu'].mean().reindex(dates)
harvest_flag = (hv >= np.nanquantile(hv[ist], 0.75)).astype(int).to_numpy()
ramadan_flag = df.groupby('tanggal').Ramadan.max().reindex(dates).fillna(0).astype(int).to_numpy()

LADDER = {k: {} for k in ('L1_encoder_only', 'L3_learned_static', 'L6_ctx_conditioned', 'L7_ctx_shuffled')}
rung_name = lambda b, h: '%s@%s' % (b, h)
hier_name = lambda k, h: 'H_%s@%s' % (k, h)
rung_ckpt = lambda n, s: os.path.join(PROTO_DIR, 'w_%s__s%d.pt' % (n, s))
build_ladder_model = lambda cfg, head='R': (None, None)
train_ladder_rung = lambda *a, **k: None
Sbar_t = S_t = Aphys_t = None

# ---------------------------------------------------------------- run the cells
cells = json.load(open(NB, encoding='utf-8'))['cells']
code = {''.join(c['source']).split('.')[0].replace('# === ', ''): ''.join(c['source'])
        for c in cells if ''.join(c['source']).startswith('# === F')}
g = dict(globals())


def run(tag):
    t0 = time.time()
    print('\n' + '#' * 78 + '\n#  ' + tag + '\n' + '#' * 78)
    exec(compile(code[tag], '<' + tag + '>', 'exec'), g)
    print('  [%s ran in %.1fs]' % (tag, time.time() - t0))


run('F0')
run('F2a')

# ---------------------------------------------------------------- synthetic F2b output
EXT_DIR, n_all, n_te = g['EXT_DIR'], len(g['ALL_ORIGINS']), len(TEST_ORIGINS)
base_ctx = rng.random((R, R)); base_ctx /= base_ctx.sum(1, keepdims=True)
lab_te = g['regime_labels'](TEST_ORIGINS)


def fake(nm, seed, full, trained=True):
    pub = g['PUBLISHED'].get(nm, [2100, 4700, 7500, 11000, 15500])
    ctx = (base_ctx + rng.normal(0, 0.002, (R, R))) if trained else rng.dirichlet(np.ones(R), R)
    ctx = np.clip(ctx, 1e-6, None); ctx /= ctx.sum(1, keepdims=True)
    al = 0.45
    pay = dict(version=np.array(g['EXT_VERSION']),
               win_mae=rng.normal(pub[2], 300, (5, n_te)).astype(np.float32),
               mae=np.array(pub, float) * rng.normal(1, 0.01, 5),
               A_mean=(al * A_phys + (1 - al) * ctx).astype(np.float32),
               Actx_mean=ctx.astype(np.float32), n_params=np.array(184636))
    if full:
        pay.update(lat=rng.normal(size=(n_all, 64)).astype(np.float32),
                   enc=rng.normal(size=(n_all, 64)).astype(np.float32),
                   attr=rng.dirichlet(np.ones(4), n_all).astype(np.float32),
                   attr_sd=rng.uniform(0.001, 0.004, (n_all, 4)).astype(np.float32),
                   cw=rng.dirichlet(np.ones(4), n_all).astype(np.float32),
                   eff=rng.normal(25, 1, n_all).astype(np.float32),
                   alpha=rng.normal(0.45, 0.01, n_all).astype(np.float32),
                   temp=rng.normal(0.74, 0.005, n_all).astype(np.float32),
                   gate=rng.normal(0.55, 0.01, n_all).astype(np.float32),
                   domains=np.array(['price', 'weather', 'supply', 'economy']))
    if nm in g['PER_ORIGIN_A']:
        At = np.repeat(pay['A_mean'][None], n_te, 0) + rng.normal(0, 1e-3, (n_te, R, R))
        At[lab_te == 1] += 2e-3                          # a small planted regime response
        pay['A_test'] = At.astype(np.float16)
    np.savez_compressed(os.path.join(EXT_DIR, '%s__s%d.npz' % (nm, seed)), **pay)


for nm in g['FULL']:
    for s in range(10):
        fake(nm, s, True)
for nm in g['ACC'] + g['PROV']:
    for s in range(10):
        fake(nm, s, False)
for s in g['UNTRAINED_SEEDS']:
    fake(g['UNTRAINED'], s, False, trained=False)
print('\nwrote %d synthetic extraction files' % len(os.listdir(EXT_DIR)))

for tag in ('F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'):
    try:
        run(tag)
    except Exception:
        import traceback
        traceback.print_exc()
        print('\n!!! %s FAILED' % tag)
        sys.exit(1)
g['_scale_intact']()
print('\nALL ANALYSIS CELLS RAN; the price-scale guard still holds.')
