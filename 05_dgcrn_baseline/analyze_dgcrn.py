"""DGCRN results into the paper's own inference, computed locally (17 Sep 2026).

Why local: on Colab, D3 found no reference predictions for most models (the Drive protocol folder
holds only seed-0 files, some of them from later non-protocol runs).  The ORIGINAL ten-seed
predictions behind the published tables are in the 1 September Drive exports in Downloads.

Steps
  1. Run the protocol definition cells of CMIN_final.ipynb (data, features, metric suite) on CPU.
     Nothing is trained.
  2. Score the original reference predictions and check they reproduce metrics_long.csv, the
     source of the published tables, to the rupiah.
  3. Score the 20 DGCRN runs from their .pkl files and check them against the Colab
     dgcrn_metrics_long.csv and window_mae.npz.
  4. Re-run the capacity family exactly as H4/H5 did (paired_boot with default_rng(0) per call,
     TOST at 2%, BH over the family), first on the original 60 comparisons to prove the
     reimplementation reproduces contrast_capacity.csv, then with DGCRN_matched added (70).
  5. Seed-paired t-tests of DGCRN against CMIN, CMIN-static and the matched graph baselines.
Writes dgcrn_analysis.json and dgcrn_contrasts_capacity70.csv next to this script.
"""
import glob, io, json, os, pickle, re, sys, time, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = r'C:\Users\HP\Documents\Claude\cmin_local'
NB = r'C:\Users\HP\Documents\Claude\cmin_protocol\CMIN_final.ipynb'
DATA = r'C:\Users\HP\Documents\Claude\cmin_report\data'
DG_ZIP = r'C:\Users\HP\Downloads\dgcrn-20260917T005941Z-1-001.zip'
EXPORTS = sorted(glob.glob(r'C:\Users\HP\Downloads\Compressed\protocol-20260901*.zip')
                 + glob.glob(r'C:\Users\HP\Downloads\cmin_v4_out-20260901*.zip'))

# ------------------------------------------------------------------ 1. protocol definitions
import torch
torch.backends.mkldnn.enabled = False
os.chdir(LOCAL)
cells = json.load(open(NB, encoding='utf-8'))['cells']
g = {'__name__': '__main__'}
t0 = time.time()
for c in cells:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    first = src.strip().split('\n')[0]
    if first.startswith('# === F0'):
        break
    exec(compile(src, '<cell>', 'exec'), g)
print('definitions loaded in %.0fs; test origins %d' % (time.time() - t0, len(g['TEST_ORIGINS'])))
os.chdir(HERE)

RP, Y, obs, CFG, TEST_ORIGINS, metric_block = (g[k] for k in ('RP', 'Y', 'obs', 'CFG', 'TEST_ORIGINS', 'metric_block'))
HZ = list(CFG['HORIZONS'])
_YT = RP(np.stack([Y[t + 1:t + 1 + CFG['H']].T for t in TEST_ORIGINS]))
_OB = np.stack([obs[t + 1:t + 1 + CFG['H']].T for t in TEST_ORIGINS])


def win_mae(P):
    pred = RP(P)
    out = []
    for h in HZ:
        k = h - 1
        e = np.abs(pred[:, :, k] - _YT[:, :, k]) * _OB[:, :, k]
        out.append(e.sum(1) / np.maximum(_OB[:, :, k].sum(1), 1.0))
    return np.stack(out)


def load_pred(raw):
    try:
        return pickle.loads(raw)['pred']
    except Exception:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return pickle.loads(z.read(z.namelist()[0]))['pred']


# ------------------------------------------------------------------ 2. reference predictions
REFS = ['H_real@R', 'H_real@S', 'L3_learned_static@R', 'L4_price_dynamic@R', 'LSTM_matched', 'GRU_matched',
        'TCN_matched', 'STGCN_matched', 'AGCRN_matched', 'GWNet_matched', 'AGCRN_style', 'GWNet_style',
        'STGCN_phys', 'NHiTS', 'TCN', 'GRU', 'LSTM']
index = {}
for f in EXPORTS:
    z = zipfile.ZipFile(f)
    for i in z.infolist():
        m = re.match(r'(?:.*/)?protocol/([^/]+)__s(\d+)\.pkl(\.zip)?$', i.filename)
        if m and m.group(1) in REFS and '_smoke' not in i.filename:
            key = (m.group(1), int(m.group(2)))
            if key not in index or i.date_time > index[key][2]:
                index[key] = (f, i.filename, i.date_time)
ML = pd.read_csv(os.path.join(DATA, 'metrics_long.csv'))
WIN, MAE, check = {}, {}, []
for key in sorted(index):
    f, name, _ = index[key]
    P = load_pred(zipfile.ZipFile(f).read(name))
    WIN[key] = win_mae(P)
    mb = metric_block(P, TEST_ORIGINS)
    for h in HZ:
        MAE[(key[0], key[1], h)] = mb[h]['MAE']
        pub = ML[(ML.model == key[0]) & (ML.seed == key[1]) & (ML.h == h)].MAE
        check.append(dict(model=key[0], seed=key[1], h=h, ours=mb[h]['MAE'], published=float(pub.iloc[0]) if len(pub) else np.nan))
CHK = pd.DataFrame(check)
CHK['absdiff'] = (CHK.ours - CHK.published).abs()
print('references scored: %d model-seeds; max |MAE - metrics_long| = %.4f IDR'
      % (len(WIN), CHK.absdiff.max()))
bad = CHK[CHK.absdiff > 0.5]
if len(bad):
    print('!! references that do NOT reproduce metrics_long:')
    print(bad.groupby('model').absdiff.max())

# ------------------------------------------------------------------ 3. DGCRN predictions
DGM = pd.read_csv(os.path.join(HERE, 'dgcrn_metrics_long.csv'))
npz = np.load(os.path.join(HERE, 'window_mae.npz'))
z = zipfile.ZipFile(DG_ZIP)
dg_check = []
for i in z.infolist():
    m = re.match(r'dgcrn/(DGCRN(?:_matched)?)__s(\d+)\.pkl$', i.filename)
    if not m:
        continue
    key = (m.group(1), int(m.group(2)))
    P = load_pred(z.read(i.filename))
    WIN[key] = win_mae(P)
    mb = metric_block(P, TEST_ORIGINS)
    for h in HZ:
        MAE[(key[0], key[1], h)] = mb[h]['MAE']
        colab = DGM[(DGM.model == key[0]) & (DGM.seed == key[1]) & (DGM.h == h)].MAE.iloc[0]
        dg_check.append(abs(mb[h]['MAE'] - colab))
    dg_check.append(float(np.abs(WIN[key] - npz['%s|%d' % key]).max()))
print('DGCRN runs scored: %d; max |local - Colab| over MAE and window MAE = %.4f'
      % (sum(1 for k in WIN if k[0].startswith('DGCRN')), max(dg_check)))

# ------------------------------------------------------------------ 4. capacity family, as H4/H5
MARGIN = 2.0


def paired_boot(a, b, h, B=1000):
    rng = np.random.default_rng(0)
    hi = HZ.index(h)
    sa = sorted(s for (m, s) in WIN if m == a)
    sb = sorted(s for (m, s) in WIN if m == b)
    A = np.stack([WIN[(a, s)][hi] for s in sa])
    Bm = np.stack([WIN[(b, s)][hi] for s in sb])
    T = A.shape[1]
    draws = np.empty(B)
    for i in range(B):
        wi = rng.integers(0, T, T)
        ai = rng.integers(0, len(sa), len(sa))
        bi = rng.integers(0, len(sb), len(sb))
        draws[i] = A[np.ix_(ai, wi)].mean() - Bm[np.ix_(bi, wi)].mean()
    lo, hi_ = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(diff=float(A.mean() - Bm.mean()), lo=float(lo), hi=float(hi_), p=float(min(p, 1.0)), ref=float(Bm.mean()))


def verdict(r):
    m = MARGIN / 100 * r['ref']
    inside = (r['lo'] > -m) and (r['hi'] < m)
    excl0 = (r['lo'] > 0) or (r['hi'] < 0)
    return 'equivalent' if inside else ('different' if excl0 else 'inconclusive')


def bh(p, q=0.05):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    passed = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool); rej[o[:k]] = True
    return rej


def family(pairs):
    rows = []
    for a, b in pairs:
        for h in HZ:
            r = paired_boot(a, b, h)
            rows.append(dict(a=a, b=b, h=h, pct=100 * r['diff'] / r['ref'], verdict=verdict(r), **r))
    D = pd.DataFrame(rows)
    D['bh_reject'] = bh(D.p.values)
    D.loc[(D.verdict == 'different') & ~D.bh_reject, 'verdict'] = 'inconclusive'
    return D


orig_pairs = [(hd, b) for hd in ('H_real@R', 'H_real@S')
              for b in ('LSTM_matched', 'GRU_matched', 'TCN_matched', 'STGCN_matched', 'AGCRN_matched', 'GWNet_matched')]
C60 = family(orig_pairs)
PUB = pd.read_csv(os.path.join(DATA, 'contrast_capacity.csv'))
mm = C60.merge(PUB, on=['a', 'b', 'h'], suffixes=('', '_pub'))
print('capacity family, 60 original comparisons: max |p - published p| = %.4f; verdicts identical: %s'
      % ((mm.p - mm.p_pub).abs().max(), bool((mm.verdict == mm.verdict_pub).all())))
C70 = family(orig_pairs + [(hd, 'DGCRN_matched') for hd in ('H_real@R', 'H_real@S')])
C70.to_csv(os.path.join(HERE, 'dgcrn_contrasts_capacity70.csv'), index=False)
print('\ncapacity family with DGCRN_matched (70):')
print(C70.verdict.value_counts().to_string())
changed = C70.merge(C60[['a', 'b', 'h', 'verdict']], on=['a', 'b', 'h'], how='left', suffixes=('', '_60'))
changed = changed[changed.verdict_60.notna() & (changed.verdict != changed.verdict_60)]
print('original verdicts changed by the larger family: %d' % len(changed))
print(C70[C70.b == 'DGCRN_matched'][['a', 'h', 'pct', 'lo', 'hi', 'p', 'verdict']].round(3).to_string(index=False))

# ------------------------------------------------------------------ 5. seed-paired t-tests
from scipy import stats
rows = []
for a in ('DGCRN', 'DGCRN_matched'):
    for b in ('H_real@R', 'L3_learned_static@R', 'L4_price_dynamic@R', 'AGCRN_matched', 'GWNet_matched',
              'STGCN_matched', 'AGCRN_style', 'GWNet_style'):
        for h in HZ:
            sa = np.array([MAE[(a, s, h)] for s in range(10)])
            sb = np.array([MAE[(b, s, h)] for s in range(10)])
            rows.append(dict(a=a, b=b, h=h, mae_a=sa.mean(), mae_b=sb.mean(), pct=100 * (sa.mean() / sb.mean() - 1),
                             t_p=float(stats.ttest_rel(sa, sb).pvalue)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, 'dgcrn_seed_paired.csv'), index=False)
pd.set_option('display.width', 200)
print('\nseed-paired (positive pct = DGCRN worse):')
print(T.pivot_table(index=['a', 'b'], columns='h', values='pct').round(2).to_string())
print(T.pivot_table(index=['a', 'b'], columns='h', values='t_p').round(4).to_string())

summary = dict(
    reference_check_max_abs_idr=float(CHK.absdiff.max()),
    dgcrn_check_max_abs=float(max(dg_check)),
    capacity60_reproduced=bool((mm.verdict == mm.verdict_pub).all()),
    capacity60_max_p_diff=float((mm.p - mm.p_pub).abs().max()),
    capacity70_verdicts=C70.verdict.value_counts().to_dict(),
    capacity70_changed_original=int(len(changed)),
    dgcrn_mae={a: {h: float(np.mean([MAE[(a, s, h)] for s in range(10)])) for h in HZ} for a in ('DGCRN', 'DGCRN_matched')},
    dgcrn_mae_sd={a: {h: float(np.std([MAE[(a, s, h)] for s in range(10)], ddof=1)) for h in HZ} for a in ('DGCRN', 'DGCRN_matched')},
)
json.dump(summary, open(os.path.join(HERE, 'dgcrn_analysis.json'), 'w'), indent=1)
print('\nwrote dgcrn_analysis.json')
