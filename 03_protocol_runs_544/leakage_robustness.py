"""Does the validation/test window overlap change any published verdict?

Checkpoint selection used a validation score whose last 29 windows reach into the first 29 days
of the test period.  Selection therefore saw a small part of the test calendar.  Training never
did, and no test origin was ever used for selection, but the question a reviewer will ask is
whether the reported comparisons survive if those days are removed from the test set entirely.

WARNING, read leakage_robustness_FINDINGS.md before using this output: the 29 dropped origins
turn out to cover the Ramadan 2025 price spike, where prices are 2x and errors 3.3x the rest of
the test year.  Dropping them therefore measures sensitivity to that regime, NOT the effect of
the overlap, and 42 of the 105 verdicts move.  The leakage question needs retraining to settle.

This script re-scores without retraining.  Every run's stored predictions are re-scored on the
337 published test origins and again on the 308 whose thirty-day windows begin after the last
contaminated day, and the paper's own inference (two-level paired bootstrap, TOST at 2%,
Benjamini-Hochberg within family) is run on both.

Reads the prediction files from the Drive exports; see ../11_raw_runs_index/.
"""
import glob, io, json, os, pickle, re, sys, time, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = r'C:\Users\HP\Documents\Claude\cmin_local'
NB = r'C:\Users\HP\Documents\Claude\cmin_protocol\CMIN_final.ipynb'
EXPORTS = sorted(glob.glob(r'C:\Users\HP\Downloads\cmin_v4_out-2026*.zip'))
EXTRACTED = (r'C:\Users\HP\Downloads\Compressed\cmin_v4_out-20260901T044628Z-1-001'
             r'\cmin_v4_out\protocol')

ARMS = ['H_real@R', 'H_real@S', 'L1_encoder_only@R', 'L2_fixed_physical@R', 'L2_fixed_physical@S',
        'L3_learned_static@R', 'L4_price_dynamic@R', 'L4_price_dynamic@S', 'L5_ctx_features_only@R',
        'L5_ctx_features_only@S', 'L6_ctx_conditioned@R', 'L6_ctx_conditioned@S',
        'H_uniform_price@R', 'LSTM_matched', 'GRU_matched', 'TCN_matched', 'STGCN_matched',
        'AGCRN_matched', 'GWNet_matched']

import torch
torch.backends.mkldnn.enabled = False
os.chdir(LOCAL)
g = {'__name__': '__main__'}
t0 = time.time()
for c in json.load(open(NB, encoding='utf-8'))['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if src.strip().split('\n')[0].startswith('# === F0'):
        break
    exec(compile(src, '<cell>', 'exec'), g)
os.chdir(HERE)
RP, Y, obs, CFG, TEST_ORIGINS, = (g[k] for k in ('RP', 'Y', 'obs', 'CFG', 'TEST_ORIGINS'))
HZ = list(CFG['HORIZONS'])
print('definitions loaded in %.0fs; %d test origins' % (time.time() - t0, len(TEST_ORIGINS)))

# the contaminated calendar days are the first 29 test target days; a test origin is clean when
# its own window starts after them
FIRST_TEST_TARGET = int(TEST_ORIGINS[0]) + 1
CLEAN = np.array([i for i, t in enumerate(TEST_ORIGINS) if t + 1 > FIRST_TEST_TARGET + 28])
print('clean test origins: %d of %d (dropping the first %d)'
      % (len(CLEAN), len(TEST_ORIGINS), len(TEST_ORIGINS) - len(CLEAN)))

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


index = {}
for f in EXPORTS:
    z = zipfile.ZipFile(f)
    for i in z.infolist():
        m = re.match(r'(?:.*/)?protocol/([^/]+)__s(\d+)\.pkl(\.zip)?$', i.filename)
        if m and m.group(1) in ARMS and '_smoke' not in i.filename:
            index.setdefault((m.group(1), int(m.group(2))), ('zip', f, i.filename))
for p in glob.glob(os.path.join(EXTRACTED, '*.pkl')):
    m = re.match(r'([^/\\]+)__s(\d+)\.pkl$', os.path.basename(p))
    if m and m.group(1) in ARMS:
        index.setdefault((m.group(1), int(m.group(2))), ('file', p, None))

WIN = {}
for key in sorted(index):
    kind, a, b = index[key]
    raw = zipfile.ZipFile(a).read(b) if kind == 'zip' else open(a, 'rb').read()
    WIN[key] = win_mae(load_pred(raw))
print('scored %d model-seeds across %d arms' % (len(WIN), len({k[0] for k in WIN})))

MARGIN = 2.0


def paired_boot(a, b, h, cols, B=1000):
    rng = np.random.default_rng(0)
    hi = HZ.index(h)
    sa = sorted(s for (m, s) in WIN if m == a)
    sb = sorted(s for (m, s) in WIN if m == b)
    A = np.stack([WIN[(a, s)][hi][cols] for s in sa])
    Bm = np.stack([WIN[(b, s)][hi][cols] for s in sb])
    T = A.shape[1]
    draws = np.empty(B)
    for i in range(B):
        wi = rng.integers(0, T, T)
        ai = rng.integers(0, len(sa), len(sa))
        bi = rng.integers(0, len(sb), len(sb))
        draws[i] = A[np.ix_(ai, wi)].mean() - Bm[np.ix_(bi, wi)].mean()
    lo, hi_ = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(diff=float(A.mean() - Bm.mean()), lo=float(lo), hi=float(hi_),
                p=float(min(p, 1.0)), ref=float(Bm.mean()))


def verdict(r):
    m = MARGIN / 100 * r['ref']
    inside = (r['lo'] > -m) and (r['hi'] < m)
    excl0 = (r['lo'] > 0) or (r['hi'] < 0)
    return 'equivalent' if inside else ('different' if excl0 else 'inconclusive')


def bh(p, q=0.05):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    passed = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[o[:k]] = True
    return rej


def family(pairs, cols):
    rows = []
    for a, b in pairs:
        for h in HZ:
            r = paired_boot(a, b, h, cols)
            rows.append(dict(a=a, b=b, h=h, pct=100 * r['diff'] / r['ref'], verdict=verdict(r), **r))
    D = pd.DataFrame(rows)
    D['bh_reject'] = bh(D.p.values)
    D.loc[(D.verdict == 'different') & ~D.bh_reject, 'verdict'] = 'inconclusive'
    return D


FAMILIES = {
    'context conditioning (L6 vs L5)': [('L6_ctx_conditioned@R', 'L5_ctx_features_only@R'),
                                        ('L6_ctx_conditioned@S', 'L5_ctx_features_only@S')],
    'context as input (L5 vs L2, L6 vs L4)': [('L5_ctx_features_only@R', 'L2_fixed_physical@R'),
                                              ('L5_ctx_features_only@S', 'L2_fixed_physical@S'),
                                              ('L6_ctx_conditioned@R', 'L4_price_dynamic@R'),
                                              ('L6_ctx_conditioned@S', 'L4_price_dynamic@S')],
    'pooling (learned graph vs uniform, no graph)': [('L3_learned_static@R', 'H_uniform_price@R'),
                                                     ('L6_ctx_conditioned@R', 'H_uniform_price@R'),
                                                     ('L1_encoder_only@R', 'H_uniform_price@R')],
    'capacity (CMIN vs matched baselines)': [(hd, b) for hd in ('H_real@R', 'H_real@S')
                                             for b in ('LSTM_matched', 'GRU_matched', 'TCN_matched',
                                                       'STGCN_matched', 'AGCRN_matched', 'GWNet_matched')],
}

ALL = np.arange(len(TEST_ORIGINS))
out, flips = [], 0
for name, pairs in FAMILIES.items():
    full, clean = family(pairs, ALL), family(pairs, CLEAN)
    m = full.merge(clean, on=['a', 'b', 'h'], suffixes=('_full', '_clean'))
    changed = m[m.verdict_full != m.verdict_clean]
    flips += len(changed)
    print('\n%-46s %d comparisons | verdicts changed: %d | max |pct shift| %.2f points'
          % (name, len(m), len(changed), (m.pct_full - m.pct_clean).abs().max()))
    print('   full  :', full.verdict.value_counts().to_dict())
    print('   clean :', clean.verdict.value_counts().to_dict())
    for r in changed.itertuples():
        print('   CHANGED %s vs %s h=%d: %s -> %s (%.2f%% -> %.2f%%)'
              % (r.a, r.b, r.h, r.verdict_full, r.verdict_clean, r.pct_full, r.pct_clean))
    m.insert(0, 'family', name)
    out.append(m)

R = pd.concat(out)
R.to_csv(os.path.join(HERE, 'leakage_robustness.csv'), index=False)
print('\n%d comparisons in total; %d verdicts change when the overlapping days are dropped.' % (len(R), flips))
print('written: leakage_robustness.csv')
