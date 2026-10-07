"""F11 re-run locally: (a) all ten seeds, to confirm this reimplementation reproduces the notebook's
f11_contrasts.csv; (b) seeds 1-9 only, where the reference L6@R and every variant were trained in
the same Colab session.  Seed 0 of L6@R is an older checkpoint whose h=30 MAE (19,296) is 25% above
the other nine, so any ten-seed contrast at h=30 carries a session difference, not a mechanism.
The inference is the notebook's own: two-level bootstrap, TOST at 2%, BH within family."""
import glob, io, json, os, re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(BASE, 'final')
BOOT_B, MARGIN = 1000, 2.0
HZ = [1, 3, 7, 14, 30]
_W = {}


def win(nm, seeds_):
    key = (nm, tuple(seeds_))
    if key not in _W:
        _W[key] = np.stack([np.load(io.BytesIO(open(os.path.join(F, 'extract', '%s__s%d.npz.zip' % (nm, s)), 'rb').read()))['win_mae']
                            for s in seeds_])
    return _W[key]


def paired_boot(A_, B_, rng):
    T_ = A_.shape[1]
    obs_ = A_.mean() - B_.mean()
    dr = np.empty(BOOT_B)
    for i in range(BOOT_B):
        wi = rng.integers(0, T_, T_)
        dr[i] = A_[np.ix_(rng.integers(0, len(A_), len(A_)), wi)].mean() - \
                B_[np.ix_(rng.integers(0, len(B_), len(B_)), wi)].mean()
    lo, hi_ = np.percentile(dr, [2.5, 97.5])
    return obs_, lo, hi_, float(min(1.0, 2 * min((dr <= 0).mean(), (dr >= 0).mean()))), B_.mean()


def bh(p, q=0.05):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(n, bool); rej[o[:k]] = True
    return rej


REF = 'L6_ctx_conditioned@R'
FAMILIES = {'one_point_removed': ['L6nofilm@R', 'L6noheads@R', 'L6notemp@R', 'L6noalpha@R'],
            'head_count': ['L6nh1@R', 'L6nh2@R', 'L6nh8@R'],
            'width': ['L6d32@R', 'L6d128@R']}


def run(seeds_, label):
    rng = np.random.default_rng(0)
    allc = []
    for fam, names in FAMILIES.items():
        rows = []
        for nm in names:
            for hi, h in enumerate(HZ):
                d_, lo, hi_, p, ref = paired_boot(win(nm, seeds_)[:, hi], win(REF, seeds_)[:, hi], rng)
                m = MARGIN / 100 * ref
                inside, excl = (lo > -m) and (hi_ < m), (lo > 0) or (hi_ < 0)
                rows.append(dict(seeds=label, family=fam, variant=nm, h=h, diff=float(d_), lo=float(lo), hi=float(hi_),
                                 pct=float(100 * d_ / ref), lo_pct=float(100 * lo / ref), hi_pct=float(100 * hi_ / ref), p=p,
                                 verdict='equivalent' if inside else ('different' if excl else 'inconclusive')))
        D = pd.DataFrame(rows)
        D['bh_reject'] = bh(D.p.values)
        D.loc[(D.verdict == 'different') & ~D.bh_reject, 'verdict'] = 'inconclusive'
        allc.append(D)
    return pd.concat(allc)


ten = run(list(range(10)), 'all ten')
nb = pd.read_csv(os.path.join(F, 'f11_contrasts.csv'))
chk = ten.merge(nb, on=['family', 'variant', 'h'], suffixes=('', '_nb'))
print('reimplementation vs notebook: max |pct diff| %.4f, verdicts identical: %s'
      % ((chk.pct - chk.pct_nb).abs().max(), bool((chk.verdict == chk.verdict_nb).all())))
same = run(list(range(1, 10)), 'seeds 1-9')
for D in (ten, same):
    print('\n=== %s ===' % D.seeds.iloc[0])
    print(D.pivot(index='variant', columns='h', values='pct').round(2).to_string())
    print(D.groupby('variant').verdict.value_counts().unstack(fill_value=0).to_string())
out = pd.concat([ten, same])
out.to_csv(os.path.join(BASE, 'contrasts_same_session.csv'), index=False)
print('\nhalf-widths of the interval, percent of reference (seeds 1-9):')
same['half'] = (same.hi_pct - same.lo_pct) / 2
print(same.pivot(index='variant', columns='h', values='half').round(2).to_string())
