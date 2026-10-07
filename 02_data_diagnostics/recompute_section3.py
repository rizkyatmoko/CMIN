"""Recompute the Section 3 diagnostics for cabai merah straight from the raw panel.

Run from this folder:   python recompute_section3.py

It rebuilds, with no model and no trained checkpoint:
  * the panel size reported in Table 3
  * Table 4, the share of city log-return variance in the first principal component
  * Table 5, the correlation between city-pair return correlation and road distance
  * Table 6, the same three quantities under weekly sampling and on change days only
and prints each recomputed value next to the one printed in the manuscript.

Inputs are read from ../09_raw_inputs/.
"""
import os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '09_raw_inputs')

PUB_PC1 = {1: 0.118, 3: 0.248, 7: 0.434, 14: 0.619, 30: 0.760}
PUB_RHO = {1: 0.084, 3: 0.216, 7: 0.408, 14: 0.596, 30: 0.742}
PUB_DEC = {1: 0.012, 3: -0.029, 7: -0.079, 14: -0.094, 30: -0.117}
PUB_FREQ = {'weekly 1': (0.402, 0.378, -0.041), 'weekly 2': (0.603, 0.577, -0.048),
            'weekly 4': (0.745, 0.725, -0.085)}
PUB_CHANGE = (0.156, 0.059)

df = pd.concat([pd.read_excel(os.path.join(RAW, f), usecols=['tanggal', 'kab_kota', 'pasar', 'harga'])
                for f in ('dataset_train.xlsx', 'dataset_test.xlsx')])
df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df[df.harga > 0]

D = pd.read_excel(os.path.join(RAW, 'Matrix_Jarak_Jawa_Timur_simetris.xlsx'))
labels = [str(x) for x in D.iloc[:, 0]]
M = D.iloc[:, 1:].to_numpy(float)
WB = {l.lower(): i for i, l in enumerate(labels)}


def row_of(c):
    s, low = str(c).strip(), str(c).strip().lower()
    if low.startswith('kabupaten '):
        for cand in (s[10:].strip() + ' Kab.', s[10:].strip()):
            if cand.lower() in WB:
                return WB[cand.lower()]
    return WB.get(low)


cities = sorted(df.kab_kota.unique())
rows = [row_of(c) for c in cities]
assert None not in rows and len(cities) == 38
iu = np.triu_indices(38, 1)
DIST = M[np.ix_(rows, rows)][iu]
wide = df.pivot_table(index='tanggal', columns='kab_kota', values='harga', aggfunc='mean')[cities].asfreq('D')
LOGP = np.log(wide.ffill().bfill())

print('Table 3, panel size')
print('   markets, counted as distinct (city, market) pairs : %d   paper: 120'
      % df.groupby(['kab_kota', 'pasar']).ngroups)
print('   distinct market names                             : %d' % df.pasar.nunique())
print('   cities                                            : %d   paper: 38' % len(cities))
print('   days                                              : %d   paper: 1,826\n' % len(LOGP))


def diag(logp, step):
    ret = (logp.shift(-step) - logp).dropna().to_numpy()
    C = np.corrcoef(ret.T)
    ev = np.linalg.eigvalsh(C)
    return ev[-1] / ev.sum(), float(np.mean(C[iu])), float(np.corrcoef(C[iu], DIST)[0, 1])


print('Tables 4, 5 and 6           recomputed / published')
print('%-26s %8s %8s | %8s %8s | %8s %8s' % ('', 'PC1', 'paper', 'mean r', 'paper', 'decay', 'paper'))
for h in PUB_PC1:
    p, r, d = diag(LOGP, h)
    print('daily, h=%-17d %8.3f %8.3f | %8.3f %8.3f | %8.3f %8.3f'
          % (h, p, PUB_PC1[h], r, PUB_RHO[h], d, PUB_DEC[h]))

weekly = LOGP.resample('W').last()
for k, step in (('weekly 1', 1), ('weekly 2', 2), ('weekly 4', 4)):
    p, r, d = diag(weekly, step)
    pp, pr, pd_ = PUB_FREQ[k]
    print('%-26s %8.3f %8.3f | %8.3f %8.3f | %8.3f %8.3f' % (k + ' step', p, pp, r, pr, d, pd_))

# change-only: each city's daily returns, keeping only the days on which its price moved
chg = LOGP.diff().replace(0.0, np.nan)
C = chg.corr().to_numpy()
v = C[iu]
ok = ~np.isnan(v)
rho, dec = float(np.nanmean(v)), float(np.corrcoef(v[ok], DIST[ok])[0, 1])
print('%-26s %8s %8s | %8.3f %8.3f | %8.3f %8.3f'
      % ('change-only', '-', '-', rho, PUB_CHANGE[0], dec, PUB_CHANGE[1]))

print("""
Reading this table
------------------
* Table 3 reproduces once markets are counted the way the paper counts them: distinct
  (city, market) pairs, 120, rather than distinct market names, 113. Seven market names
  recur in more than one regency.
* Table 5, the distance-decay column, reproduces exactly at every horizon, daily and weekly.
* The mean-correlation column reproduces exactly, daily and weekly.
* The change-only row reproduces exactly: 0.156 and +0.059. Each city's daily returns are
  kept only on the days its own price moved, which is what "change-only" means here.
* Table 4, the PC1 column, comes out 0.002 to 0.008 lower. The code that produced the
  published column is not among the saved files, so the exact pooling it used cannot be
  confirmed; every alternative tried here (per-market normalisation, mean of market logs,
  dropping the first lookback window) moves further away, so this construction is the
  closest available. The ordering and the size of every entry are unaffected.
""")
