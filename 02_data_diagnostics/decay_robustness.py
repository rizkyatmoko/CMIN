"""Distance-decay diagnostic for cabai merah with (a) the manuscript's distance matrix and
(b) OSRM road distances between regency and city seats.  Step (a) must reproduce the published
row of Table 5 before (b) means anything.

Published (commodity_structure.csv): PC1 share 0.118 0.248 0.434 0.619 0.760;
distance correlation +0.012 -0.029 -0.079 -0.094 -0.117 at h = 1, 3, 7, 14, 30.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.concat([pd.read_excel(os.path.join(HERE, f), usecols=['tanggal', 'kab_kota', 'pasar', 'harga'])
                for f in ('dataset_train.xlsx', 'dataset_test.xlsx')])
df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df[(df.harga > 0)]
D = pd.read_excel(os.path.join(HERE, 'Matrix_Jarak_Jawa_Timur_simetris.xlsx'))
labels = [str(x) for x in D.iloc[:, 0]]
M = D.iloc[:, 1:].to_numpy(float)
WB = {l.lower(): i for i, l in enumerate(labels)}


def to_row(c):
    s, low = str(c).strip(), str(c).strip().lower()
    if low.startswith('kabupaten '):
        base = s[10:].strip()
        for cand in (base + ' Kab.', base):
            if cand.lower() in WB:
                return WB[cand.lower()]
    elif low.startswith('kota '):
        if low in WB:
            return WB[low]
    return WB.get(low)


cities = sorted(df.kab_kota.unique())
rows = [to_row(c) for c in cities]
assert None not in rows and len(set(rows)) == len(cities) == 38, (len(cities), rows)
logp = np.log(df.pivot_table(index='tanggal', columns='kab_kota', values='harga', aggfunc='mean')[cities])
logp = logp.asfreq('D').ffill().bfill()

# OSRM matrix in the same row order as the manuscript matrix
chk = pd.read_csv(os.path.join(HERE, 'distance_check.csv'))
O = np.zeros_like(M)
idx = {l: i for i, l in enumerate(labels)}
for r in chk.itertuples():
    O[idx[r.a], idx[r.b]] = O[idx[r.b], idx[r.a]] = r.osrm_km

iu = np.triu_indices(38, 1)
out = {'h': [], 'pc1': [], 'decay_manuscript': [], 'decay_osrm': []}
for h in (1, 3, 7, 14, 30):
    ret = (logp.shift(-h) - logp).dropna().to_numpy()
    C = np.corrcoef(ret.T)
    ev = np.linalg.eigvalsh(np.cov(((ret - ret.mean(0)) / ret.std(0)).T))
    out['h'].append(h)
    out['pc1'].append(float(ev[-1] / ev.sum()))
    Dm = M[np.ix_(rows, rows)]
    Do = O[np.ix_(rows, rows)]
    out['decay_manuscript'].append(float(np.corrcoef(C[iu], Dm[iu])[0, 1]))
    out['decay_osrm'].append(float(np.corrcoef(C[iu], Do[iu])[0, 1]))
R = pd.DataFrame(out).round(3)
print(R.to_string(index=False))
json.dump(out, open(os.path.join(HERE, 'decay_robustness.json'), 'w'), indent=1)
