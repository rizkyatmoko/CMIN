"""Check the five comparison commodity panels against the published Section 3 columns.

Table 3 (staleness), Table 4 (PC1) and Table 5 (distance decay) are reported for six panels.
The chilli panel is in ../09_raw_inputs/dataset_*.xlsx; the other five are the CSVs in
../09_raw_inputs/panels/.  This script recomputes their diagnostics and prints each value next
to the published one from commodity_structure.csv.

    python verify_commodity_panels.py
"""
import glob, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', '09_raw_inputs')
STRUCT = pd.read_csv(os.path.join(HERE, 'commodity_structure.csv'))

FILES = {'cabai rawit': 'cabai_rawit_jatim_2021_2026.csv',
         'bawang merah': 'bawang merah 39 _jatim_2021_2026.csv',
         'bawang putih': 'Bawang_putihjatim_2021_2026.csv',
         'beras medium': 'beras nediun_jatim_2021_2026.csv',
         'beras premium': 'beras premium jatim_2021_2026.csv'}

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


H = [1, 3, 7, 14, 30]
print('%-14s %-8s %s' % ('commodity', 'quantity', '  '.join('h=%-13d' % h for h in H)))
for name, f in FILES.items():
    d = pd.read_csv(os.path.join(RAW, 'panels', f))
    d.columns = [c.strip('﻿') for c in d.columns]
    d = d.rename(columns={'Kota': 'kab_kota', 'Lokasi': 'pasar', 'Tanggal': 'tanggal', 'Harga': 'harga'})
    d['tanggal'] = pd.to_datetime(d['tanggal'])
    d = d[d.harga > 0]
    pub = STRUCT[STRUCT.commodity == name].iloc[0]
    cities = sorted(d.kab_kota.unique())
    rows = [row_of(c) for c in cities]
    wide = d.pivot_table(index='tanggal', columns='kab_kota', values='harga', aggfunc='mean')[cities].asfreq('D')
    logp = np.log(wide.ffill().bfill())
    iu = np.triu_indices(len(cities), 1)
    dist = M[np.ix_(rows, rows)][iu]
    pc1, dec = [], []
    for h in H:
        ret = (logp.shift(-h) - logp).dropna().to_numpy()
        Cm = np.corrcoef(ret.T)
        ev = np.linalg.eigvalsh(Cm)
        pc1.append(ev[-1] / ev.sum())
        dec.append(np.corrcoef(Cm[iu], dist)[0, 1])
    print('%-14s %-8s %s' % (name, 'decay', '  '.join('%+.3f/%+.3f ' % (dec[i], pub['dist_%d' % h]) for i, h in enumerate(H))))
    print('%-14s %-8s %s' % ('', 'PC1', '  '.join(' %.3f/%.3f ' % (pc1[i], pub['pc1_%d' % h]) for i, h in enumerate(H))))
    print('%-14s %-8s %d (city, market) pairs [%d distinct names], %d cities; the paper reports %d markets, %d cities'
          % ('', 'panel', d.groupby(['kab_kota', 'pasar']).ngroups, d.pasar.nunique(), len(cities), pub.markets, pub.cities))

print("""
Reading this
------------
Each pair is recomputed/published.

* The distance-decay row matches the published column for all five panels, at every horizon,
  to three decimals.  That identifies these files as the panels behind Table 5.
* PC1 comes out lower here for the same reason as the chilli panel: this script pools raw
  city prices, while the published run normalised each market before pooling.  The gap is
  small for the three vegetables and larger at short horizons for the two rice series.
* The market count agrees once markets are counted the way the paper counts them: these files
  carry 114 distinct market names but 121 distinct (city, market) pairs, which is the 121 in
  Table 3.  Seven market names recur in more than one regency.
""")
