"""Weight magnitudes of the four conditioning points in the ORIGINAL ten-seed checkpoints
(cmin_local/protocol, the 50-epoch runs behind the published tables).

If the query/key projections that score edges have decayed to ~0, the context adjacency is
softmax(0) = uniform for every input, and Eqs. 1-3 cannot act.  Reads weights only; no data."""
import glob, json, os, re
import numpy as np
import torch

P = r'C:\Users\HP\Documents\Claude\cmin_local\protocol'
KEYS = {'Wq (edge query)': 'cmil.Wq.weight', 'Wk (edge key)': 'cmil.Wk.weight',
        'FiLM gamma (Eq. 1)': 'cmil.gamma.weight', 'FiLM beta (Eq. 1)': 'cmil.beta.weight',
        'head gate (Eq. 2)': 'cmil.head_gate.weight',
        'temperature out (Eq. 3)': 'cmil.context_edge_bias.2.weight',
        'alpha out (Eq. 4)': 'cmil.alpha_head.2.weight', 'Wv (message value)': 'cmil.Wv.weight'}
rows = []
for f in sorted(glob.glob(os.path.join(P, 'w_*.pt'))):
    m = re.match(r'w_(.+)__s(\d+)\.pt$', os.path.basename(f))
    rung, seed = m.group(1), int(m.group(2))
    sd = torch.load(f, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    if 'cmil.Wq.weight' not in sd:
        continue
    row = dict(rung=rung, seed=seed)
    for lab, k in KEYS.items():
        row[lab] = float(sd[k].float().abs().max()) if k in sd else np.nan
    rows.append(row)

import pandas as pd
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('max |weight| per tensor, median [max] over seeds')
g = D.groupby('rung')
tab = pd.DataFrame({lab: g[lab].median().map('{:.1e}'.format) + ' [' + g[lab].max().map('{:.1e}'.format) + ']'
                    for lab in KEYS})
print(tab.T.to_string())
D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conditioning_weights_original.csv'), index=False)
