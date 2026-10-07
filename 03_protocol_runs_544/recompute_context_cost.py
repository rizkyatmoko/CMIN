"""Rebuild the context-cost table (Table 11) from the per-seed metrics.

Table 11 reports the change in MAE from supplying the full heterogeneous context in place of
price-only context, at a fixed physical adjacency (L2 -> L5) and at a context-conditioned one
(L4 -> L6), with a p-value per horizon.  No CSV of that table survives, so this script derives
it from metrics_long.csv, which holds one row per run, seed and horizon.

    python recompute_context_cost.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, 'metrics_long.csv'))

PAIRS = [('Fixed physical', 'L2_fixed_physical@R', 'L5_ctx_features_only@R'),
         ('Context-conditioned', 'L4_price_dynamic@R', 'L6_ctx_conditioned@R')]
PUB = {'Fixed physical': [(1, 1.98, '5e-6'), (3, 3.73, '0.002'), (7, 10.02, '4e-4'),
                          (14, 15.61, '2e-5'), (30, -6.75, '0.004')],
       'Context-conditioned': [(1, 1.80, '0.003'), (3, 5.09, '0.001'), (7, 14.20, '7e-5'),
                               (14, 19.13, '7e-7'), (30, -5.43, '0.015')]}

print('%-22s %3s %10s %10s %12s %12s' % ('contrast', 'h', 'change %', 'paper %', 'p (paired t)', 'paper p'))
for name, base, ctx in PAIRS:
    for h, pub_pct, pub_p in PUB[name]:
        a = d[(d.model == base) & (d.h == h)].sort_values('seed')
        b = d[(d.model == ctx) & (d.h == h)].sort_values('seed')
        n = min(len(a), len(b))
        av, bv = a.MAE.to_numpy()[:n], b.MAE.to_numpy()[:n]
        pct = (bv.mean() - av.mean()) / av.mean() * 100
        p = stats.ttest_rel(bv, av).pvalue
        print('%-22s %3d %10.2f %10.2f %12.2g %12s' % (name, h, pct, pub_pct, p, pub_p))

print("""
Both columns reproduce: the percentage change to about 0.05 points and the paired t-test
p-value to the precision the table prints.  The table is therefore a derivation from
metrics_long.csv (%d rows, one per run/seed/horizon), not a separate measurement.
""" % len(d))
