import pandas as pd, numpy as np, align
A = pd.read_csv(r'C:\Users\HP\Documents\Claude\CMIM\CMIN 1\global_context_timeseries.csv',
                parse_dates=['target_date']).set_index('target_date')
A.index.name = 'date'
E = pd.read_csv('event_log.csv', parse_dates=['date'])
print(f'pilot attribution: {A.shape[0]} days {A.index.min().date()}..{A.index.max().date()}')
print('domain sd (collapsed global mixture):', A.std().round(5).to_dict())
res, conf = align.run(A, E, label='pilot_single_seed_global')
pd.set_option('display.width', 200)
print('\n--- alignment vs circular-shift null ---')
print(res[['domain','n_peaks','n_events','precision','recall','f1','null_f1_mean','pct_above_null','p_value']]
      .round(3).to_string(index=False))
print('\n--- confusion: share of each domain\'s peaks near each event domain ---')
print(conf.pivot(index='peak_domain', columns='event_domain', values='share').round(3).to_string())
res.to_csv('pilot_alignment.csv', index=False); conf.to_csv('pilot_confusion.csv', index=False)
