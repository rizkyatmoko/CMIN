"""
Build the two model-free inputs the validation experiments need.

  1. factor_series.csv  -- the province-wide common factor (PC1) of city log returns at
     h in {1,3,7,14,30}. Loadings and standardisation are fitted on TRAIN DAYS ONLY and
     applied unchanged to validation and test days, so the test-period factor is a
     projection with fixed loadings, not a refit.

  2. event_log.csv      -- one row per (date, event_type), built without any model output.

Everything here is arithmetic on the price/weather files. No seeds, no training.
"""
import numpy as np, pandas as pd, json, sys

TRAIN = r'C:\Users\HP\Documents\Claude\dataset_train.xlsx'
TEST  = r'C:\Users\HP\Documents\Claude\dataset_test.xlsx'
NASA  = r'C:\Users\HP\Documents\Claude\CMIM\CMIN 2\nasa_power_jawa_timur_merged.csv'
POLICY= r'C:\Users\HP\Documents\Claude\CMIM\CMIN 2\policy_event_features.csv'
HZ    = [1, 3, 7, 14, 30]

tr = pd.read_excel(TRAIN); te = pd.read_excel(TEST)
for d in (tr, te): d['tanggal'] = pd.to_datetime(d['tanggal'])
TEST_START = te.tanggal.min()
al = pd.concat([tr, te], ignore_index=True)

# ---------------------------------------------------------------- 1. the factor
city = (al.groupby(['tanggal', 'kab_kota'])['harga']
          .apply(lambda s: np.log(s.dropna()).mean())
          .unstack('kab_kota').sort_index())
dates = city.index
print(f'city panel: {city.shape[0]} days x {city.shape[1]} cities, '
      f'{city.isna().mean().mean()*100:.2f}% missing')
city = city.ffill(limit=7)

rows, meta = [], {}
for h in HZ:
    R = (city.shift(-h) - city)                     # FORWARD h-day log return from day t
    ok = R.notna().all(axis=1)
    is_tr = (R.index < TEST_START) & ok
    mu, sd = R[is_tr].mean(), R[is_tr].std(ddof=0).replace(0, np.nan)
    Z = ((R - mu) / sd)                             # train-fitted standardisation
    Ztr = Z[is_tr].to_numpy()
    C = np.cov(Ztr, rowvar=False)
    w, V = np.linalg.eigh(C)
    load = V[:, -1]
    if load.sum() < 0: load = -load                 # sign: a rise in the factor = prices up
    share_tr = w[-1] / w.sum()
    F = pd.Series(Z.to_numpy() @ load, index=Z.index).where(ok)
    F = F / F[is_tr].std(ddof=0)                    # scale on train only
    # out-of-sample share of variance carried by the SAME loading vector
    Zte = Z[(Z.index >= TEST_START) & ok].to_numpy()
    share_te = float(np.var(Zte @ load, ddof=0) / np.trace(np.cov(Zte, rowvar=False)))
    meta[h] = dict(pc1_share_train=float(share_tr), pc1_share_test_fixed_loadings=share_te,
                   n_train_days=int(is_tr.sum()), n_test_days=int(len(Zte)),
                   loading_min=float(load.min()), loading_max=float(load.max()))
    rows.append(F.rename(f'F_h{h}'))
    print(f'  h={h:2d}  PC1 share train={share_tr:.3f}  test(fixed loadings)={share_te:.3f}  '
          f'loadings all same sign={bool((load > 0).all())}')

fac = pd.concat(rows, axis=1)
fac.index.name = 'date'
fac['split'] = np.where(fac.index < TEST_START, 'train_or_val', 'test')
fac.to_csv('factor_series.csv')
json.dump(meta, open('factor_meta.json', 'w'), indent=2)
print(f'\nfactor_series.csv: {len(fac)} rows, test rows {(fac.split=="test").sum()}')

# ---------------------------------------------------------------- 2. the event log
ev = []

# weather -- NASA POWER precipitation, province mean, 95th pct of the TRAIN distribution
na = pd.read_csv(NASA, usecols=['wilayah', 'date', 'PRECTOTCORR'], parse_dates=['date'])
prec = na.groupby('date')['PRECTOTCORR'].mean().reindex(dates)
thr = prec[prec.index < TEST_START].quantile(0.95)
wet = prec[prec >= thr]
for d in wet.index: ev.append((d, 'weather', 'extreme_rain', 'faithfulness', 'NASA POWER'))
print(f'\nweather: threshold {thr:.2f} mm/day (train p95) -> {len(wet)} days '
      f'({(wet.index>=TEST_START).sum()} in test)')

# economy -- Ramadan blocks, Idul Fitri (day after each block), fuel-price changes
g = al[['tanggal', 'Ramadan', 'Harga BBM (Pertalite)', 'Harga BBM (Pertamax)',
        'Harga BBM (Solar Subsidi)']].drop_duplicates('tanggal').set_index('tanggal').sort_index()
r = g.Ramadan.astype(int); blocks = (r.diff() != 0).cumsum()
for _, b in g[r == 1].groupby(blocks[r == 1]):
    for d in b.index: ev.append((d, 'economy', 'ramadan', 'faithfulness', 'calendar'))
    idul = b.index.max() + pd.Timedelta(days=1)
    if idul in dates: ev.append((idul, 'economy', 'idul_fitri', 'faithfulness', 'calendar'))
fuel = g[['Harga BBM (Pertalite)', 'Harga BBM (Pertamax)', 'Harga BBM (Solar Subsidi)']]
for d in g.index[fuel.diff().abs().sum(axis=1) > 0]:
    ev.append((d, 'economy', 'fuel_price_change', 'faithfulness', 'dataset fuel columns'))

# supply -- regulatory documents. NEVER an input: the protocol sets DROP_POLICY = True.
po = pd.read_csv(POLICY, parse_dates=['event_date'])
SUP = ['market_operation', 'import_export', 'supply_reserve', 'procurement_distribution',
       'production', 'price_management', 'subsidy_assistance']
po = po[(po.event_date >= dates.min()) & (po.event_date <= dates.max())]
tagged = po[po[SUP].sum(axis=1) > 0]
for _, x in tagged.iterrows():
    kinds = [c for c in SUP if x[c] == 1]
    ev.append((x.event_date, 'supply', 'policy_' + '+'.join(kinds), 'external',
               f"JDIH {x.document_type} {x.legal_number}/{x.legal_year}"))

E = pd.DataFrame(ev, columns=['date', 'domain', 'event_type', 'independence', 'source'])
E = E[E.date.isin(dates)].drop_duplicates(['date', 'event_type']).sort_values('date')
E['split'] = np.where(E.date < TEST_START, 'train_or_val', 'test')
E.to_csv('event_log.csv', index=False)

print('\n--- event log power ---')
p = (E.groupby(['domain', 'independence', 'split']).size().unstack('split', fill_value=0))
p['total'] = p.sum(axis=1); print(p.to_string())
print('\nby event_type:')
q = E.groupby(['event_type', 'split']).size().unstack('split', fill_value=0)
q['total'] = q.sum(axis=1); print(q.sort_values('total', ascending=False).to_string())
print(f'\nevent_log.csv: {len(E)} rows over {E.date.nunique()} distinct days')
