"""Checks on the CMIN_final run (13 Sep 2026) before any number goes into the report or paper.

Reads final/ (the unzipped cmin_final_results) and writes final_analysis.json.
  1. Is the learned context adjacency A_ctx uniform in the trained models?
  2. Is every day's A_t an affine function of one fixed matrix, scaled by alpha_t?
  3. Cross-seed reproducibility of A_ctx where it is not degenerate.
  4. Regime effects with Benjamini-Hochberg over each head's 24 effects.
  5. Per-seed accuracy of the variants against L6@R, and the seed-0 session question.
  6. Summaries of alignment, specificity, regime response, factor probe.
"""
import glob, io, itertools, json, os, re
from math import comb
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(BASE, 'final')
OUT = {}
U = 1 / 38


def load(nm, s):
    return np.load(io.BytesIO(open(os.path.join(F, 'extract', '%s__s%d.npz.zip' % (nm, s)), 'rb').read()),
                   allow_pickle=True)


def seeds(nm):
    out = []
    for f in glob.glob(os.path.join(F, 'extract', glob.escape(nm) + '__s*.npz.zip')):
        m = re.match(r'.*__s(\d+)\.npz\.zip$', f)
        out.append(int(m.group(1)))
    return sorted(out)


off = ~np.eye(38, dtype=bool)

# ---------------------------------------------------------------- 1. A_ctx uniformity
uni = []
for nm in ('L6_ctx_conditioned@S', 'L6_ctx_conditioned@R', 'L7_ctx_shuffled@S', 'UNTRAINED_L6@S',
           'L3_learned_static@R', 'L6nofilm@R', 'L6noheads@R', 'L6notemp@R', 'L6noalpha@R'):
    for s in seeds(nm):
        e = load(nm, s)
        C = e['Actx_mean'].astype(np.float64)
        A = e['A_mean'].astype(np.float64)
        uni.append(dict(rung=nm, seed=s, ctx_rel_sd_pct=100 * C.std() / C.mean(),
                        ctx_max_dev_pct=100 * np.abs(C - U).max() / U, ctx_max_edge=C.max(),
                        A_rel_sd_pct=100 * A.std() / A.mean(), A_max_edge=A.max()))
UNI = pd.DataFrame(uni)
summ = UNI.groupby('rung').agg(n=('seed', 'size'),
                               ctx_rel_sd_median=('ctx_rel_sd_pct', 'median'),
                               ctx_rel_sd_max=('ctx_rel_sd_pct', 'max'),
                               ctx_max_dev_max=('ctx_max_dev_pct', 'max'),
                               exactly_uniform=('ctx_max_dev_pct', lambda x: int((x < 0.01).sum())),
                               A_rel_sd_median=('A_rel_sd_pct', 'median'))
print('=== 1. learned adjacency A_ctx: spread relative to uniform 1/38 (percent) ===')
print(summ.round(4).to_string())
OUT['actx_uniformity'] = summ.reset_index().to_dict('records')
OUT['actx_by_seed'] = UNI.to_dict('records')

# ---------------------------------------------------------------- 2. daily identity A_t = U + (alpha_t/abar)(A_mean - U)
ident = []
for nm in ('L6_ctx_conditioned@S', 'L6_ctx_conditioned@R', 'L7_ctx_shuffled@S'):
    for s in seeds(nm):
        e = load(nm, s)
        if 'A_test' not in e.files:
            continue
        At = e['A_test'].astype(np.float64).reshape(len(e['A_test']), -1)
        D = e['A_mean'].astype(np.float64).ravel() - U
        alpha_te = e['alpha'][-len(At):].astype(np.float64)
        X = np.column_stack([np.ones_like(D), D])
        coef, *_ = np.linalg.lstsq(X, (At - U).T, rcond=None)
        fit = (X @ coef).T
        ss_res = ((At - U - fit) ** 2).sum(1)
        ss_tot = ((At - At.mean(1, keepdims=True)) ** 2).sum(1)
        r2 = 1 - ss_res / ss_tot
        b = coef[1]
        # share of the day-to-day variance of A_t that the scalar alpha_t accounts for
        pred = U + np.outer(alpha_te / alpha_te.mean(), D)
        var_tot = ((At - At.mean(0)) ** 2).sum()
        var_res = ((At - pred) ** 2).sum()
        ident.append(dict(rung=nm, seed=s, r2_min=float(r2.min()), r2_median=float(np.median(r2)),
                          corr_b_alpha=float(np.corrcoef(b, alpha_te)[0, 1]) if b.std() > 0 else np.nan,
                          max_abs_b_minus_ratio=float(np.abs(b - alpha_te / alpha_te.mean()).max()),
                          alpha_explains_daily_var=float(1 - var_res / var_tot) if var_tot > 0 else np.nan,
                          alpha_sd=float(alpha_te.std())))
IDN = pd.DataFrame(ident)
print('\n=== 2. is every day of A_t the anchor scaled by alpha_t? ===')
print(IDN.groupby('rung').agg(n=('seed', 'size'), r2_min=('r2_min', 'min'), r2_median=('r2_median', 'median'),
                              corr_b_alpha_min=('corr_b_alpha', 'min'),
                              alpha_explains_min=('alpha_explains_daily_var', 'min'),
                              alpha_explains_median=('alpha_explains_daily_var', 'median')).round(5).to_string())
OUT['daily_identity'] = IDN.to_dict('records')

# ---------------------------------------------------------------- 3. A_ctx reproducibility where it has structure
rep = []
for nm in ('L6_ctx_conditioned@S', 'L6_ctx_conditioned@R', 'L7_ctx_shuffled@S', 'L3_learned_static@R', 'UNTRAINED_L6@S'):
    M = {s: load(nm, s)['Actx_mean'].astype(np.float64) for s in seeds(nm)}
    keep = [s for s, C in M.items() if C[off].std() / U > 1e-4]
    rs = [np.corrcoef(M[a][off], M[b][off])[0, 1] for a, b in itertools.combinations(keep, 2)]
    rep.append(dict(rung=nm, seeds_with_structure=len(keep), pairs=len(rs),
                    r_mean=float(np.mean(rs)) if rs else np.nan, r_min=float(np.min(rs)) if rs else np.nan))
REPc = pd.DataFrame(rep)
print('\n=== 3. A_ctx edge correlation across seeds, only seeds whose A_ctx is not flat ===')
print(REPc.round(3).to_string(index=False))
OUT['actx_reproducibility'] = REPc.to_dict('records')

# ---------------------------------------------------------------- 4. regime effects with BH
by = pd.read_csv(os.path.join(F, 'f5_regime_by_seed.csv'))
eff = pd.read_csv(os.path.join(F, 'f5_regime_effects.csv'))
rows = []
for (nm, meas, reg), g in by[by.regime != 'normal'].groupby(['rung', 'measure', 'regime']):
    base = by[(by.rung == nm) & (by.measure == meas) & (by.regime == 'normal')].set_index('seed').value
    d = (g.set_index('seed').value - base).dropna()
    n = len(d)
    k = int(max((d > 0).sum(), (d < 0).sum()))
    sign_p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    t_p = float(stats.ttest_1samp(d, 0).pvalue)
    rows.append(dict(rung=nm, measure=meas, regime=reg, n=n, effect=float(d.mean()),
                     rel_pct=float(100 * d.mean() / base.mean()), agree=k, sign_p=sign_p, t_p=t_p))
RG = pd.DataFrame(rows)


def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    q = np.empty(m); prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[o[i]] * m / (i + 1)); q[o[i]] = prev
    return q


RG['sign_q'] = RG.groupby('rung').sign_p.transform(bh)
RG['t_q'] = RG.groupby('rung').t_p.transform(bh)
RG = RG.merge(eff[['rung', 'measure', 'regime', 'survives', 'effect_ci95', 'normal', 'normal_ci95']],
              on=['rung', 'measure', 'regime'], how='left')
print('\n=== 4. regime effects: notebook survivors, then BH over the 24 effects of each head ===')
cols = ['rung', 'measure', 'regime', 'rel_pct', 'agree', 'sign_p', 'sign_q', 't_p', 't_q', 'survives']
print(RG[RG.survives | (RG.t_q < 0.05)][cols].round(4).to_string(index=False))
OUT['regime'] = RG.to_dict('records')

# ---------------------------------------------------------------- 5. variants per seed
ref = 'L6_ctx_conditioned@R'
var = ['L6nofilm@R', 'L6noheads@R', 'L6notemp@R', 'L6noalpha@R', 'L6nh1@R', 'L6nh2@R', 'L6nh8@R', 'L6d32@R', 'L6d128@R']
H = [1, 3, 7, 14, 30]
mae = {nm: {s: load(nm, s)['mae'] for s in seeds(nm)} for nm in [ref] + var + ['L6_ctx_conditioned@S', 'L1_encoder_only@R',
                                                                          'L3_learned_static@R', 'H_uniform_price@R', 'L7_ctx_shuffled@S']}
pv = []
for nm in var:
    for s in range(10):
        pct = 100 * (mae[nm][s] / mae[ref][s] - 1)
        pv.append(dict(variant=nm, seed=s, **{'h%d' % h: float(v) for h, v in zip(H, pct)}))
PV = pd.DataFrame(pv)
print('\n=== 5. variant vs L6@R, per-seed percent difference, seed 0 against the mean of seeds 1-9 ===')
t = PV.groupby([PV.variant, PV.seed == 0])[['h%d' % h for h in H]].mean().round(2)
print(t.to_string())
OUT['variant_seed0_vs_rest'] = t.reset_index().to_dict('records')
ref_seed = pd.DataFrame({s: mae[ref][s] for s in range(10)}, index=H).T
print('\nL6@R per-seed MAE (seed 0 is the older checkpoint):')
print(ref_seed.round(0).to_string())
OUT['ref_mae_by_seed'] = ref_seed.reset_index().to_dict('records')

# ---------------------------------------------------------------- 6. summaries
fp = pd.read_csv(os.path.join(F, 'f4_factor_probe.csv'))
OUT['factor_probe'] = fp.to_dict('records')
OUT['factor_probe_by_seed'] = pd.read_csv(os.path.join(F, 'f4_factor_probe_by_seed.csv')).to_dict('records')
OUT['factor_contrasts'] = pd.read_csv(os.path.join(F, 'f4_factor_probe_contrasts.csv')).to_dict('records')
OUT['factor_target'] = pd.read_csv(os.path.join(F, 'f4_factor_target.csv')).to_dict('records')
raw = pd.read_csv(os.path.join(F, 'f6_alignment_raw.csv'))
raw = raw[raw.series == 'node']
rs = raw.groupby(['rung', 'domain']).agg(n=('n_events', 'first'), f1=('f1', 'mean'), null_f1=('null_f1', 'mean'),
                                         p_median=('p', 'median'), seeds_p05=('p', lambda x: int((x < 0.05).sum())))
print('\n=== 6a. raw alignment, node series ===')
print(rs.round(3).to_string())
OUT['raw_alignment'] = rs.reset_index().to_dict('records')
sp = pd.read_csv(os.path.join(F, 'f6_specificity.csv'))
sp = sp[sp.series == 'node']
ss = sp.groupby(['rung', 'event_domain']).agg(n=('n_events', 'first'), own=('own', 'mean'), others=('others', 'mean'),
                                              specificity=('specificity', 'mean'), p_median=('p', 'median'),
                                              seeds_p05=('p', lambda x: int((x < 0.05).sum())))
print('\n=== 6b. specificity, node series ===')
print(ss.round(3).to_string())
OUT['specificity'] = ss.reset_index().to_dict('records')
OUT['sensitivity'] = pd.read_csv(os.path.join(F, 'f6_sensitivity.csv')).to_dict('records')
OUT['confusion'] = pd.read_csv(os.path.join(F, 'f6_confusion.csv')).to_dict('records')
OUT['dof'] = pd.read_csv(os.path.join(F, 'f6_dof.csv')).to_dict('records')
rr = pd.read_csv(os.path.join(F, 'f8_regime_response_by_seed.csv'))
OUT['regime_response_by_seed'] = rr.to_dict('records')
a = rr[rr.rung == 'L6_ctx_conditioned@R'].F_over_null
b = rr[rr.rung == 'L7_ctx_shuffled@S'].F_over_null
c = rr[rr.rung == 'L6_ctx_conditioned@S'].F_over_null
OUT['regime_response_mw'] = dict(S_vs_L7=float(stats.mannwhitneyu(c, b, alternative='greater').pvalue),
                                 R_vs_L7=float(stats.mannwhitneyu(a, b, alternative='greater').pvalue))
print('\n=== 6c. regime response F/null medians ===')
print(rr.groupby('rung').agg(F_over_null_median=('F_over_null', 'median'), p_median=('p', 'median'),
                             seeds_p05=('p', lambda x: int((x < 0.05).sum()))).round(3).to_string())
print(OUT['regime_response_mw'])
OUT['market_level'] = pd.read_csv(os.path.join(F, 'f9_market_level.csv')).to_dict('records')
OUT['homogeneity'] = pd.read_csv(os.path.join(F, 'f10_attribution_homogeneity.csv')).to_dict('records')
OUT['contrasts'] = pd.read_csv(os.path.join(F, 'f11_contrasts.csv')).to_dict('records')
OUT['reproduction'] = pd.read_csv(os.path.join(F, 'f3_reproduction.csv')).to_dict('records')
OUT['reproducibility_nb'] = pd.read_csv(os.path.join(F, 'f7_reproducibility.csv')).to_dict('records')
json.dump(OUT, open(os.path.join(BASE, 'final_analysis.json'), 'w'), indent=1, default=float)
print('\nwrote final_analysis.json')
