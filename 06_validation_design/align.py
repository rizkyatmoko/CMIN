"""
Event-alignment test for CMIN's daily domain attribution.  PRE-SPECIFIED, and used
unchanged by the pilot and by the ten-seed run.

Peak rule       a domain peaks on day t when its attribution exceeds the 90th percentile
                of its OWN trailing 60 days (t-60 .. t-1).  Causal, scale-free, so a
                collapsed mixture and a well-spread one are treated identically.
Match rule      a peak in domain d matches an event of domain d dated within +/- 3 days.
Null            each event TYPE's dates are circularly shifted by one random offset, which
                preserves that type's clustering (Ramadan stays a 30-day block) and
                destroys only its alignment with the attribution.  1000 draws.
Reported        precision, recall, F1 per domain, the null distribution percentile, and a
                peak-domain x event-domain confusion matrix.
"""
import numpy as np, pandas as pd

W_TRAIL, Q_TRAIL, MATCH_DAYS, N_NULL, SEED = 60, 0.90, 3, 1000, 20260909


def peaks(attr, w=W_TRAIL, q=Q_TRAIL):
    """attr: DataFrame indexed by date, one column per domain -> boolean DataFrame."""
    thr = attr.shift(1).rolling(w, min_periods=w // 2).quantile(q)
    return (attr > thr) & thr.notna()


def _match(peak_days, event_days, k=MATCH_DAYS):
    """Number of peaks within k days of an event, and events within k days of a peak."""
    if len(peak_days) == 0 or len(event_days) == 0:
        return 0, 0
    p = np.sort(np.asarray(peak_days)); e = np.sort(np.asarray(event_days))
    i = np.searchsorted(e, p)
    d = np.full(len(p), np.inf)
    for off in (-1, 0):
        j = np.clip(i + off, 0, len(e) - 1)
        d = np.minimum(d, np.abs(p - e[j]))
    hit_p = int((d <= k).sum())
    i2 = np.searchsorted(p, e)
    d2 = np.full(len(e), np.inf)
    for off in (-1, 0):
        j = np.clip(i2 + off, 0, len(p) - 1)
        d2 = np.minimum(d2, np.abs(e - p[j]))
    hit_e = int((d2 <= k).sum())
    return hit_p, hit_e


def prf(hit_p, n_peaks, hit_e, n_events):
    pr = hit_p / n_peaks if n_peaks else np.nan
    rc = hit_e / n_events if n_events else np.nan
    f1 = 2 * pr * rc / (pr + rc) if (pr and rc and pr + rc > 0) else 0.0
    return pr, rc, f1


def run(attr, events, k=MATCH_DAYS, n_null=N_NULL, seed=SEED, label=''):
    """attr: DataFrame date x domain.  events: date, domain, event_type columns."""
    attr = attr.sort_index()
    d0, d1 = attr.index.min(), attr.index.max()
    ev = events[(events.date >= d0) & (events.date <= d1)].copy()
    idx = attr.index
    day = {d: i for i, d in enumerate(idx)}
    n = len(idx)
    P = peaks(attr)
    rng = np.random.default_rng(seed)

    ev['i'] = ev.date.map(day)
    ev = ev.dropna(subset=['i']); ev['i'] = ev.i.astype(int)

    out, conf = [], []
    for dom in attr.columns:
        pk = np.flatnonzero(P[dom].to_numpy())
        edom = ev[ev.domain == dom]
        ei = edom.i.to_numpy()
        hp, he = _match(pk, ei, k)
        pr, rc, f1 = prf(hp, len(pk), he, len(ei))

        null = np.empty(n_null)
        for b in range(n_null):
            shifted = []
            for _, grp in edom.groupby('event_type'):
                shifted.append((grp.i.to_numpy() + rng.integers(0, n)) % n)
            si = np.concatenate(shifted) if shifted else np.array([], int)
            h1, h2 = _match(pk, si, k)
            null[b] = prf(h1, len(pk), h2, len(si))[2]
        pct = float((null < f1).mean()) if len(ei) else np.nan
        out.append(dict(label=label, domain=dom, n_peaks=len(pk), n_events=len(ei),
                        precision=pr, recall=rc, f1=f1,
                        null_f1_mean=float(null.mean()), null_f1_p95=float(np.quantile(null, .95)),
                        pct_above_null=pct, p_value=float(1 - pct) if len(ei) else np.nan))
        for dom_e in attr.columns:
            ee = ev[ev.domain == dom_e]
            h, _ = _match(pk, ee.i.to_numpy(), k)
            conf.append(dict(label=label, peak_domain=dom, event_domain=dom_e,
                             n_peaks=len(pk), matched=h,
                             share=h / len(pk) if len(pk) else np.nan))
    return pd.DataFrame(out), pd.DataFrame(conf)

def specificity(attr, events, k=MATCH_DAYS, n_null=N_NULL, seed=SEED, label=''):
    """The decisive statistic.

    For events of domain e, ask how often EACH domain's peaks sit near them.  If the
    attribution is domain-specific, domain e's own peaks should match more often than the
    other domains' peaks do.  If the four weights are one scalar wearing four labels --
    which is what a saturated softmax produces -- every domain matches equally and the
    index is zero BY CONSTRUCTION, however significant the raw alignment looks.

        specificity(e) = recall_e(e-events) - mean_{d != e} recall_d(e-events)

    The same circular shift that nulls the raw test nulls this one.
    """
    attr = attr.sort_index()
    idx = attr.index; day = {d: i for i, d in enumerate(idx)}; n = len(idx)
    ev = events[(events.date >= idx.min()) & (events.date <= idx.max())].copy()
    ev['i'] = ev.date.map(day); ev = ev.dropna(subset=['i']); ev['i'] = ev.i.astype(int)
    P = peaks(attr)
    pk = {d: np.flatnonzero(P[d].to_numpy()) for d in attr.columns}
    rng = np.random.default_rng(seed)

    def rec(peak_days, event_i):
        return _match(peak_days, event_i, k)[1] / len(event_i) if len(event_i) else np.nan

    rows = []
    for e_dom in attr.columns:
        ee = ev[ev.domain == e_dom]
        if len(ee) == 0:
            continue
        ei = ee.i.to_numpy()
        own = rec(pk[e_dom], ei)
        others = [rec(pk[d], ei) for d in attr.columns if d != e_dom]
        obs = own - np.nanmean(others)
        null = np.empty(n_null)
        for b in range(n_null):
            sh = []
            for _, grp in ee.groupby('event_type'):
                sh.append((grp.i.to_numpy() + rng.integers(0, n)) % n)
            si = np.concatenate(sh)
            o = rec(pk[e_dom], si)
            ot = [rec(pk[d], si) for d in attr.columns if d != e_dom]
            null[b] = o - np.nanmean(ot)
        rows.append(dict(label=label, event_domain=e_dom, n_events=len(ee),
                         recall_own=own, recall_others_mean=float(np.nanmean(others)),
                         specificity=obs, null_mean=float(null.mean()),
                         null_p95=float(np.quantile(null, .95)),
                         p_value=float((null >= obs).mean())))
    return pd.DataFrame(rows)


def dof(attr):
    """Effective degrees of freedom in the attribution: share of variance in PC1 of the
    four standardised weights.  Near 1.0 means one scalar wearing four labels."""
    X = (attr - attr.mean()) / attr.std()
    e = np.linalg.eigvalsh(np.cov(X.dropna().to_numpy(), rowvar=False))[::-1]
    return float(e[0] / e.sum())
