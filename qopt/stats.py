"""Paired statistics: medians, IQR, bootstrap CIs, Wilcoxon with Holm correction, TOST equivalence."""
import numpy as np
from scipy import stats


def summary(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def bootstrap_ci(x, stat=np.median, n_boot=10000, conf=0.95, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    a = (1 - conf) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    if len(d) == 0 or np.all(d == 0):
        return {"n": int(len(d)), "statistic": np.nan, "p_value": 1.0, "wins_a": 0, "wins_b": 0, "ties": int(len(d))}
    r = stats.wilcoxon(a[m], b[m], zero_method="zsplit")
    return {"n": int(len(d)), "statistic": float(r.statistic), "p_value": float(r.pvalue),
            "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()), "ties": int((d == 0).sum())}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tost_paired(diff, margin, alpha=0.05):
    """Two one-sided t-tests on paired differences: equivalence if both p < alpha (|mean diff| < margin)."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "equivalent": False, "p_lower": np.nan, "p_upper": np.nan}
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        eq = abs(m) < margin
        return {"n": n, "mean_diff": float(m), "p_lower": 0.0 if eq else 1.0, "p_upper": 0.0 if eq else 1.0, "equivalent": bool(eq)}
    t_l = (m + margin) / se
    t_u = (m - margin) / se
    p_l = 1 - stats.t.cdf(t_l, n - 1)
    p_u = stats.t.cdf(t_u, n - 1)
    return {"n": n, "mean_diff": float(m), "margin": margin, "p_lower": float(p_l), "p_upper": float(p_u),
            "equivalent": bool(max(p_l, p_u) < alpha)}
