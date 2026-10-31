"""
Training (CVaR + COBYLA + INTERP depth chain) and evaluation metrics.
Training uses exact (statevector) probabilities; the reported solution comes from finite shots.
"""
import time

import numpy as np
from scipy.optimize import minimize

from .common import cached, fingerprint
from .methods import interp
from .sim import cvar


def metrics(method, theta, p, exact, shots, rng, tol=1e-9):
    xint, prob, _ = method.dist(theta, p)
    prob = prob / prob.sum()
    f = exact.f_all[xint]
    feas = exact.feas_all[xint]
    is_opt = feas & (f <= exact.f_opt + tol * max(1.0, abs(exact.f_opt)))
    p_feas = float(prob[feas].sum())
    p_opt = float(prob[is_opt].sum())
    span = exact.f_worst - exact.f_opt
    e_feas = float(np.dot(prob[feas], f[feas]) / p_feas) if p_feas > 0 else np.nan
    draws = rng.choice(len(prob), size=shots, p=prob)
    fd = f[draws]
    fe = feas[draws]
    best_f = float(fd[fe].min()) if fe.any() else np.nan
    best_x = int(xint[draws[fe][np.argmin(fd[fe])]]) if fe.any() else None
    # most probable feasible item bitstring (aggregated over slack configurations)
    agg = {}
    for x, pr, ok in zip(xint, prob, feas):
        if ok:
            agg[int(x)] = agg.get(int(x), 0.0) + pr
    mp_x = max(agg, key=agg.get) if agg else None
    ar = lambda v: 1.0 if span <= 0 else (exact.f_worst - v) / span
    return {
        "p_feasible": p_feas, "p_opt": p_opt,
        "shots_feasible_frac": float(fe.mean()), "shots_hit_opt": bool(fe.any() and best_f <= exact.f_opt + tol * max(1.0, abs(exact.f_opt))),
        "best_sampled_f": best_f, "best_sampled_xint": best_x,
        "ar_best_sampled": ar(best_f) if fe.any() else np.nan,
        "ar_expected_given_feasible": ar(e_feas) if p_feas > 0 else np.nan,
        "most_probable_feasible_xint": mp_x,
    }


def method_fingerprint(method):
    """Identity of a method instance: problem data, training energies, structure and initialisation settings."""
    inst = method.inst
    return fingerprint(method.name, inst.lin, inst.W, np.asarray(inst.k), np.asarray(method.e_train, float),
                       method.n_params(1), getattr(method, "params", {}), getattr(method, "init_gamma", None),
                       getattr(method, "init_beta", None), getattr(method, "init_halfwidth", None),
                       method.shot_seed_stride, float(method.span), float(method.exact.f_opt))


def run_chain(method, exact, seed, p_max, maxiter, alpha, shots, p_list=None):
    """One seed: p = 1..p_max with INTERP warm starts. Returns a list of per-p records (checkpointed)."""
    p_list = p_list or list(range(1, p_max + 1))
    key = fingerprint(method_fingerprint(method), seed, p_max, maxiter, alpha, shots, p_list)
    return cached("chain", key, lambda: _run_chain(method, exact, seed, p_max, maxiter, alpha, shots, p_list))


def _run_chain(method, exact, seed, p_max, maxiter, alpha, shots, p_list):
    rng = np.random.default_rng(seed)
    p_list = p_list or list(range(1, p_max + 1))
    theta = method.init(rng, 1)
    order = np.argsort(method.dist(theta, 1)[2], kind="stable")   # training energies are fixed
    out = []
    prev = None
    for p in range(1, max(p_list) + 1):
        if p > 1:
            theta = interp(prev, p - 1)
        n = method.n_params(p)
        budget = max(maxiter, 2 * n + 2)
        hist = []

        def obj(t):
            _, pr, e = method.dist(t, p)
            v = cvar(e, pr / pr.sum(), alpha, order)
            hist.append(v)
            return v

        t0 = time.time()
        res = minimize(obj, theta, method="COBYLA", options={"maxiter": budget})
        dt = time.time() - t0
        prev = res.x
        if p in p_list:
            m = metrics(method, res.x, p, exact, shots, np.random.default_rng(method.shot_seed_stride * seed + p))
            out.append({"method": method.name, "seed": seed, "p": p, "n_params": n, "budget": budget,
                        "nfev": int(res.nfev), "cvar_final": float(res.fun), "train_seconds": dt,
                        "theta": res.x.tolist(), "history": hist, **m})
    return out
