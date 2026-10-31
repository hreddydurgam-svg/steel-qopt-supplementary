"""
Steel case study (Salem testbed): scenario sampling, risk matrix, safe-count chance-constraint
calibration, out-of-sample certification, and construction of the Instance used by every solver.

Blocks inside one route are identical, so a selection is fully described by its route-count vector
c = (c_H1, c_R1, c_R2, c_R3, c_R4). Feasibility under the joint chance constraint depends on c only.
"""
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist, genpareto, norm, t as t_dist

from .instance import build_from_raw


# ------------------------------------------------------------------ scenario sampling
def route_corr(p, rho_cross=None):
    cop = p.raw["risk"]["copula"]
    rho_m = cop["rho_same_material_other_route"]
    rho_x = cop["rho_cross_material"] if rho_cross is None else rho_cross
    mats = [p.raw["routes"][r]["material"] for r in p.routes]
    R = len(p.routes)
    C = np.eye(R)
    for i in range(R):
        for j in range(R):
            if i != j:
                C[i, j] = rho_m if mats[i] == mats[j] else rho_x
    return C


def sample_scenarios(p, n, seed, family="gaussian", nu=None, sigma=None, rho_cross=None):
    """Return F (n x R) delivered fraction and M (n x R) cost multiplier, from one copula draw U.
    F = max(0, 1 - G), G ~ GPD(xi, mu, sigma) [capacity]; M ~ GPD(xi_c, mu_c, sigma_c) [cost], from the
    same copula draw U (comonotone within a route)."""
    cs = p.raw["risk"]["capacity_shock"]
    cc = p.raw["risk"]["cost_shock"]
    C = route_corr(p, rho_cross)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(p.routes))) @ np.linalg.cholesky(C).T
    if family == "gaussian":
        U = norm.cdf(Z)
    elif family == "student_t":
        if nu is None:
            raise ValueError("student_t copula needs nu (risk.chance_constraint.validation_cases)")
        W = rng.chisquare(nu, (n, 1)) / nu
        U = t_dist.cdf(Z / np.sqrt(W), nu)
    else:
        raise ValueError(family)
    G = genpareto.ppf(U, c=cs["xi"], loc=cs["mu"], scale=cs["sigma"] if sigma is None else sigma)
    F = np.clip(1.0 - G, 0.0, None)
    M = genpareto.ppf(U, c=cc["xi"], loc=cc["mu"], scale=cc["sigma"])
    return F, M


# ------------------------------------------------------------------ route model
@dataclass
class RouteModel:
    routes: list
    mats: list           # material per route
    size: np.ndarray     # block size per route (t)
    tlc: np.ndarray      # US$/t per route
    emis: np.ndarray     # kg/t per route
    pool: np.ndarray     # blocks available per route
    demand: dict         # material -> t/month
    materials: list      # ordered material names (group order)


def route_model(p, mods=None):
    mods = mods or {}
    routes = list(p.routes)
    mats = [p.raw["routes"][r]["material"] for r in routes]
    size = np.array([next(b.size_t for b in p.blocks if b.route == r) for r in routes])
    tlc = np.array([p.tlc[r] for r in routes], float)
    emis = np.array([p.emis[r] for r in routes], float)
    pool = np.array([sum(1 for b in p.blocks if b.route == r) for r in routes])
    # stress modifications (risk.stress_scenarios)
    if "H1.sea_freight_usd_multiplier" in mods:
        sea = p.raw["routes"]["H1"]["sea_freight_usd"]["value"]
        tlc[routes.index("H1")] += sea * (mods["H1.sea_freight_usd_multiplier"] - 1.0)
    if "R2.price_multiplier" in mods:
        price = p.raw["routes"]["R2"]["price_inr"]["value"] / p.fx
        tlc[routes.index("R2")] += price * (mods["R2.price_multiplier"] - 1.0)
    if "R4.pool" in mods:
        pool[routes.index("R4")] = mods["R4.pool"]
    materials = []
    for m in mats:
        if m not in materials:
            materials.append(m)
    return RouteModel(routes, mats, size, tlc, emis, pool, dict(p.demand), materials)


def count_vectors(rm):
    return np.array(list(itertools.product(*[range(n + 1) for n in rm.pool])), dtype=int)


def violation(rm, CV, F, chunk=256):
    """Fraction of scenarios in which any material's delivered tonnage < demand, for each count vector."""
    CV = np.atleast_2d(CV)
    out = np.empty(len(CV))
    for s in range(0, len(CV), chunk):
        cv = CV[s:s + chunk]
        ok = np.ones((len(cv), len(F)), bool)
        for m in rm.materials:
            r_idx = [i for i, mm in enumerate(rm.mats) if mm == m]
            deliv = (cv[:, r_idx] * rm.size[r_idx]) @ F[:, r_idx].T
            ok &= deliv >= rm.demand[m]
        out[s:s + chunk] = 1.0 - ok.mean(1)
    return out


def material_counts(rm, CV):
    return np.stack([CV[:, [i for i, mm in enumerate(rm.mats) if mm == m]].sum(1) for m in rm.materials], axis=1)


# ------------------------------------------------------------------ instance
def build_instance(rm, F_cal, weights, k, name="steel"):
    """Items = blocks (route order, then block). Q from calibration scenarios (objective.risk_term)."""
    item_route = np.concatenate([[r] * n for r, n in enumerate(rm.pool)]).astype(int)
    item_group = np.array([rm.materials.index(rm.mats[r]) for r in item_route])
    sizes = rm.size[item_route]
    cost = sizes * rm.tlc[item_route]
    emis = sizes * rm.emis[item_route]
    Y = F_cal[:, item_route] * sizes                       # delivered tonnes per item per scenario
    N = len(item_route)
    Q = np.zeros((N, N))
    for g, m in enumerate(rm.materials):
        idx = np.flatnonzero(item_group == g)
        if len(idx) == 0:
            continue
        C = np.atleast_2d(np.cov(Y[:, idx].T)) / rm.demand[m] ** 2
        Q[np.ix_(idx, idx)] = C
    labels = [f"{rm.routes[item_route[i]]}#{i}" for i in range(N)]
    inst = build_from_raw(name, rm.materials, item_group, k, cost, emis, Q, weights, labels,
                          {"item_route": item_route, "routes": rm.routes})
    return inst


def representative_bits(rm, CV):
    """Item bit vectors for count vectors (first c_r blocks of route r selected)."""
    X = []
    for cv in np.atleast_2d(CV):
        x = []
        for r, n in enumerate(rm.pool):
            x += [1] * cv[r] + [0] * (n - cv[r])
        X.append(x)
    return np.array(X, dtype=np.int8)


def counts_from_xint(rm, xint):
    item_route = np.concatenate([[r] * n for r, n in enumerate(rm.pool)]).astype(int)
    bits = (np.asarray(xint)[..., None] >> np.arange(len(item_route))) & 1
    return np.stack([bits[..., item_route == r].sum(-1) for r in range(len(rm.routes))], axis=-1)


# ------------------------------------------------------------------ statistics
def clopper_pearson_upper(x, n, conf=0.95):
    return 1.0 if x >= n else float(beta_dist.ppf(conf, x + 1, n - x))


def wilson_interval(successes, n, conf=0.95):
    z = norm.ppf(0.5 + conf / 2)
    ph = successes / n
    den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - h), float(c + h)


# ------------------------------------------------------------------ calibration pipeline
def calibrate(p, rm, weights, eps, eps_cal_grid, F_cal, F_val):
    """Safe-count rule with out-of-sample certification (risk.chance_constraint)."""
    CV = count_vectors(rm)
    v_cal = violation(rm, CV, F_cal)
    v_val = violation(rm, CV, F_val)
    MC = material_counts(rm, CV)
    inst0 = build_instance(rm, F_cal, weights, np.zeros(len(rm.materials), int))
    f_cv = inst0.f(representative_bits(rm, CV))
    n_val = len(F_val)
    ranges = [range(MC[:, g].max() + 1) for g in range(MC.shape[1])]
    records, accepted = [], None
    for ec in [e for e in eps_cal_grid if e <= eps + 1e-12]:
        best = None
        for kv in itertools.product(*ranges):
            meets = np.all(MC >= np.array(kv), axis=1)
            if not meets.any() or np.any(v_cal[meets] > ec):
                continue
            j = np.flatnonzero(meets)[np.argmin(f_cv[meets])]
            cand = (f_cv[j], sum(kv), kv, j)
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is None:
            records.append({"eps_cal": ec, "k": None, "certified": False, "note": "no safe k"})
            continue
        _, _, kv, j = best
        x = round(v_val[j] * n_val)
        conf = p.raw["statistics"]["confidence"]
        ub = clopper_pearson_upper(x, n_val, conf)
        # exact SAA optimum (no count-buffer approximation), for comparison
        okc = v_cal <= ec
        js = np.flatnonzero(okc)[np.argmin(f_cv[okc])] if okc.any() else None
        rec = {"eps_cal": ec, "k": list(map(int, kv)), "opt_counts": CV[j].tolist(), "opt_f": float(f_cv[j]),
               "viol_cal": float(v_cal[j]), "viol_val": float(v_val[j]), "cp95_upper": ub, "certified": ub <= eps,
               "saa_exact_counts": None if js is None else CV[js].tolist(),
               "saa_exact_f": None if js is None else float(f_cv[js]),
               "saa_exact_viol_val": None if js is None else float(v_val[js]),
               "saa_exact_cp95_upper": None if js is None else clopper_pearson_upper(round(v_val[js] * n_val), n_val, conf)}
        records.append(rec)
        if rec["certified"]:
            accepted = rec
            break
    return {"records": records, "accepted": accepted, "CV": CV, "v_cal": v_cal, "v_val": v_val, "f_cv": f_cv}


def steel_case(p, weights=None, eps=None, family="gaussian", nu=None, sigma=None, mods=None, rho_cross=None):
    """End-to-end: scenarios -> calibration -> Instance with certified k (or None)."""
    cc = p.raw["risk"]["chance_constraint"]
    weights = weights or {k: v for k, v in p.raw["objective"]["weights"].items() if k.startswith("w_")}
    eps = cc["epsilon"] if eps is None else eps
    rm = route_model(p, mods)
    F_cal, M_cal = sample_scenarios(p, cc["calibration"]["n_cal"], cc["calibration"]["seed"], family, nu, sigma, rho_cross)
    F_val, M_val = sample_scenarios(p, cc["validation"]["n_val"], cc["validation"]["seed"], family, nu, sigma, rho_cross)
    cal = calibrate(p, rm, weights, eps, cc["calibration"]["epsilon_cal_grid"], F_cal, F_val)
    inst = None
    if cal["accepted"] is not None:
        inst = build_instance(rm, F_cal, weights, np.array(cal["accepted"]["k"]))
    return {"rm": rm, "inst": inst, "cal": cal, "eps": eps, "weights": weights, "family": family, "sigma": sigma,
            "F_cal": F_cal, "M_cal": M_cal, "F_val": F_val, "M_val": M_val}
