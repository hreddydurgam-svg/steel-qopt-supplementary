"""
load_params_v18.py
Single loader for the Salem testbed (spec v18). Model scripts import `load()` and must not
hard-code any data, risk or solver number. Running this file performs the lock checks.

Only unit constants appear here: 12 months/year, 1000 g/kg, 1000 kg/t.
"""
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist, genpareto, norm, t as t_dist

MONTHS = 12
G_PER_KG = 1000.0
DEFAULT_PATH = Path(__file__).with_name("testbed_params_v18.json")


@dataclass
class Block:
    idx: int
    route: str
    material: str
    size_t: float
    tlc_usd_t: float
    emis_kg_t: float


@dataclass
class Params:
    raw: dict
    sha256: str
    fx: float
    demand: dict
    tlc: dict
    emis: dict
    blocks: list = field(default_factory=list)
    routes: list = field(default_factory=list)
    c_ref: float = 0.0
    e_ref: float = 0.0


def _get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d["value"] if isinstance(d, dict) and "value" in d else d


def load(path=DEFAULT_PATH, scrap_case=None):
    text = Path(path).read_bytes()
    raw = json.loads(text)
    sha = hashlib.sha256(text).hexdigest()
    fx = raw["fx_inr_per_usd"]["value"]
    rc, rd = raw["rail_common"], raw["road_common"]
    rake_t = rc["wagons_per_rake"] * rc["pcc_t_per_wagon"]
    gst_div = (1 + rc["gst_rate"]) if rc["gst_treatment"] == "exclude" else 1.0

    # demands (t/month)
    dm = raw["demand_t_per_month"]
    s = dm["scrap"]
    case = scrap_case or s["base_case"]
    e = s["eaf_only"]
    scrap_eaf = e["billet_annual_t"] / MONTHS * e["eaf_to_bof_ratio"][0] / e["eaf_to_bof_ratio"][1] * e["scrap_t_per_t"]
    scrap = scrap_eaf + (s["sensitivity_upper"]["add_melting_shop_annual_t"] / MONTHS if case == "sensitivity_upper" else 0.0)
    demand = {"coal": dm["coal"]["annual_t"] / MONTHS,
              "ore": sum(dm["ore"]["annual_t_components"]) / MONTHS,
              "scrap": scrap}

    # per-tonne cost and emissions per route
    ef = {k: v["value"] for k, v in raw["emission_factors_g_tkm"].items()}
    tlc, emis = {}, {}
    for r, d in raw["routes"].items():
        price = d["price_usd"]["value"] if "price_usd" in d else d["price_inr"]["value"] / fx
        sea = d.get("sea_freight_usd", {}).get("value", 0.0)
        port = d.get("port_handling_inr", {}).get("value", 0.0) / fx
        rail = d["rail"]["rake_total_inr_incl_gst"] / rake_t / gst_div / fx if "rail" in d else 0.0
        road = rd["truck_rate_inr_tkm"] * d["road_km"]["value"] / fx if "road_km" in d else 0.0
        tlc[r] = price + sea + port + rail + road
        emis[r] = sum(ef[mode] * _get(d, key) / G_PER_KG for mode, key in d["legs_ef"]) + d["hub_ef_kg_t"]["value"]

    # blocks
    blocks, idx = [], 0
    for m, spec in raw["blocks"].items():
        if m.startswith("_") or not isinstance(spec, dict) or "pool" not in spec:
            continue
        unit = rake_t if spec["unit_t"] == "rake" else float(spec["unit_t"])
        size = math.ceil(demand[m] / spec["k_req"] / unit) * unit
        for r, n in spec["pool"].items():
            for _ in range(n):
                blocks.append(Block(idx, r, m, size, tlc[r], emis[r]))
                idx += 1
    p = Params(raw, sha, fx, demand, tlc, emis, blocks, list(raw["routes"]))
    p.c_ref = sum(b.size_t * b.tlc_usd_t for b in blocks)
    p.e_ref = sum(b.size_t * b.emis_kg_t for b in blocks)
    return p


def route_corr(p, rho_cross=None):
    cop = p.raw["risk"]["copula"]
    rho_m = cop["rho_same_material_other_route"]
    rho_x = cop["rho_cross_material"] if rho_cross is None else rho_cross
    mat = {r: p.raw["routes"][r]["material"] for r in p.routes}
    C = np.eye(len(p.routes))
    for i, a in enumerate(p.routes):
        for j, b in enumerate(p.routes):
            if i != j:
                C[i, j] = rho_m if mat[a] == mat[b] else rho_x
    return C


def sample_route_factors(p, n, seed, family="gaussian", nu=None, sigma=None, rho_cross=None):
    """Delivered fraction per route, shape (n, n_routes): max(0, 1 - G), G ~ GPD via copula."""
    cs = p.raw["risk"]["capacity_shock"]
    C = route_corr(p, rho_cross)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(p.routes))) @ np.linalg.cholesky(C).T
    if family == "gaussian":
        U = norm.cdf(Z)
    else:
        W = rng.chisquare(nu, (n, 1)) / nu
        U = t_dist.cdf(Z / np.sqrt(W), nu)
    G = genpareto.ppf(U, c=cs["xi"], loc=cs["mu"], scale=sigma if sigma is not None else cs["sigma"])
    return np.clip(1.0 - G, 0.0, None)


def violation_rate(p, counts, F):
    """counts: dict route -> number of blocks selected; F: route factors."""
    ok = np.ones(len(F), bool)
    size = {b.route: b.size_t for b in p.blocks}
    for m, D in p.demand.items():
        tot = sum(F[:, i] * counts.get(r, 0) * size[r]
                  for i, r in enumerate(p.routes) if p.raw["routes"][r]["material"] == m)
        ok &= tot >= D
    return 1.0 - ok.mean()


def clopper_pearson_upper(x, n, conf=0.95):
    return 1.0 if x >= n else float(beta_dist.ppf(conf, x + 1, n - x))


def _self_test():
    p = load()
    raw = p.raw
    print(f"spec v{raw['_spec']['version']}  sha256 {p.sha256}")
    print("Demand t/month:", {k: round(v, 1) for k, v in p.demand.items()})
    print("Scrap upper-bound case:", round(load(scrap_case="sensitivity_upper").demand["scrap"], 1))
    tgt = raw["regression_targets"]
    ok = True
    for r in p.routes:
        dt, de = p.tlc[r] - tgt["tlc_usd_t"][r], p.emis[r] - tgt["emissions_kg_t"][r]
        tol = tgt["tolerance"]
        flag = "OK" if abs(dt) <= tol and abs(de) <= tol else "MISMATCH"
        ok &= flag == "OK"
        print(f"  {r}: TLC {p.tlc[r]:.4f} (table {tgt['tlc_usd_t'][r]})  E {p.emis[r]:.4f} (table {tgt['emissions_kg_t'][r]})  {flag}")
    print("Blocks (qubits):", len(p.blocks))
    for b in p.blocks:
        print(f"  x{b.idx:<2} {b.route} {b.material:<5} {b.size_t:>10,.1f} t  {b.tlc_usd_t:8.2f} $/t  {b.emis_kg_t:6.2f} kg/t")
    # capacity checks
    caps = raw["route_capacity_caps_t_per_month"]
    for r in p.routes:
        pool_t = sum(b.size_t for b in p.blocks if b.route == r)
        cap = caps[r]["value"]
        status = "no cap (NF)" if cap is None else ("OK" if pool_t <= cap else "EXCEEDS CAP")
        ok &= status != "EXCEEDS CAP"
        print(f"  pool {r}: {pool_t:,.1f} t vs cap {cap} -> {status}")
    for m, D in p.demand.items():
        tot = sum(b.size_t for b in p.blocks if b.material == m)
        print(f"  nominal {m}: pool {tot:,.1f} >= D {D:,.1f}: {tot >= D}")
        ok &= tot >= D
    C = route_corr(p)
    psd = bool(np.all(np.linalg.eigvalsh(C) > 0))
    print("Route correlation positive definite:", psd)
    ok &= psd
    print(f"C_ref = US${p.c_ref:,.2f}  E_ref = {p.e_ref:,.2f} kg CO2e")

    # chance-constraint pre-check (count vectors, calibration sample)
    cc = raw["risk"]["chance_constraint"]
    F = sample_route_factors(p, cc["calibration"]["n_cal"], cc["calibration"]["seed"])
    pools = {r: sum(1 for b in p.blocks if b.route == r) for r in p.routes}
    w = raw["objective"]["weights"]
    size = {b.route: b.size_t for b in p.blocks}
    feas = []
    for counts in itertools.product(*[range(pools[r] + 1) for r in p.routes]):
        k = dict(zip(p.routes, counts))
        v = violation_rate(p, k, F)
        if v <= cc["epsilon"]:
            cost = sum(k[r] * size[r] * p.tlc[r] for r in p.routes)
            em = sum(k[r] * size[r] * p.emis[r] for r in p.routes)
            feas.append((w["w_cost"] * cost / p.c_ref + w["w_emit"] * em / p.e_ref, counts, v))
    feas.sort()
    print(f"Feasible count vectors at epsilon={cc['epsilon']} (calibration sample): {len(feas)}")
    all_on = violation_rate(p, pools, F)
    eps_t = min(e["epsilon"] if isinstance(e, dict) else e for e in cc["sensitivity_epsilon"])
    print(f"All blocks on: calibration violation {all_on:.4f}  (epsilon {eps_t} feasible at base severity: {all_on <= eps_t})")
    if feas:
        Fv = sample_route_factors(p, cc["validation"]["n_val"], cc["validation"]["seed"])
        f, counts, v = feas[0]
        vv = violation_rate(p, dict(zip(p.routes, counts)), Fv)
        n = cc["validation"]["n_val"]
        ub = clopper_pearson_upper(round(vv * n), n, raw["statistics"]["confidence"])
        print(f"Pre-check, cheapest count vector at epsilon_cal=epsilon: counts {dict(zip(p.routes, counts))} cost+emission terms={f:.6f} (risk term excluded in this pre-check) "
              f"val viol {vv:.4f} CP95 upper {ub:.4f} -> certified {ub <= cc['epsilon']}")
    print("LOCK CHECKS:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
