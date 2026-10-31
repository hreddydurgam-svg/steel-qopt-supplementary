"""
Section 3.6: cost-emissions trade-off on the steel case.

ALL feasible configurations (route counts meeting the certified safe counts k) are enumerated and listed; the
exact non-dominated set is computed in (cost, emissions) and in (cost, emissions, risk); supported vs unsupported
points are identified (lower convex hull); weighted-sum optima (exact) and the headline quantum method's modal
best-sampled solutions are marked against it. Risk weight is held at w_risk_fixed; w_cost is swept.
`--alpha 1` trains on the expectation and writes to results/3_6_pareto_<profile>_alpha1/.
Usage:  python run_3_6_pareto.py --profile quick|paper [--alpha 1]
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.experiment import run_quantum
from qopt.instance import solve_exact
from qopt.plotting import COLOR, MARKER, save, style
from qopt.steel import build_instance, count_vectors, counts_from_xint, material_counts, representative_bits, steel_case

import matplotlib.pyplot as plt

args = parse_profile(alpha=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
alpha, suffix = apply_alpha(cfg, args)
out = results_dir(f"3_6_pareto_{args.profile}{suffix}")
banner(f"3.6 COST-EMISSIONS TRADE-OFF  (profile {args.profile}, CVaR alpha {alpha:g})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()

sc = steel_case(p)
if sc["inst"] is None:
    raise SystemExit("no certified safe counts")
rm, k = sc["rm"], sc["inst"].k
CV = count_vectors(rm)
feas = np.all(material_counts(rm, CV) >= k, axis=1)
CVf = CV[feas]
X = representative_bits(rm, CVf).astype(float)
base = sc["inst"]
cost = X @ base.cost_usd
emis = X @ base.emis_kg
risk = np.einsum("si,ij,sj->s", X, base.Q_raw, X)
configs = pd.DataFrame({"route_counts": [dict(zip(rm.routes, map(int, c))) for c in CVf], "cost_usd": cost, "emis_kg": emis, "risk": risk})

# exact non-dominated set in (cost, emissions); then supported points = lower-left convex hull vertices
order = np.lexsort((emis, cost))
nd, best_e = [], np.inf
for i in order:
    if emis[i] < best_e - 1e-9:
        nd.append(i); best_e = emis[i]
nd = np.array(nd)
pts = sorted(zip(cost[nd], emis[nd], nd))
hull = []
for pt in pts:
    while len(hull) >= 2:
        (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
        if (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1) <= 0:
            hull.pop()
        else:
            break
    hull.append(pt)
supported = {h[2] for h in hull}
configs["non_dominated"] = False
configs.loc[nd, "non_dominated"] = True
configs["supported"] = [i in supported for i in range(len(configs))]
V = np.stack([cost, emis, risk], 1)
configs["non_dominated_cost_emis_risk"] = [not np.any(np.all(V <= V[i], 1) & np.any(V < V[i], 1)) for i in range(len(V))]
configs.to_csv(out / "all_feasible_configurations.csv", index=False)
print(f"{len(configs)} feasible configurations; {len(nd)} non-dominated (cost, emissions); "
      f"{len(supported)} supported, {len(nd) - len(supported)} unsupported; "
      f"{int(configs['non_dominated_cost_emis_risk'].sum())} non-dominated in (cost, emissions, risk)")

# weighted-sum sweep: exact and headline method
H = cfg["methods"]["headline"]
ws = cfg["objective"]["weight_sweep"]
w_r = ws["w_risk_fixed"]
sweep = []
for wc in np.round(np.arange(ws["w_cost_start"], ws["w_cost_stop"] + 1e-9, ws["step"]), 2):
    w = {"w_cost": float(wc), "w_emit": float(round(1 - w_r - wc, 2)), "w_risk": w_r}
    inst = build_instance(rm, sc["F_cal"], w, k)
    ex = solve_exact(inst)
    ex_cv = counts_from_xint(rm, ex.opt_set[0])
    dq, _ = run_quantum(inst, ex, cfg, [H], list(range(prof["seeds"])), prof["p_max"], prof["maxiter"],
                        cfg["methods"]["common"]["shots"], {})
    last = dq[dq["p"] == prof["p_max"]]
    q_cvs = [tuple(counts_from_xint(rm, int(x))) for x in last["best_sampled_xint"].dropna()]
    q_mode = max(set(q_cvs), key=q_cvs.count)
    idx_ex = int(np.flatnonzero((CVf == ex_cv).all(1))[0])
    idx_q = int(np.flatnonzero((CVf == np.array(q_mode)).all(1))[0])
    sweep.append({**w, "exact_config": idx_ex, "exact_route_counts": dict(zip(rm.routes, map(int, ex_cv))),
                  "method": H, "q_modal_config": idx_q, "q_route_counts": dict(zip(rm.routes, map(int, q_mode))),
                  "q_matches_exact": idx_q == idx_ex, "q_seeds_hitting_exact": int(sum(c == tuple(ex_cv) for c in q_cvs)),
                  "q_p_opt_median": float(last["p_opt"].median())})
    print(f"  w={w}: exact {sweep[-1]['exact_route_counts']} | {H} modal {sweep[-1]['q_route_counts']} "
          f"({sweep[-1]['q_seeds_hitting_exact']}/{len(q_cvs)} seeds hit exact)")
sw = pd.DataFrame(sweep)
sw.to_csv(out / "weight_sweep.csv", index=False)
uniq_exact = sorted(set(sw["exact_config"]))
uniq_q = sorted(set(sw["q_modal_config"]))
print(f"Unique configurations: exact weighted-sum {len(uniq_exact)}, {H} {len(uniq_q)}; "
      f"non-dominated recovered by weighted sum {len(set(uniq_exact) & set(nd.tolist()))} of {len(nd)}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter(cost / 1e6, emis / 1e3, s=18, color="#c3c2b7", label="feasible configurations")
ndp = configs.loc[nd].sort_values("cost_usd")
ax.plot(ndp["cost_usd"] / 1e6, ndp["emis_kg"] / 1e3, color="#222222", linewidth=1.5, marker="o", markersize=5, label="exact non-dominated set")
ax.scatter(cost[uniq_exact] / 1e6, emis[uniq_exact] / 1e3, s=110, facecolors="none", edgecolors="#222222", linewidths=2,
           marker="s", label="weighted-sum optima (exact)")
ax.scatter(cost[uniq_q] / 1e6, emis[uniq_q] / 1e3, s=60, color=COLOR[H], marker=MARKER[H], label=f"{H} modal best-sampled")
style(ax, "Steel case: cost vs upstream transport emissions", "cost (US$ million / month)", "emissions (t CO₂e / month)")
ax.legend(fontsize=7)
save(fig, out / "fig_pareto.png")
write_json(out / "summary.json", {"environment": environment(), "params_sha256": p.sha256, "profile": args.profile, "cvar_alpha": alpha,
                                  "n_feasible_configs": len(configs), "n_non_dominated": int(len(nd)),
                                  "n_supported": len(supported), "n_non_dominated_3d": int(configs["non_dominated_cost_emis_risk"].sum()),
                                  "weighted_sum_optima_all_3d_non_dominated": bool(configs.loc[uniq_exact, "non_dominated_cost_emis_risk"].all()), "unique_exact_weighted": len(uniq_exact),
                                  "method": H, "unique_quantum": len(uniq_q), "wall_seconds": time.time() - t_all})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
