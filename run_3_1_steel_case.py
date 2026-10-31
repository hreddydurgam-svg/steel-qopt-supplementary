"""
Section 3.1: steel case study (Salem testbed).

Outputs: variables_table.csv (every binary variable: route, material, block tonnes, US$/t, kg/t, coefficients),
calibration_table.csv (chance-constraint calibration and out-of-sample certification), worked_example.json
(step-by-step objective evaluation of the optimum), MILP cross-check (McCormick-linearised QUBO, optional),
classical_runs.csv, quantum_runs.csv, quantum_summary.csv, cvar_saturation.csv, summary.json (environment, budgets,
evaluations). The steel instance is small: it is the case study; algorithm comparisons use run_B_synthetic_suite.py.

`--alpha 1` trains on the expectation and writes to results/3_1_steel_case_<profile>_alpha1/.

Usage:  python run_3_1_steel_case.py --profile quick|paper [--alpha 1] [--methods M1,M2,M4,M5]
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.experiment import aggregate, run_classical, run_quantum, tune_penalties
from qopt.instance import solve_exact
from qopt.steel import counts_from_xint, steel_case

args = parse_profile(alpha=True, methods=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
alpha, suffix = apply_alpha(cfg, args)
out = results_dir(f"3_1_steel_case_{args.profile}{suffix}")
banner(f"3.1 STEEL CASE STUDY  (profile {args.profile}, CVaR alpha {alpha:g})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()

sc = steel_case(p)
cal = pd.DataFrame(sc["cal"]["records"])
cal.to_csv(out / "calibration_table.csv", index=False)
print(cal.to_string(index=False))
if sc["inst"] is None:
    raise SystemExit("No certified safe-count k at the configured epsilon; see calibration_table.csv")
inst, rm = sc["inst"], sc["rm"]
ex = solve_exact(inst)

# --- every binary variable ---------------------------------------------------------
rows = []
for i in range(inst.N):
    r = rm.routes[inst.meta["item_route"][i]]
    rows.append({"qubit_x": i, "route": r, "material": rm.mats[rm.routes.index(r)],
                 "block_t_per_month": rm.size[rm.routes.index(r)], "TLC_usd_per_t": rm.tlc[rm.routes.index(r)],
                 "E_kgCO2e_per_t": rm.emis[rm.routes.index(r)], "cost_usd_per_month": inst.cost_usd[i],
                 "emis_kg_per_month": inst.emis_kg[i], "linear_coef": inst.lin[i], "risk_self_coef": inst.W[i, i]})
pd.DataFrame(rows).to_csv(out / "variables_table.csv", index=False)

# --- exact optimum and worked example -------------------------------------------------
x_opt = ((ex.opt_set[0] >> np.arange(inst.N)) & 1)
c, e, r = inst.components(x_opt)
w = inst.weights
worked = {
    "demand_t_per_month": rm.demand, "k_counts_per_material": dict(zip(rm.materials, inst.k.tolist())),
    "epsilon": sc["eps"], "accepted_calibration": sc["cal"]["accepted"],
    "optimal_route_counts": dict(zip(rm.routes, counts_from_xint(rm, ex.opt_set[0]).tolist())),
    "n_feasible_item_bitstrings": ex.n_feasible, "n_optimal_item_bitstrings": int(len(ex.opt_set)),
    "cost_usd": c, "C_ref_usd": inst.refs["C_ref"], "cost_term": w["w_cost"] * c / inst.refs["C_ref"],
    "emissions_kg": e, "E_ref_kg": inst.refs["E_ref"], "emission_term": w["w_emit"] * e / inst.refs["E_ref"],
    "risk": r, "R_ref": inst.refs["R_ref"], "risk_term": w["w_risk"] * r / inst.refs["R_ref"],
    "f": float(inst.f(x_opt)[0]), "weights": w,
    "check_sum_of_terms": w["w_cost"] * c / inst.refs["C_ref"] + w["w_emit"] * e / inst.refs["E_ref"] + w["w_risk"] * r / inst.refs["R_ref"],
}
write_json(out / "worked_example.json", worked)
print(f"\nExact optimum f* = {ex.f_opt:.6f}; route counts {worked['optimal_route_counts']}; "
      f"{ex.n_feasible} feasible bitstrings, {len(ex.opt_set)} optimal (identical blocks are interchangeable)")

# --- MILP cross-check (optional) -------------------------------------------------------
milp_rec = {"status": "SKIPPED (pulp not installed)"}
try:
    from qopt.classical import milp
    xm, dt, ver = milp(inst)
    milp_rec = {"status": "OK", "f": float(inst.f(xm)[0]), "matches_enumeration": bool(abs(inst.f(xm)[0] - ex.f_opt) < 1e-7),
                "seconds": dt, "pulp_version": ver}
except ImportError:
    pass
print("MILP:", milp_rec)

# --- classical heuristics ----------------------------------------------------------------
seeds = list(range(cfg["classical"]["seeds"]["start"], cfg["classical"]["seeds"]["start"] + prof["seeds"]))
df_c = run_classical(inst, ex, cfg, seeds)
df_c.to_csv(out / "classical_runs.csv", index=False)
print(df_c.groupby("method")[["feasible", "hit_opt", "ar"]].mean())

# --- penalty tuning and quantum methods --------------------------------------------------
methods = args.methods.split(",") if args.methods else prof["steel_methods"]
tune_seeds = [cfg["methods"]["tuning"]["seed_offset"] + s for s in range(prof["tune_seeds"])]
to_tune = [m for m in ("M3", "M4") if m in methods]
tuned, tune_tab = tune_penalties([(inst, ex)], cfg, tune_seeds, prof["tune_p"], prof["maxiter"],
                                 cfg["methods"]["common"]["shots"], methods=to_tune) if to_tune else ({}, pd.DataFrame())
tune_tab.to_csv(out / "penalty_tuning.csv", index=False)
df_q, skipped = run_quantum(inst, ex, cfg, methods, list(range(prof["seeds"])), prof["p_max"], prof["maxiter"],
                            cfg["methods"]["common"]["shots"], tuned)
df_q.to_csv(out / "quantum_runs.csv", index=False)
agg = aggregate(df_q)
agg.to_csv(out / "quantum_summary.csv", index=False)
print(agg.to_string(index=False))
sat = (df_q.groupby(["method", "p"]).agg(cvar_saturated_frac=("cvar_saturated", "mean"), seeds=("seed", "count")).reset_index()
       if not df_q.empty else pd.DataFrame())
sat.to_csv(out / "cvar_saturation.csv", index=False)
if not sat.empty:
    print("Fraction of seeds whose training CVaR reached f* (cvar_saturated):")
    print(sat.to_string(index=False))

write_json(out / "summary.json", {
    "environment": environment(), "params_sha256": p.sha256, "profile": args.profile, "cvar_alpha": alpha,
    "methods": methods, "tuned_penalties": tuned,
    "skipped_methods": skipped, "milp": milp_rec, "exact_f_opt": ex.f_opt, "wall_seconds": time.time() - t_all,
    "note": ("Case study. The structural methods (M1, M2) can reach only the feasible set; with identical blocks the "
             "steel instance has few feasible bitstrings and many tied optima, so P(optimum) is compared with "
             "uniform_p_opt. Algorithm comparisons are made on the synthetic suite (run_B_synthetic_suite.py).")})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
