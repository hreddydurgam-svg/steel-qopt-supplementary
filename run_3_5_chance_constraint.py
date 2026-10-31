"""
Section 3.5: the chance constraint, written, calibrated and validated out of sample.

    P_xi[ delivered_m(x, xi) >= D_m  for m in {coal, ore, scrap} ] >= 1 - epsilon        (joint chance constraint)
    delivered_m(x, xi) = sum_{blocks i of material m} b_i x_i max(0, 1 - G_route(i)(xi)),  G ~ GPD via copula

Cases are read from risk.chance_constraint.validation_cases (Gaussian base case; Student-t copula for tail
dependence; tighter epsilon with milder severity). Calibration by SAA on N_cal scenarios; validation on N_val
independent scenarios with Clopper-Pearson and Wilson bounds; paired scenario-wise regret (medians, IQR, bootstrap
CIs); every failure listed; equivalence by TOST with a pre-specified margin. The headline quantum method commits
to ONE bitstring before the validation scenarios are used.
`--alpha 1` trains on the expectation and writes to results/3_5_chance_constraint_<profile>_alpha1/.
Usage:  python run_3_5_chance_constraint.py --profile quick|paper [--alpha 1]
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.experiment import run_quantum
from qopt.instance import solve_exact
from qopt.plotting import COLOR, save, style
from qopt.stats import bootstrap_ci, summary, tost_paired, wilcoxon_paired
from qopt.steel import (clopper_pearson_upper, count_vectors, counts_from_xint, material_counts, representative_bits,
                        steel_case, wilson_interval)

import matplotlib.pyplot as plt

args = parse_profile(alpha=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
cc = cfg["risk"]["chance_constraint"]
margin = cc["tost_margin_objective"]["value"]
st = cfg["statistics"]
CONF = st["confidence"]
BOOT = dict(n_boot=st["bootstrap_resamples"], conf=CONF, seed=st["bootstrap_seed"])
alpha, suffix = apply_alpha(cfg, args)
out = results_dir(f"3_5_chance_constraint_{args.profile}{suffix}")
banner(f"3.5 CHANCE CONSTRAINT  (profile {args.profile}, CVaR alpha {alpha:g})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()

H = cfg["methods"]["headline"]
HC = f"{H}_committed"
cases = [{"case": v["id"], "family": v["family"], "nu": v["nu"], "sigma": v["sigma"], "eps": v["epsilon"]}
         for v in cc["validation_cases"]]
all_sol, all_fail, all_tests, all_cal = [], [], [], []
fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 3.6), squeeze=False)
for ci, case in enumerate(cases):
    print(f"\n--- {case['case']}")
    sc = steel_case(p, eps=case["eps"], family=case["family"], sigma=case["sigma"],
                    **({"nu": case["nu"]} if case["nu"] is not None else {}))
    for r in sc["cal"]["records"]:
        all_cal.append({"case": case["case"], **r})
    if sc["inst"] is None:
        print("  no certified safe counts at this epsilon; see calibration table")
        continue
    inst, rm = sc["inst"], sc["rm"]
    ex = solve_exact(inst)
    CV = count_vectors(rm)
    acc = sc["cal"]["accepted"]
    w = inst.weights
    # candidate solutions (route-count vectors)
    nominal_t = np.stack([(CV * rm.size * np.array([mm == m for mm in rm.mats])).sum(1) for m in rm.materials], 1)
    nominal_ok = np.all(nominal_t >= np.array([rm.demand[m] for m in rm.materials]), axis=1)   # no shocks (F = 1)
    f_cv = sc["cal"]["f_cv"]
    naive = CV[np.flatnonzero(nominal_ok)[np.argmin(f_cv[nominal_ok])]]
    dq, _ = run_quantum(inst, ex, cfg, [H], list(range(prof["seeds"])), prof["p_max"], prof["maxiter"],
                        cfg["methods"]["common"]["shots"], {})
    last = dq[dq["p"] == prof["p_max"]].sort_values(["best_sampled_f", "p_opt"], ascending=[True, False])
    committed_x = int(last.iloc[0]["best_sampled_xint"])          # committed BEFORE validation scenarios are used
    sols = {"exact_safe_count": np.array(acc["opt_counts"]), "exact_SAA_no_buffer_rule": np.array(acc["saa_exact_counts"]),
            "deterministic_nominal": naive, HC: counts_from_xint(rm, committed_x)}
    # validation scenarios: realised objective per scenario and ex-post optimum
    F, M = sc["F_val"], sc["M_val"]
    risk_cv = np.array([float(x @ inst.Q_raw @ x) for x in representative_bits(rm, CV).astype(float)])

    def realised(cvs):
        cvs = np.atleast_2d(cvs)
        cost = (cvs * rm.size * rm.tlc) @ M.T
        emis = (cvs * rm.size * rm.emis) @ F.T
        ok = np.ones(cost.shape, bool)
        short = {}
        for m in rm.materials:
            idx = [i for i, mm in enumerate(rm.mats) if mm == m]
            deliv = (cvs[:, idx] * rm.size[idx]) @ F[:, idx].T
            ok &= deliv >= rm.demand[m]
            short[m] = np.maximum(0.0, rm.demand[m] - deliv)
        return cost, emis, ok, short

    best_post = np.full(len(F), np.inf)
    for s0 in range(0, len(CV), 128):
        cvs = CV[s0:s0 + 128]
        cost, emis, ok, _ = realised(cvs)
        rk = np.array([risk_cv[np.flatnonzero((CV == c).all(1))[0]] for c in cvs])
        fr = w["w_cost"] * cost / inst.refs["C_ref"] + w["w_emit"] * emis / inst.refs["E_ref"] + (w["w_risk"] * rk / inst.refs["R_ref"])[:, None]
        best_post = np.minimum(best_post, np.where(ok, fr, np.inf).min(0))
    f_real = {}
    n_val = len(F)
    for name, cv in sols.items():
        cost, emis, ok, short = realised(cv)
        rk = risk_cv[np.flatnonzero((CV == cv).all(1))[0]]
        fr = (w["w_cost"] * cost / inst.refs["C_ref"] + w["w_emit"] * emis / inst.refs["E_ref"] + w["w_risk"] * rk / inst.refs["R_ref"])[0]
        f_real[name] = fr
        ok = ok[0]
        viol = int((~ok).sum())
        regret = np.where(ok, fr - best_post, np.nan)
        lo, hi = wilson_interval(n_val - viol, n_val, CONF)
        rs = summary(regret)
        all_sol.append({"case": case["case"], "solution": name, "route_counts": dict(zip(rm.routes, map(int, cv))),
                        "violations": viol, "violation_rate": viol / n_val, "cp95_upper": clopper_pearson_upper(viol, n_val, CONF),
                        "certified_at_eps": clopper_pearson_upper(viol, n_val, CONF) <= case["eps"],
                        "survival_wilson95_lo": lo, "survival_wilson95_hi": hi,
                        "regret_median": rs.get("median"), "regret_q1": rs.get("q1"), "regret_q3": rs.get("q3"),
                        "regret_median_ci95": bootstrap_ci(regret, **BOOT), "f_realised_mean": float(fr.mean())})
        for si in np.flatnonzero(~ok):
            all_fail.append({"case": case["case"], "solution": name, "scenario": int(si),
                             **{f"shortfall_t_{m}": float(short[m][0, si]) for m in rm.materials}})
        r_ok = np.sort(regret[~np.isnan(regret)])
        if name in ("exact_safe_count", HC, "deterministic_nominal"):
            axes[0][ci].plot(r_ok, np.arange(1, len(r_ok) + 1) / n_val, linewidth=2,
                             color={"exact_safe_count": "#222222", HC: COLOR[H], "deterministic_nominal": "#7a7a7a"}[name],
                             linestyle={"exact_safe_count": "-", HC: "--", "deterministic_nominal": ":"}[name], label=name)
        print(f"  {name:28s} counts {dict(zip(rm.routes, map(int, cv)))}  violations {viol}/{n_val} "
              f"(CP95 {clopper_pearson_upper(viol, n_val, CONF):.4f})  regret median {rs.get('median')}")
    d = f_real[HC] - f_real["exact_safe_count"]
    all_tests.append({"case": case["case"], "comparison": f"{HC} - exact_safe_count (realised objective, all scenarios)",
                      "identical_plan": bool(np.array_equal(sols[HC], sols["exact_safe_count"])),
                      **wilcoxon_paired(f_real[HC], f_real["exact_safe_count"]),
                      "median_diff": float(np.median(d)), "median_diff_ci95": bootstrap_ci(d, **BOOT), **tost_paired(d, margin, alpha=st["tost_alpha"])})
    style(axes[0][ci], case["case"], "regret vs ex-post optimum (normalised objective)", "fraction of validation scenarios")
    axes[0][ci].legend(fontsize=7)
save(fig, out / "fig_regret_ecdf.png")
pd.DataFrame(all_cal).to_csv(out / "calibration.csv", index=False)
pd.DataFrame(all_sol).to_csv(out / "solutions_validation.csv", index=False)
pd.DataFrame(all_fail).to_csv(out / "failures_listed.csv", index=False)
pd.DataFrame(all_tests).to_csv(out / "paired_tests.csv", index=False)
print(pd.DataFrame(all_tests).to_string(index=False))
write_json(out / "summary.json", {"environment": environment(), "params_sha256": p.sha256, "profile": args.profile, "cvar_alpha": alpha,
                                  "tost_margin": margin, "wall_seconds": time.time() - t_all,
                                  "method": H,
                                  "note": ("Regret is computed only on scenarios where the solution meets demand; every "
                                           "failure is listed in failures_listed.csv with shortfall tonnes. When identical_plan "
                                           "is true the paired tests are trivially exact (difference 0 in every scenario).")})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
