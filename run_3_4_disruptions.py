"""
Section 3.4: stress scenarios on the steel case.

Each scenario re-runs the full pipeline: scenarios -> chance-constraint calibration -> safe counts k -> exact
optimum -> headline quantum method (seeds) -> gap. Reported per scenario: route counts of the exact optimum and of
the quantum method, whether the optimum moves relative to the baseline, whether a certified plan exists, and the
mean gap with and without the baseline. Normalisation constants are each scenario's own all-on references.
`--alpha 1` trains on the expectation and writes to results/3_4_disruptions_<profile>_alpha1/.
Usage:  python run_3_4_disruptions.py --profile quick|paper [--alpha 1]
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.experiment import run_quantum
from qopt.instance import solve_exact
from qopt.plotting import COLOR, save, style
from qopt.steel import counts_from_xint, steel_case

import matplotlib.pyplot as plt

args = parse_profile(alpha=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
alpha, suffix = apply_alpha(cfg, args)
out = results_dir(f"3_4_disruptions_{args.profile}{suffix}")
banner(f"3.4 STRESS SCENARIOS  (profile {args.profile}, CVaR alpha {alpha:g})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()
base_w = {k: v for k, v in cfg["objective"]["weights"].items() if k.startswith("w_")}
H = cfg["methods"]["headline"]
scen = [{"id": "S0_baseline"}] + cfg["risk"]["stress_scenarios"]["list"]
rows, runs = [], []
base_counts = None
for s in scen:
    w = {k: s.get(k, base_w[k]) for k in base_w}
    mods = {k: v for k, v in s.items() if "." in k}
    sc = steel_case(p, weights=w, mods=mods)
    if sc["inst"] is None:
        rows.append({"scenario": s["id"], "status": "no certified safe counts", "calibration": sc["cal"]["records"]})
        print(f"  {s['id']}: no certified safe counts")
        continue
    inst, rm = sc["inst"], sc["rm"]
    ex = solve_exact(inst)
    exact_counts = dict(zip(rm.routes, counts_from_xint(rm, ex.opt_set[0]).tolist()))
    if base_counts is None:
        base_counts = exact_counts
    moved = exact_counts != base_counts
    dq, skipped = run_quantum(inst, ex, cfg, [H], list(range(prof["seeds"])), prof["p_max"], prof["maxiter"],
                              cfg["methods"]["common"]["shots"], {})
    dq["scenario"] = s["id"]
    runs.append(dq)
    best = dq[dq["p"] == prof["p_max"]]
    i_best = best["best_sampled_f"].idxmin()
    q_counts = dict(zip(rm.routes, counts_from_xint(rm, int(best.loc[i_best, "best_sampled_xint"])).tolist()))
    gap = (best["best_sampled_f"] - ex.f_opt) / ex.f_opt * 100
    rec = {"scenario": s["id"], "weights": w, "mods": mods, "k": inst.k.tolist(), "eps_cal": sc["cal"]["accepted"]["eps_cal"],
           "cp95_upper": sc["cal"]["accepted"]["cp95_upper"], "exact_f": ex.f_opt, "exact_route_counts": exact_counts,
           "optimum_moved_vs_baseline": moved, "method": H, "best_route_counts": q_counts,
           "gap_pct_median": float(gap.median()), "gap_pct_max": float(gap.max()),
           "p_opt_median": float(best["p_opt"].median()), "C_ref": inst.refs["C_ref"], "E_ref": inst.refs["E_ref"]}
    rows.append(rec)
    print(f"  {s['id']}: exact {exact_counts} moved={moved}; {H} best {q_counts}; gap median {rec['gap_pct_median']:.3f}%")
df = pd.DataFrame(rows)
df.to_csv(out / "scenarios.csv", index=False)
pd.concat(runs).to_csv(out / "quantum_runs.csv", index=False)
ok = df.dropna(subset=["gap_pct_median"]) if "gap_pct_median" in df else df
summary = {
    "method": H, "mean_gap_pct_including_baseline": float(ok["gap_pct_median"].mean()),
    "mean_gap_pct_excluding_baseline": float(ok[ok["scenario"] != "S0_baseline"]["gap_pct_median"].mean()),
    "scenarios_that_move_optimum": ok[ok["optimum_moved_vs_baseline"] == True]["scenario"].tolist(),
    "scenarios_that_do_not_move_optimum": ok[ok["optimum_moved_vs_baseline"] == False]["scenario"].tolist(),
    "scenarios_without_certified_plan": [r["scenario"] for r in rows if "exact_route_counts" not in r],
}
print(summary)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
routes = list(base_counts)
x = np.arange(len(ok))
width = 0.8 / len(routes)
greys = ["#1c5cab", "#3987e5", "#86b6ef", "#b7d3f6", "#cde2fb"]       # one-hue ramp, route identity by position + legend
for j, r in enumerate(routes):
    ax.bar(x + j * width, [c[r] for c in ok["exact_route_counts"]], width * 0.9, color=greys[j], label=r)
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(x + 0.4 - width / 2)
ax.set_xticklabels(ok["scenario"], rotation=15, fontsize=8)
missing = [r["scenario"] for r in rows if "exact_route_counts" not in r]
style(ax, "Exact optimal blocks per route, by scenario" + (f"  (no certified plan: {', '.join(missing)})" if missing else ""),
      "", "blocks selected")
ax.legend(fontsize=7, title="route", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
save(fig, out / "fig_route_counts.png")
write_json(out / "summary.json", {"environment": environment(), "params_sha256": p.sha256, "profile": args.profile, "cvar_alpha": alpha,
                                  **summary, "wall_seconds": time.time() - t_all})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
