"""
Benchmark B: algorithm comparison on a SYNTHETIC suite of multi-group cardinality-constrained QUBOs.

This suite varies size and instance, uses many seeds and fixed evaluation budgets, sweeps depth p = 1..p_max, and reports success
probability, feasibility, approximation ratio, evaluations and qubit counts (simulator cost is not a quantum
resource). Methods listed in synthetic_suite.method_exclusions are not run at the stated sizes.

Hypotheses (pre-specified; reported whatever the outcome; Wilcoxon tests are two-sided, Holm over all rows):
  H1  M1 and M2 have P(feasible) = 1 by construction; the penalty methods (M3, M4) do not.
  H2  M1 median P(optimum) > M3 and M4 at equal p (paired Wilcoxon over instances, Holm-corrected).
  H3  M1 vs M5 (MA-PC): two-sided paired comparison of P(optimum), reported with parameter counts and P(feasible).
CVaR training (alpha = methods.common.cvar_alpha) and the declared sensitivity alpha (--alpha 1) are both run;
cvar_saturated flags runs whose training CVaR reached f* (CVaR <= f* + tolerance).
No claim against classical solvers: exhaustive enumeration solves every instance exactly.

Usage:  python run_B_synthetic_suite.py --profile quick|paper [--alpha 1]   (re-running resumes finished instances)
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.experiment import aggregate, run_classical, run_quantum, tune_penalties
from qopt.instance import solve_exact, synthetic_instance
from qopt.plotting import COLOR, LABEL, MARKER, save, style
from qopt.stats import holm, wilcoxon_paired

import matplotlib.pyplot as plt

args = parse_profile(alpha=True, methods=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
spec = cfg["synthetic_suite"]
alpha, suffix = apply_alpha(cfg, args)
out = results_dir(f"B_synthetic_suite_{args.profile}{suffix}")
banner(f"B SYNTHETIC SUITE  (profile {args.profile}, CVaR alpha {alpha})", p)
set_checkpoint(out, p.sha256)
methods = args.methods.split(",") if args.methods else spec["methods"]
sizes = prof["synthetic_sizes"]
n_inst = prof["synthetic_instances_per_size"]
seeds = list(range(prof["seeds"]))
shots = cfg["methods"]["common"]["shots"]

q_path, c_path, i_path = out / "quantum_runs.csv", out / "classical_runs.csv", out / "instances.csv"
done = set()
if i_path.exists():
    prev = pd.read_csv(i_path)
    if "params_sha256" not in prev or set(prev["params_sha256"]) != {p.sha256} or \
            set(prev.get("cvar_alpha", pd.Series([alpha]))) != {alpha}:
        raise SystemExit(f"{out} holds results from a different parameter file or CVaR alpha; move it away before re-running "
                         "(results from different settings must never be mixed).")
    if "methods" in prev and set(prev["methods"]) != {",".join(methods)}:
        raise SystemExit(f"{out} was run with methods {set(prev['methods'])}; this run asks for {','.join(methods)}. "
                         "Resume works per instance, so a different method list would silently skip methods. Use the same list.")
    done = set(prev["instance"])
# drop rows of an instance interrupted between writing its run rows and its instances.csv row
for path in (q_path, c_path):
    if path.exists():
        d = pd.read_csv(path)
        d[d["instance"].isin(done)].to_csv(path, index=False)
tuned_all = {}
t_all = time.time()
for si, sz in enumerate(sizes):
    s0 = spec["base_seed"] + spec["size_seed_stride"] * si
    if n_inst > spec["tuning_seed_offset"]:
        raise SystemExit("synthetic_instances_per_size must be < synthetic_suite.tuning_seed_offset (disjoint seeds)")
    insts = [synthetic_instance(sz, s0 + j, spec) for j in range(n_inst)]
    exacts = [solve_exact(i) for i in insts]
    n_tune = cfg["methods"]["tuning"]["n_tuning_instances"]
    tune_insts = [synthetic_instance(sz, s0 + spec["tuning_seed_offset"] + j, spec) for j in range(n_tune)]
    excl_sz = [e["method"] for e in spec.get("method_exclusions", []) if list(sz) in [list(z) for z in e["sizes"]]]
    to_tune = [m for m in ("M3", "M4") if m in methods and m not in excl_sz]
    if all(i.name in done for i in insts):
        print(f"\nSize {sz}: all instances already done (resume); tuning not repeated")
        tuned_all[str(sz)] = f"see penalty_tuning_{'x'.join(map(str, sz))}.csv"
        continue
    print(f"\nSize {sz}: tuning {to_tune} on {n_tune} held-out instance(s)")
    tuned, tab = tune_penalties([(ti, solve_exact(ti)) for ti in tune_insts], cfg,
                                [cfg["methods"]["tuning"]["seed_offset"] + s for s in range(prof["tune_seeds"])],
                                prof["tune_p"], prof["maxiter"], shots, methods=to_tune)
    tab.assign(size=str(sz)).to_csv(out / f"penalty_tuning_{'x'.join(map(str, sz))}.csv", index=False)
    tuned_all[str(sz)] = tuned
    for inst, ex in zip(insts, exacts):
        if inst.name in done:
            print(f"  {inst.name}: already done, skipped (resume)")
            continue
        t0 = time.time()
        print(f"  {inst.name}: k={inst.k.tolist()}, feasible={ex.n_feasible}, optima={len(ex.opt_set)}")
        run_m = [m for m in methods if m not in excl_sz]
        dq, skipped = run_quantum(inst, ex, cfg, run_m, seeds, prof["p_max"], prof["maxiter"], shots, tuned)
        skipped += [{"method": m, "reason": "excluded by design (synthetic_suite.method_exclusions)"} for m in excl_sz if m in methods]
        dc = run_classical(inst, ex, cfg, [cfg["classical"]["seeds"]["start"] + s for s in seeds])
        dq["size"] = str(sz); dc["size"] = str(sz)
        dq.to_csv(q_path, mode="a", header=not q_path.exists(), index=False)
        dc.to_csv(c_path, mode="a", header=not c_path.exists(), index=False)
        pd.DataFrame([{"instance": inst.name, "params_sha256": p.sha256, "cvar_alpha": alpha, "methods": ",".join(methods), "size": str(sz), "k": inst.k.tolist(), "n_feasible": ex.n_feasible,
                       "n_optima": len(ex.opt_set), "f_opt": ex.f_opt, "skipped": skipped,
                       "seconds": time.time() - t0}]).to_csv(i_path, mode="a", header=not i_path.exists(), index=False)

# ------------------------------------------------------------------ analysis
dq = pd.read_csv(q_path)
dc = pd.read_csv(c_path)
agg = aggregate(dq)
agg.to_csv(out / "summary_per_instance.csv", index=False)
per_inst = dq.groupby(["size", "instance", "method", "p"]).agg(p_opt=("p_opt", "median"), p_feas=("p_feasible", "median"),
                                                              ar=("ar_best_sampled", "median"), uni=("uniform_p_opt", "first"),
                                                              n_params=("n_params", "first"), n_qubits=("n_qubits", "first"),
                                                              nfev=("nfev", "median"),
                                                              cvar_saturated=("cvar_saturated", "mean")).reset_index()
overall = per_inst.groupby(["size", "method", "p"]).agg(instances=("instance", "nunique"), p_opt_median=("p_opt", "median"),
                                                        p_feasible_median=("p_feas", "median"), ar_median=("ar", "median"),
                                                        n_params=("n_params", "median"), n_qubits_max=("n_qubits", "max"),
                                                        nfev_median=("nfev", "median"),
                                                        cvar_saturated_frac=("cvar_saturated", "mean")).reset_index()
overall.to_csv(out / "summary_overall.csv", index=False)
print("\n", overall.to_string(index=False))

HM = cfg["methods"]["headline"]
tests = []
for (sz, pp), g in per_inst.groupby(["size", "p"]):
    piv = g.pivot(index="instance", columns="method", values="p_opt")
    if HM not in piv:
        continue
    for other in [m for m in piv.columns if m != HM]:
        r = wilcoxon_paired(piv[HM], piv[other])
        tests.append({"size": sz, "depth": pp, "comparison": f"{HM} vs {other}", **r})
tests = pd.DataFrame(tests)
if not tests.empty:
    tests["p_holm"] = holm(tests["p_value"].values)
tests.to_csv(out / "paired_tests_p_opt.csv", index=False)
print("\nPaired tests on per-instance median P(optimum) (Holm-corrected over all rows):")
print(tests.to_string(index=False))

cls = dc.groupby(["size", "method"]).agg(feasible=("feasible", "mean"), hit_opt=("hit_opt", "mean"), ar=("ar", "median")).reset_index()
cls.to_csv(out / "summary_classical.csv", index=False)

# figure: median P(optimum) vs p, one panel per size (log scale), uniform baseline for structural methods
fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.6), squeeze=False)
for ax, sz in zip(axes[0], [str(s) for s in sizes]):
    g = overall[overall["size"] == sz]
    for m in methods:
        h = g[g["method"] == m]
        if len(h):
            ax.plot(h["p"], h["p_opt_median"].clip(lower=1e-6), color=COLOR[m], marker=MARKER[m], linewidth=2, markersize=8, label=LABEL[m])
    uni = per_inst[(per_inst["size"] == sz) & (per_inst["method"] == HM)]["uni"].median()
    if not np.isnan(uni):
        ax.axhline(uni, color="#7a7a7a", linestyle="--", linewidth=1, label=f"uniform over {HM} reachable states")
    ax.set_yscale("log")
    style(ax, f"Sizes {sz}", "QAOA depth p", "median P(optimum)")
axes[0][0].legend(fontsize=7, loc="lower left")
save(fig, out / "fig_p_opt_vs_depth.png")

write_json(out / "summary.json", {"environment": environment(), "params_sha256": p.sha256, "profile": args.profile, "cvar_alpha": alpha,
                                  "sizes": sizes, "instances_per_size": n_inst, "seeds": len(seeds), "tuned_penalties": tuned_all,
                                  "wall_seconds_this_run": time.time() - t_all,
                                  "note": "cvar_saturated = training CVaR <= f* + tolerance; above that point P(optimum) is not pushed by the objective (for M3/M4 the penalised training energy of infeasible states can lie below f*, so there the flag also signals that infeasible states are rewarded). Synthetic instances (declared). Noiseless statevector simulation. No claim against classical solvers."})
print(f"\nDone -> {out}")
