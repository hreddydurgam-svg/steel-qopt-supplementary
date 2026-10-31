"""
Section 3.2: trainability diagnostics.

(a) Gradient variance across random initialisations, problem sizes and depths, for uniform and near-zero
    initialisation (finite differences on last-layer mixer angles; diagnostics only, training uses COBYLA).
(b) Dynamical Lie algebra dimension of each method's generators on small cases (Ragone et al. 2024;
    Kordonowy & Leipold 2026).
(c) Depth sweep with optimisation trajectories.

Outputs: gradient_variance.csv (+ figure), dla_dimensions.csv, depth_sweep.csv (+ figure), trajectories.json.
Usage:  python run_3_2_trainability.py --profile quick|paper
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.experiment import run_quantum
from qopt.instance import build_from_raw, solve_exact, synthetic_instance
from qopt.methods import make_method
from qopt.plotting import COLOR, LABEL, MARKER, save, style
from qopt.trainability import dla_dimension, generators_for, grad_variance
from qopt.train import method_fingerprint, run_chain

import matplotlib.pyplot as plt

args = parse_profile()
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
tr = cfg["trainability"]
spec = cfg["synthetic_suite"]
out = results_dir(f"3_2_trainability_{args.profile}")
banner(f"3.2 TRAINABILITY  (profile {args.profile})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()

# ------------------------------------------------------------------ (a) gradient variance
gv = tr["gradient_variance"]
rows = []
for si, sz in enumerate(gv["sizes"]):
    inst = synthetic_instance(sz, gv["instance_seed_base"] + si, spec)
    ex = solve_exact(inst)
    for m in gv["methods"]:
        try:
            meth = make_method(m, inst, ex, cfg, {"M3": gv["penalty_params"]["M3"]})
        except MemoryError as e:
            print(f"  {sz} {m}: skipped ({e})")
            continue
        for pp in gv["p_list"]:
            for kind in gv["inits"]:
                t0 = time.time()
                gkey = fingerprint(method_fingerprint(meth), pp, prof["grad_inits"], gv["inits"][kind],
                                   gv["init_seed_stride"] * si + pp, gv["fd_step"], prof["grad_k_max"])
                r = cached("gradvar", gkey, lambda: grad_variance(meth, pp, prof["grad_inits"], gv["inits"][kind],
                                                                  seed=gv["init_seed_stride"] * si + pp,
                                                                  h=gv["fd_step"], k_max=prof["grad_k_max"]))
                rows.append({"sizes": str(sz), "N_items": inst.N, "n_qubits": meth.n_qubits, "method": m, "p": pp,
                             "init": kind, "n_params": meth.n_params(pp), **r, "seconds": time.time() - t0})
                print(f"  {sz} {m} p={pp} {kind}: mean Var[dE/dbeta, last-layer mixer] = {r['var']:.3e}  (N={inst.N} items, {meth.n_qubits} qubits)")
gvdf = pd.DataFrame(rows)
gvdf.to_csv(out / "gradient_variance.csv", index=False)

fig, axes = plt.subplots(1, len(gv["inits"]), figsize=(5 * len(gv["inits"]), 3.8), squeeze=False)
for ax, kind in zip(axes[0], gv["inits"]):
    for m in gv["methods"]:
        for pp, ls in zip(gv["p_list"], ["-", "--", ":"]):
            h = gvdf[(gvdf["method"] == m) & (gvdf["p"] == pp) & (gvdf["init"] == kind)]
            if len(h):
                h = h.sort_values("N_items")      # x = number of items (same for every method); qubit counts differ by method and are in the CSV
                ax.plot(h["N_items"], h["var"], color=COLOR[m], marker=MARKER[m], linestyle=ls, linewidth=2,
                        markersize=8, label=f"{m}, p={pp}")
    ax.set_yscale("log")
    ax.set_xticks(sorted(gvdf["N_items"].unique()))
    style(ax, f"Gradient variance, init {kind}", "items N (one synthetic instance per size)",
          "mean Var[∂Ê/∂β], last-layer mixer angles")
axes[0][0].legend(fontsize=7, ncol=2)
save(fig, out / "fig_gradient_variance.png")

# ------------------------------------------------------------------ (b) DLA dimension (small cases)
dl = tr["dla"]
drow = []
for case in dl["cases"]:
    rng = np.random.default_rng(dl["seed"])
    N = sum(case)
    item_group = np.concatenate([[g] * n for g, n in enumerate(case)])
    Q = np.zeros((N, N)); s = 0
    for n in case:
        B = rng.normal(0, 1, (n, n)); Q[s:s + n, s:s + n] = B @ B.T / n; s += n
    inst = build_from_raw(f"dla{case}", [f"g{g}" for g in range(len(case))], item_group,
                          [n - dl["k_offset_below_n"] for n in case], rng.uniform(*dl["cost_emis_range"], N),
                          rng.uniform(*dl["cost_emis_range"], N), Q, spec["weights"])
    ex = solve_exact(inst)
    for m in dl["methods"]:
        try:
            meth = make_method(m, inst, ex, cfg, {"M4": dl["M4_penalty"]})
            gens, D = generators_for(meth)
        except (MemoryError, ValueError) as e:
            drow.append({"case": str(case), "method": m, "space_dim": None, "dla_dim": None, "note": str(e)}); continue
        if D > dl["max_space_dim"]:
            drow.append({"case": str(case), "method": m, "space_dim": D, "dla_dim": None, "note": f"space dim {D} > cap"}); continue
        t0 = time.time()
        dim, capped = cached("dla", fingerprint(method_fingerprint(meth), str(case), len(gens), D),
                             lambda: list(dla_dimension(gens)))
        drow.append({"case": str(case), "method": m, "space_dim": D, "dla_dim": dim, "max_possible_u": D * D,
                     "fraction_of_u(D)": dim / (D * D), "n_generators": len(gens), "capped": capped, "seconds": time.time() - t0})
        print(f"  DLA {case} {m}: dim {dim} of u({D}) = {D * D}")
pd.DataFrame(drow).to_csv(out / "dla_dimensions.csv", index=False)

# ------------------------------------------------------------------ (c) depth sweep with trajectories
ds = tr["depth_sweep"]
inst = synthetic_instance(ds["instance"]["sizes"], ds["instance"]["seed"], spec)
ex = solve_exact(inst)
p_list = [x for x in ds["p_list"] if x <= prof["p_max"]] or [1]
traj = []
recs = []
for m in ds["methods"]:
    meth = make_method(m, inst, ex, cfg)
    for s in range(prof["seeds"]):
        for r in run_chain(meth, ex, s, max(p_list), prof["maxiter"], cfg["methods"]["common"]["cvar_alpha"],
                           cfg["methods"]["common"]["shots"], p_list):
            traj.append({"method": m, "seed": s, "p": r["p"], "history": r["history"]})
            recs.append({k: v for k, v in r.items() if k not in ("theta", "history")})
dsd = pd.DataFrame(recs)
dsd.to_csv(out / "depth_sweep.csv", index=False)
write_json(out / "trajectories.json", traj)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in ds["methods"]:
    h = dsd[dsd["method"] == m].groupby("p")["p_opt"].median()
    ax.plot(h.index, h.values, color=COLOR[m], marker=MARKER[m], linewidth=2, markersize=8, label=LABEL[m])
ax.axhline(1 / ex.n_feasible, color="#7a7a7a", linestyle="--", linewidth=1, label="1 / |feasible set|")
ax.set_yscale("log")
style(ax, f"Depth sweep, {inst.name}", "QAOA depth p", "median P(optimum)")
ax.legend(fontsize=7)
save(fig, out / "fig_depth_sweep.png")

write_json(out / "summary.json", {"environment": environment(), "params_sha256": p.sha256, "profile": args.profile,
                                  "wall_seconds": time.time() - t_all,
                                  "note": "Trajectories show optimisation behaviour only; gradient-variance statistics, not trajectories, are the trainability evidence."})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
