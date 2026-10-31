"""
Section 3.3: simulated noise sensitivity (a noisy-simulator robustness test, not hardware validation;
error rates are a sensitivity scenario, not a hardware calibration).

Each method is trained noiselessly on one synthetic instance, then evaluated under Pauli-trajectory depolarising
noise (sweep of the two-qubit rate) and readout flips. For the MPSX methods (M1, M2), symmetry verification
(post-selection on the conserved register Hamming weights) is reported separately from the raw output.
`--alpha 1` trains on the expectation and writes to results/3_3_noise_<profile>_alpha1/.
Usage:  python run_3_3_noise.py --profile quick|paper [--alpha 1]
"""
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.instance import solve_exact, synthetic_instance
from qopt.methods import make_method
from qopt.noise import MPSX_IDS, FullSpaceMPSX, noisy_distribution, sample_with_readout
from qopt.plotting import COLOR, LABEL, MARKER, save, style
from qopt.train import method_fingerprint, run_chain

import matplotlib.pyplot as plt

args = parse_profile(alpha=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
alpha, suffix = apply_alpha(cfg, args)
nz = cfg["noise"]
out = results_dir(f"3_3_noise_{args.profile}{suffix}")
banner(f"3.3 SIMULATED NOISE SENSITIVITY  (profile {args.profile}, CVaR alpha {alpha:g})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()

inst = synthetic_instance(nz["instance"]["sizes"], nz["instance"]["seed"], cfg["synthetic_suite"])
ex = solve_exact(inst)
P = prof["noise_p"]
shots = cfg["methods"]["common"]["shots"]
tol = 1e-9 * max(1.0, abs(ex.f_opt))
rows, trained = [], []
for m in nz["methods"]:
    meth = make_method(m, inst, ex, cfg, {k: v for k, v in nz["penalty_params"].items() if k != "note"})
    chains = [run_chain(meth, ex, s, P, prof["maxiter"], cfg["methods"]["common"]["cvar_alpha"], shots, [P])[-1]
              for s in range(prof["seeds"])]
    best = max(chains, key=lambda r: r["p_opt"])                     # noiseless-best seed is the one evaluated
    theta = np.array(best["theta"])
    trained.append({"method": m, "p": P, "seed": int(best["seed"]) if "seed" in best else None, "theta": theta.tolist(),
                    "method_params": {k: float(v) for k, v in getattr(meth, "params", {}).items()},
                    "noiseless_p_opt": best["p_opt"], "noiseless_p_feasible": best["p_feasible"]})
    obj = FullSpaceMPSX(meth) if m in MPSX_IDS else meth
    n_q = obj.n if m in MPSX_IDS else meth.n_qubits
    xmask = (1 << inst.N) - 1
    for p2 in nz["p2_depolarising_sweep"]:
        p1 = nz["p1_depolarising"] if p2 > 0 else 0.0
        t0 = time.time()
        n_traj = prof["noise_trajectories"] if p2 > 0 else 1
        prob = np.array(cached("noisy", fingerprint(method_fingerprint(meth), theta, P, p1, p2, n_traj, nz["trajectory_seed"]),
                               lambda: noisy_distribution(m, obj, theta, P, p1, p2, n_traj, seed=nz["trajectory_seed"]).tolist()))
        for ro in ([0.0, nz["readout_flip"]] if p2 > 0 else [0.0]):
            draws = sample_with_readout(prob, n_q, shots, ro, seed=nz["readout_seed"])
            xi = draws & xmask
            feas = ex.feas_all[xi]
            hit = feas & (ex.f_all[xi] <= ex.f_opt + tol)
            rec = {"method": m, "p": P, "p1": p1, "p2": p2, "readout": ro, "trajectories": prof["noise_trajectories"] if p2 > 0 else 1,
                   "shots": shots, "frac_feasible": float(feas.mean()), "frac_optimal": float(hit.mean()),
                   "noiseless_p_opt": best["p_opt"], "cvar_alpha": alpha, "seconds": time.time() - t0}
            if m in MPSX_IDS:
                keep = obj.weight_ok[draws]
                rec.update({"postselect_kept": float(keep.mean()),
                            "postselect_frac_feasible": float(feas[keep].mean()) if keep.any() else np.nan,
                            "postselect_frac_optimal": float(hit[keep].mean()) if keep.any() else np.nan})
            rows.append(rec)
            print(f"  {m} p2={p2:.3f} readout={ro:.2f}: feasible {rec['frac_feasible']:.3f}, optimal {rec['frac_optimal']:.4f}"
                  + (f", post-selected kept {rec['postselect_kept']:.3f} -> optimal {rec['postselect_frac_optimal']:.4f}" if m in MPSX_IDS else ""))
df = pd.DataFrame(rows)
df.to_csv(out / "noise_sweep.csv", index=False)
write_json(out / "trained_circuits.json", {"instance": nz["instance"], "cvar_alpha": alpha, "circuits": trained})

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
for m in nz["methods"]:
    h = df[(df["method"] == m) & ((df["readout"] > 0) | (df["p2"] == 0))]
    axes[0].plot(h["p2"], h["frac_feasible"], color=COLOR[m], marker=MARKER[m], linewidth=2, markersize=8, label=LABEL[m])
    axes[1].plot(h["p2"], h["frac_optimal"], color=COLOR[m], marker=MARKER[m], linewidth=2, markersize=8, label=LABEL[m])
    if m in MPSX_IDS:
        axes[1].plot(h["p2"], h["postselect_frac_optimal"], color=COLOR[m], marker=MARKER[m], linestyle="--", linewidth=2,
                     markersize=8, label=f"{m} after symmetry post-selection")
style(axes[0], f"Feasible shots (simulated noise + {nz['readout_flip']:.0%} readout)", "two-qubit depolarising rate", "fraction of shots")
style(axes[1], f"Optimal shots (training CVaR alpha {alpha:g})", "two-qubit depolarising rate", "fraction of shots")
axes[1].legend(fontsize=7)
save(fig, out / "fig_noise_sweep.png")
write_json(out / "summary.json", {"environment": environment(), "params_sha256": p.sha256, "profile": args.profile,
                                  "cvar_alpha": alpha, "instance": inst.name, "wall_seconds": time.time() - t_all,
                                  "note": "Simulated Pauli-trajectory noise; ideal state preparation assumed; not hardware validation."})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
