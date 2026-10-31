"""
Qiskit companion run (needs `pip install "qiskit>=1.2,<3" "qiskit-aer>=0.15"`). Run AFTER the other scripts of the same profile.

Part 1  Cross-validation: trained angles saved by run_3_1_steel_case.py and run_B_synthetic_suite.py (both CVaR
        alpha) are rebuilt as gate-level Qiskit circuits (all five methods, ideal state preparation) and simulated
        with qiskit.quantum_info.Statevector. P(optimum) and P(feasible) must equal the numpy-engine values
        within qiskit.cross_validation.tolerance. Any mismatch is reported and the script exits with code 1.
Part 2  Circuit resources: transpiled depth, two-qubit (CX) and one-qubit gate counts per method, instance and
        depth p (layers only; state preparation stated separately).
Part 3  Qiskit Aer noise for Section 3.3: the circuits trained by run_3_3_noise.py are executed on AerSimulator
        with depolarising noise on the transpiled native gates and readout flips (rates from the 'noise' block).

`--alpha 1` writes to results/C_qiskit_<profile>_alpha1/: Part 1 on the alpha = 1 steel-case circuits, Part 3 on the
alpha = 1 Section 3.3 circuits; Part 2 is not repeated.

Usage:  python run_C_qiskit.py --profile quick|paper [--alpha 1]
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import apply_alpha, banner, cached, environment, fingerprint, parse_profile, profile, results_dir, set_checkpoint, write_json
from qopt.instance import solve_exact, synthetic_instance
from qopt.methods import make_method
from qopt import qiskit_export as qe
from qopt.steel import steel_case

try:
    import qiskit
except ImportError:
    sys.exit('qiskit is not installed: run  pip install "qiskit>=1.2,<3" "qiskit-aer>=0.15"  and re-run this script.')

args = parse_profile(alpha=True)
p = load()
cfg = p.raw
prof = profile(cfg, args.profile)
alpha, suffix = apply_alpha(cfg, args)
qk = cfg["qiskit"]
RES = results_dir(".")
out = results_dir(f"C_qiskit_{args.profile}{suffix}")
banner(f"C QISKIT CROSS-VALIDATION, RESOURCES AND AER NOISE  (profile {args.profile}, CVaR alpha {alpha:g})", p)
set_checkpoint(out, p.sha256)
t_all = time.time()
spec = cfg["synthetic_suite"]
tol_rel = 1e-9


def rebuild(source_inst, cache={}):
    """Rebuild an instance and its exact solution from the instance name stored in the results."""
    if source_inst in cache:
        return cache[source_inst]
    if source_inst == "steel":
        inst = steel_case(p)["inst"]
        if inst is None:
            raise ValueError("steel case has no certified plan at the configured epsilon")
    else:
        m = re.fullmatch(r"synth\(([\d, ]+)\)_s(\d+)", source_inst)
        if not m:
            raise ValueError(f"unknown instance name {source_inst}")
        inst = synthetic_instance([int(v) for v in m.group(1).split(",")], int(m.group(2)), spec)
    cache[source_inst] = (inst, solve_exact(inst))
    return cache[source_inst]


def tuned_from(params):
    if "lambda1" in params:
        return {"M4": {"lambda1": params["lambda1"], "lambda2": params["lambda2"]}}
    if "A" in params and "S" not in params:
        return {"M3": {"A": params["A"]}}
    return {}


def probs_to_metrics(prob_full, inst, ex):
    xint = np.arange(len(prob_full)) & ((1 << inst.N) - 1)
    f = ex.f_all[xint]
    feas = ex.feas_all[xint]
    opt = feas & (f <= ex.f_opt + tol_rel * max(1.0, abs(ex.f_opt)))
    return float(prob_full[opt].sum()), float(prob_full[feas].sum())


# ------------------------------------------------------------------ Part 1: cross-validation
cv = qk["cross_validation"]
rows = []
sources = cv["sources"] if not suffix else ["3_1_steel_case" + suffix]
for src in sources:
    folder = RES / (f"{src}_{args.profile}" if not src.endswith("_alpha1") else f"{src[:-7]}_{args.profile}_alpha1")
    qp = folder / "quantum_runs.csv"
    if not qp.exists() or not (folder / "summary.json").exists():
        print(f"  cross-validation: {folder.name} missing or incomplete (run its script to the end first) - skipped")
        continue
    sj = json.loads((folder / "summary.json").read_text())
    if sj.get("params_sha256") != p.sha256:
        sys.exit(f"STOP: {folder} was produced with a different parameter file.")
    d = pd.read_csv(qp)
    if "size" not in d:
        d["size"] = "steel"
    for (size, method, pp), g in d.groupby(["size", "method", "p"]):
        for _, r in g.sort_values(["instance", "seed"]).head(cv["runs_per_group"]).iterrows():
            inst, ex = rebuild(r["instance"])
            if r["n_qubits"] > cv["max_qubits"]:
                rows.append({"source": folder.name, "instance": r["instance"], "method": method, "p": pp, "seed": r["seed"],
                             "status": f"skipped: {r['n_qubits']} qubits > max_qubits"})
                continue
            theta = np.array(json.loads(r["theta_json"]))
            meth = make_method(method, inst, ex, cfg, tuned_from(json.loads(r["method_params_json"])))
            t0 = time.time()
            q_opt, q_feas = cached("xval", fingerprint(folder.name, r["instance"], method, int(pp), int(r["seed"]), theta,
                                                       r["method_params_json"]),
                                   lambda: list(probs_to_metrics(qe.qiskit_probabilities(method, meth, inst, theta, int(pp)), inst, ex)))
            d_opt, d_feas = abs(q_opt - r["p_opt"]), abs(q_feas - r["p_feasible"])
            ok = d_opt <= cv["tolerance"] and d_feas <= cv["tolerance"]
            rows.append({"source": folder.name, "instance": r["instance"], "method": method, "p": pp, "seed": r["seed"],
                         "n_qubits": int(r["n_qubits"]), "numpy_p_opt": r["p_opt"], "qiskit_p_opt": q_opt,
                         "numpy_p_feasible": r["p_feasible"], "qiskit_p_feasible": q_feas, "abs_diff_p_opt": d_opt,
                         "abs_diff_p_feasible": d_feas, "status": "PASS" if ok else "FAIL", "seconds": time.time() - t0})
    print(f"  cross-validation {folder.name}: {sum(1 for x in rows if x['source'] == folder.name and x['status'] == 'PASS')} PASS, "
          f"{sum(1 for x in rows if x['source'] == folder.name and x['status'] == 'FAIL')} FAIL")
cvd = pd.DataFrame(rows, columns=["source", "instance", "method", "p", "seed", "n_qubits", "numpy_p_opt", "qiskit_p_opt", "numpy_p_feasible", "qiskit_p_feasible", "abs_diff_p_opt", "abs_diff_p_feasible", "status", "seconds"])
cvd.to_csv(out / "cross_validation.csv", index=False)
n_fail = int((cvd["status"] == "FAIL").sum()) if len(cvd) else 0
ok = cvd[cvd["status"].isin(["PASS", "FAIL"])]
if len(ok):
    print(f"Cross-validation: {len(ok)} circuits, {n_fail} FAIL; max |diff| P(opt) {ok['abs_diff_p_opt'].max():.2e}, "
          f"P(feasible) {ok['abs_diff_p_feasible'].max():.2e}")
if n_fail:
    print(f"*** CROSS-VALIDATION FAILED for {n_fail} circuits (see cross_validation.csv); Parts 2-3 continue, exit code will be 1 ***")

# ------------------------------------------------------------------ Part 2: circuit resources
if suffix:
    print("  resources: not repeated for the alpha run")
else:
    rs = qk["resources"]
    rng = np.random.default_rng(rs["angle_seed"])
    p_list = list(range(1, prof["p_max"] + 1))
    cases = [("steel", "steel")]
    for si, sz in enumerate(prof["synthetic_sizes"]):
        s0 = spec["base_seed"] + spec["size_seed_stride"] * si
        cases += [(str(sz), f"synth{tuple(sz)}_s{s0 + j}") for j in range(min(rs["instances_per_size"], prof["synthetic_instances_per_size"]))]
    excl = {(e["method"], str(list(z))) for e in spec.get("method_exclusions", []) for z in e["sizes"]}
    res_rows = []
    for size, name in cases:
        try:
            inst, ex = rebuild(name)
        except ValueError as e:
            print(f"  resources {name}: skipped ({e})")
            continue
        for method in ["M1", "M2", "M3", "M4", "M5"]:
            if (method, size) in excl:
                continue
            try:
                meth = None if method == "M5" else make_method(method, inst, ex, cfg, rs["penalty_params"])
            except MemoryError as e:
                res_rows.append({"size": size, "instance": name, "method": method, "status": f"not built: {e}"})
                continue
            if method == "M5":
                S, pairs, chain, sl = qe.mapc_structure(inst)
                n_q, per_layer = inst.N + S, inst.N + len(pairs) + len(chain) + len(sl)
            else:
                n_q, per_layer = meth.n_qubits, meth.per_layer
            for pp in p_list:
                th = rng.uniform(*rs["angle_range"], per_layer * pp)
                qc = qe.layers_circuit(method, meth, inst, th, pp)
                rc = qe.resource_counts(qc, tuple(rs["basis_gates"]), rs["optimization_level"], rs["transpiler_seed"])
                res_rows.append({"size": size, "instance": name, "N_items": inst.N, "method": method, "p": pp, "n_qubits": n_q,
                                 "angles": per_layer * pp, "depth": rc["depth"], "two_qubit": rc["two_qubit"],
                                 "one_qubit": rc["one_qubit"], "ops": json.dumps(rc["ops"]),
                                 "state_preparation": {"M1": "Dicke product state (excluded)", "M2": "Dicke product state (excluded)",
                                                       "M3": "H on every qubit (excluded)", "M4": "H on every qubit (excluded)",
                                                       "M5": "X on the seed qubits (excluded)"}[method], "status": "ok"})
        print(f"  resources {name}: done")
    rd = pd.DataFrame(res_rows)
    rd.to_csv(out / "circuit_resources.csv", index=False)
    okr = rd[rd["status"] == "ok"]
    summ = okr.groupby(["size", "method", "p"]).agg(instances=("instance", "nunique"), n_qubits_median=("n_qubits", "median"),
                                                     depth_median=("depth", "median"), two_qubit_median=("two_qubit", "median"),
                                                     one_qubit_median=("one_qubit", "median"), angles_median=("angles", "median")).reset_index()
    summ.to_csv(out / "circuit_resources_summary.csv", index=False)
    print(summ[summ["p"] == 1].to_string(index=False))

# ------------------------------------------------------------------ Part 3: Aer noise (Section 3.3 circuits)
aer_rows, aer_check = [], []
tc_path = RES / f"3_3_noise_{args.profile}{suffix}" / "trained_circuits.json"
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    have_aer = True
except ImportError:
    have_aer = False
    print('  Aer noise: qiskit-aer not installed - Part 3 skipped (pip install "qiskit-aer>=0.15")')
if have_aer and not (tc_path.exists() and (tc_path.parent / "summary.json").exists()):
    print(f"  Aer noise: {tc_path} not found (run run_3_3_noise.py first) - Part 3 skipped")
if have_aer and tc_path.exists() and (tc_path.parent / "summary.json").exists():
    sj = json.loads((tc_path.parent / "summary.json").read_text())
    if sj.get("params_sha256") != p.sha256:
        sys.exit(f"STOP: {tc_path.parent} was produced with a different parameter file.")
    nz, an = cfg["noise"], qk["aer_noise"]
    tc = json.loads(tc_path.read_text())
    inst = synthetic_instance(tc["instance"]["sizes"], tc["instance"]["seed"], spec)
    ex = solve_exact(inst)
    shots = prof["aer_shots"]
    lam = lambda pr, d: pr * d * d / (d * d - 1)
    for c in tc["circuits"]:
        m, P = c["method"], c["p"]
        meth = make_method(m, inst, ex, cfg, tuned_from(c["method_params"]))
        n = meth.n_qubits if m != "M5" else qe.mapc_seed_state(inst)[1]
        layers = qe.transpile_layers(qe.layers_circuit(m, meth, inst, np.array(c["theta"]), P), tuple(qk["resources"]["basis_gates"]),
                                     qk["resources"]["optimization_level"], qk["resources"]["transpiler_seed"])
        circ = QuantumCircuit(n)
        circ.initialize(qe.initial_state(m, meth, inst), list(range(n)))
        circ.compose(layers, inplace=True)
        circ.measure_all()
        logical = qe.layers_circuit(m, meth, inst, np.array(c["theta"]), P)
        n_logical_2q = int(sum(v for k, v in dict(logical.count_ops()).items() if k in ("rzz", "rxx", "ryy")))
        # exact (shot-free) check that the transpiled circuit sent to Aer reproduces the numpy noiseless P(optimum)
        from qiskit.quantum_info import Statevector
        pv = np.abs(np.asarray(Statevector(qe.initial_state(m, meth, inst)).evolve(layers).data)) ** 2
        exact_t_opt = probs_to_metrics(pv, inst, ex)[0]
        d_t = abs(exact_t_opt - c["noiseless_p_opt"])
        aer_check.append({"method": m, "transpiled_exact_p_opt": exact_t_opt, "numpy_noiseless_p_opt": c["noiseless_p_opt"],
                          "abs_diff": d_t, "status": "PASS" if d_t <= cv["tolerance"] else "FAIL"})
        print(f"  Aer circuit check {m} (alpha {alpha:g}): transpiled circuit exact P(opt) {exact_t_opt:.6f} vs numpy {c['noiseless_p_opt']:.6f} "
              f"-> {aer_check[-1]['status']}  (shot-sampled values below differ by sampling error, SE ~ "
              f"{np.sqrt(exact_t_opt * (1 - exact_t_opt) / shots):.4f} at {shots} shots)")
        weight_ok = None
        if m in ("M1", "M2"):
            from qopt.noise import FullSpaceMPSX
            weight_ok = FullSpaceMPSX(meth).weight_ok
        for p2 in nz["p2_depolarising_sweep"]:
            p1 = nz["p1_depolarising"] if p2 > 0 else 0.0
            for ro in ([0.0, nz["readout_flip"]] if p2 > 0 else [0.0]):
                nm = NoiseModel()
                if p1 > 0:
                    nm.add_all_qubit_quantum_error(depolarizing_error(lam(p1, 2), 1), an["one_qubit_gates"])
                if p2 > 0:
                    nm.add_all_qubit_quantum_error(depolarizing_error(lam(p2, 4), 2), an["two_qubit_gates"])
                if ro > 0:
                    nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
                sim = AerSimulator(method=an["method"], noise_model=nm if (p1 > 0 or p2 > 0 or ro > 0) else None,
                                   seed_simulator=an["simulator_seed"])
                t0 = time.time()
                counts = cached("aer", fingerprint(m, P, c["theta"], p1, p2, ro, shots, an, qiskit.__version__),
                                lambda: dict(sim.run(circ, shots=shots).result().get_counts()))
                keys = np.array([int(k.replace(" ", ""), 2) for k in counts])
                cnt = np.array(list(counts.values()), float)
                xi = keys & ((1 << inst.N) - 1)
                feas = ex.feas_all[xi]
                hit = feas & (ex.f_all[xi] <= ex.f_opt + tol_rel * max(1.0, abs(ex.f_opt)))
                rec = {"method": m, "cvar_alpha": alpha, "p": P, "p1": p1, "p2": p2, "readout": ro, "shots": shots,
                       "frac_feasible": float(cnt[feas].sum() / cnt.sum()), "frac_optimal": float(cnt[hit].sum() / cnt.sum()),
                       "noiseless_p_opt_numpy": c["noiseless_p_opt"], "two_qubit_gates": int(dict(layers.count_ops()).get("cx", 0)),
                       "logical_two_qubit_ops": n_logical_2q,
                       "seconds": time.time() - t0}
                if weight_ok is not None:
                    keep = weight_ok[keys]
                    kept = cnt[keep].sum()
                    rec.update({"postselect_kept": float(kept / cnt.sum()),
                                "postselect_frac_feasible": float(cnt[keep & feas].sum() / kept) if kept > 0 else np.nan,
                                "postselect_frac_optimal": float(cnt[keep & hit].sum() / kept) if kept > 0 else np.nan})
                aer_rows.append(rec)
                print(f"  Aer {m} p2={p2:.3f} readout={ro:.2f}: feasible {rec['frac_feasible']:.3f}, optimal {rec['frac_optimal']:.4f}"
                      + (f", post-selected kept {rec['postselect_kept']:.3f} -> optimal {rec['postselect_frac_optimal']:.4f}"
                         if weight_ok is not None else ""))
pd.DataFrame(aer_check, columns=["method", "transpiled_exact_p_opt", "numpy_noiseless_p_opt", "abs_diff", "status"]).to_csv(
    out / "aer_circuit_check.csv", index=False)
n_fail += sum(1 for r in aer_check if r["status"] == "FAIL")
pd.DataFrame(aer_rows, columns=["method", "cvar_alpha", "p", "p1", "p2", "readout", "shots", "frac_feasible", "frac_optimal",
                                 "noiseless_p_opt_numpy", "two_qubit_gates", "logical_two_qubit_ops", "seconds", "postselect_kept",
                                 "postselect_frac_feasible", "postselect_frac_optimal"]).to_csv(out / "aer_noise_sweep.csv", index=False)

vers = {"qiskit": qiskit.__version__}
try:
    import qiskit_aer
    vers["qiskit_aer"] = qiskit_aer.__version__
except ImportError:
    vers["qiskit_aer"] = "not installed"
write_json(out / "summary.json", {"environment": environment(), "qiskit_versions": vers, "params_sha256": p.sha256,
                                  "profile": args.profile, "cvar_alpha": alpha, "cross_validation_circuits": int(len(cvd)),
                                  "cross_validation_failures": n_fail, "aer_points": len(aer_rows),
                                  "wall_seconds": time.time() - t_all,
                                  "note": ("Resources: transpiled layers only; state preparation excluded and stated per method. "
                                           "Aer: depolarising parameter converted so that p is the Pauli-error probability; the rate applies per "
                                           "native CX (and p1 per sx/x), whereas the trajectory model applies it per logical two-qubit "
                                           "operation, so Aer accumulates more error per logical gate; ideal state preparation; "
                                           "rz noiseless (virtual).")})
print(f"\nDone in {time.time() - t_all:.0f}s -> {out}")
if n_fail:
    sys.exit(f"CROSS-VALIDATION FAILED for {n_fail} circuits; see cross_validation.csv and aer_circuit_check.csv")
