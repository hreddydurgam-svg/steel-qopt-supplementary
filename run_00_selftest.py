"""
run_00_selftest.py -- run first. Prints versions and hardware, checks the regression targets and
cross-checks every numerical engine against an independent dense calculation. Exit code 1 on any failure.
Optional checks (PuLP MILP, Qiskit circuits) are reported as SKIPPED if the package is not installed.
"""
import itertools
import sys

import numpy as np
from scipy.linalg import expm

from load_params_v18 import load
from qopt.common import environment, results_dir, write_json
from qopt.instance import solve_exact, synthetic_instance
from qopt.methods import MPSX, PenaltyQAOA, MAPC, interp, make_method
from qopt.noise import FullSpaceMPSX, trajectory_mpsx
from qopt.sim import FullSpace, Register, cvar

ok_all = True
report = {}


def check(name, cond, detail=""):
    global ok_all
    status = "PASS" if cond else "FAIL"
    ok_all &= bool(cond)
    report[name] = {"status": status, "detail": detail}
    print(f"[{status}] {name} {detail}")


def skip(name, why):
    report[name] = {"status": "SKIPPED", "detail": why}
    print(f"[SKIPPED] {name}: {why}")


env = environment()
print("Environment:", env)
p = load()
print("params sha256", p.sha256)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)

# 1. regression of the locked per-tonne values
tgt = p.raw["regression_targets"]
dev = max(max(abs(p.tlc[r] - tgt["tlc_usd_t"][r]), abs(p.emis[r] - tgt["emissions_kg_t"][r])) for r in p.routes)
check("regression vs manuscript Tables 1-2", dev <= tgt["tolerance"], f"max dev {dev:.4f}")

# 2. XY gate against dense expm on 3 qubits (qubits 0 and 2)
fs = FullSpace(3)
rng = np.random.default_rng(1)
psi = rng.normal(size=8) + 1j * rng.normal(size=8); psi /= np.linalg.norm(psi)
b = 0.37
I2 = np.eye(2)
XX = np.kron(np.kron(X, I2), X); YY = np.kron(np.kron(Y, I2), Y)      # kron order: qubit2 (x) qubit1 (x) qubit0
U = expm(-1j * b * (XX + YY) / 2)
check("XY gate vs expm", np.allclose(fs.xy(psi.copy(), 0, 2, b), U @ psi), "")

# 3. X mixer against dense expm
HX = sum(np.kron(np.kron(*(X if q == k else I2 for k in (2, 1))), X if q == 0 else I2) for q in range(3))
check("RX mixer vs expm", np.allclose(fs.rx_all(psi.copy(), b), expm(-1j * b * HX) @ psi))

# 4. Pauli Y implementation
for q in range(3):
    ops = [I2, I2, I2]; ops[2 - q] = Y
    M = np.kron(np.kron(ops[0], ops[1]), ops[2])
    check(f"Pauli Y on qubit {q}", np.allclose(fs.pauli(psi.copy(), q, "Y"), M @ psi))

# 5. exact enumeration vs brute force on a synthetic instance
spec = p.raw["synthetic_suite"]
inst = synthetic_instance([3, 2, 3], 11, spec)
ex = solve_exact(inst)
best = min((inst.f(np.array(x))[0], x) for x in itertools.product([0, 1], repeat=inst.N) if inst.feasible(np.array(x))[0])
check("exact enumeration vs brute force", abs(best[0] - ex.f_opt) < 1e-12, f"f*={ex.f_opt:.6f}")

# 6. MPSX subspace engine vs full-space realisation (noiseless): M2 (single-angle) and M1 (multi-angle)
m_single = MPSX(inst, ex)
th = rng.uniform(0, 1, m_single.n_params(2))
xint, prob, _ = m_single.dist(th, 2)
fm = FullSpaceMPSX(m_single)
psi_full = trajectory_mpsx(fm, th, 2, 0.0, 0.0, rng)
full_marg = np.bincount(fm.xint, weights=np.abs(psi_full) ** 2, minlength=2 ** inst.N)
sub_marg = np.bincount(xint, weights=prob, minlength=2 ** inst.N)
err = np.abs(full_marg - sub_marg).max()
check("M2 subspace == full-space", err < 1e-10, f"max |diff| {err:.2e}")
check("M2 feasibility structural", ex.feas_all[xint][prob > 1e-14].all(), "")
m_multi = MPSX(inst, ex, multi_angle=True)
th_m1 = rng.uniform(0, 1, m_multi.n_params(2))
x_m1, p_m1, _ = m_multi.dist(th_m1, 2)
psi_m1 = trajectory_mpsx(FullSpaceMPSX(m_multi), th_m1, 2, 0.0, 0.0, rng)
err_m1 = np.abs(np.bincount(FullSpaceMPSX(m_multi).xint, weights=np.abs(psi_m1) ** 2, minlength=2 ** inst.N) -
              np.bincount(x_m1, weights=p_m1, minlength=2 ** inst.N)).max()
check("M1 subspace == full-space", err_m1 < 1e-10, f"max |diff| {err_m1:.2e}")
check("M1 feasibility structural", ex.feas_all[x_m1][p_m1 > 1e-14].all(), "")

# 7. MA-PC (M5) register engine: norm (the register basis has global weight N by construction)
m5 = MAPC(inst, ex)
ps5 = m5.state(rng.uniform(-1, 1, m5.n_params(1)), 1)
check("M5 norm", abs(np.linalg.norm(ps5) - 1) < 1e-10)

# 8. Penalty methods: norm; slack penalty zero exactly on feasible encodings
m3 = PenaltyQAOA(inst, ex, "slack", A=1.0)
ps3 = m3.state(rng.uniform(0, 1, 4), 2)
check("M3 norm", abs(np.linalg.norm(ps3) - 1) < 1e-10)
zero_pen = np.isclose(m3.e_train, ex.f_all[m3.xint], atol=1e-12)
check("M3 zero-penalty states are feasible", ex.feas_all[m3.xint][zero_pen].all())
check("M3 every feasible item set has a zero-penalty encoding",
      set(np.flatnonzero(ex.feas_all)) <= set(m3.xint[zero_pen].tolist()))
check("M3 min over zero-penalty states equals f*", np.isclose(m3.e_train[zero_pen].min(), ex.f_opt))

# 9. CVaR and INTERP
e = np.array([3.0, 1.0, 2.0]); pr = np.array([0.2, 0.3, 0.5])
check("CVaR alpha=1 equals mean", np.isclose(cvar(e, pr, 1.0), e @ pr))
check("CVaR alpha=0.3 equals min", np.isclose(cvar(e, pr, 0.3), 1.0))
check("INTERP p=1->2", np.allclose(interp(np.array([0.4, 0.2]), 1), [0.4, 0.2, 0.4, 0.2]))

# 10. optional: MILP (PuLP/CBC) against enumeration
try:
    from qopt.classical import milp
    x, dt, ver = milp(inst)
    check("MILP (McCormick) == enumeration", abs(inst.f(x)[0] - ex.f_opt) < 1e-7 and inst.feasible(x)[0], f"pulp {ver}")
except ImportError:
    skip("MILP (McCormick) == enumeration", "pulp not installed (pip install pulp)")

# 11. optional: Qiskit gate-level circuits (all five methods) against the numpy engine; Aer import check
try:
    import qiskit  # noqa: F401
    from qopt import qiskit_export as qe
    small = synthetic_instance([2, 2, 2], 5, spec)
    exs = solve_exact(small)
    fixed = {"M3": {"A": 0.5}, "M4": {"lambda1": 0.5, "lambda2": 1.0}}
    for mid in ("M1", "M2", "M3", "M4", "M5"):
        ms = make_method(mid, small, exs, p.raw, fixed)
        ths = rng.uniform(-1, 1, ms.n_params(2))
        pq = qe.qiskit_probabilities(mid, ms, small, ths, 2)
        q_marg = np.bincount(np.arange(len(pq)) & ((1 << small.N) - 1), weights=pq, minlength=2 ** small.N)
        xi, pr, _ = ms.dist(ths, 2)
        d = float(np.abs(q_marg - np.bincount(xi, weights=pr, minlength=2 ** small.N)).max())
        check(f"Qiskit {mid} circuit == numpy engine", d < 1e-8, f"max |diff| {d:.2e}, qiskit {qiskit.__version__}")
    print("   M1 layer resources (p=2):", qe.resource_counts(qe.layers_circuit("M1", make_method("M1", small, exs, p.raw), small,
                                                                           rng.uniform(0.1, 1, 2 * make_method("M1", small, exs, p.raw).per_layer), 2)))
    try:
        import qiskit_aer
        check("qiskit-aer importable", True, f"qiskit-aer {qiskit_aer.__version__}")
    except ImportError:
        skip("qiskit-aer importable", "qiskit-aer not installed (needed only for run_C_qiskit.py Part 3)")
except ImportError:
    skip("Qiskit circuits == numpy engine", "qiskit not installed (optional; needed for run_C_qiskit.py)")
except Exception as e:                                   # a Qiskit API/runtime error must be reported, not crash the test
    check("Qiskit circuits == numpy engine", False, f"{type(e).__name__}: {e}")

out = results_dir("00_selftest")
write_json(out / "selftest.json", {"environment": env, "params_sha256": p.sha256, "checks": report})
print("\nSELFTEST:", "PASS" if ok_all else "FAIL", "->", out / "selftest.json")
sys.exit(0 if ok_all else 1)
