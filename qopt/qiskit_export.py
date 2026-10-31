"""
Qiskit circuits for all five methods: cross-validation against the numpy engine, circuit-resource counts and
Qiskit Aer noise simulation. Requires `pip install qiskit` (and `qiskit-aer` for noise).

Qiskit conventions used: RZ(t) = exp(-i t Z/2); RZZ/RXX/RYY(t) = exp(-i t P(x)P/2); RX(t) = exp(-i t X/2);
little-endian qubit order (qubit q is bit q of the basis-state index), Z|b> = (-1)^b |b>.

Circuit = state preparation + p layers. State preparation is returned separately as a statevector (prepared
ideally, as in the numpy engine): Dicke product state (M1, M2), uniform superposition (M3, M4), feasible basis
state (M5). Layer gates:
  M1/M2  phase: RZ/RZZ from the Ising form of f; mixer: RXX(beta) RYY(beta) on each register ring edge
  M3/M4  phase: RZ/RZZ from the Ising form of the penalised training energy (Walsh-Hadamard transform, exact);
         mixer: RX(2 beta) on every qubit
  M5     phase: RZ(t_i) per item, RZZ(t_ij) per coupled item pair; mixer: RXX/RYY chain + slack-item edges
"""
import numpy as np

BASIS = ("cx", "rz", "sx", "x")


def qubo_to_ising(lin, W):
    """f(x) = lin.x + x^T W x, x = (1 - z)/2  ->  f = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    N = len(lin)
    h = np.zeros(N)
    J = {}
    for i in range(N):
        h[i] += -0.5 * (lin[i] + W[i, i])
        for j in range(i + 1, N):
            w = 2.0 * W[i, j]
            if w != 0.0:
                h[i] += -0.25 * w
                h[j] += -0.25 * w
                J[(i, j)] = 0.25 * w
    return h, J


def ising_from_diagonal(e, n, tol=1e-12):
    """Exact Walsh-Hadamard expansion of a diagonal energy e[x] over n qubits: e = sum_S c_S prod_{i in S} Z_i.
    Returns (h, J) for |S| = 1, 2; raises if a higher-order term is non-zero (the energies used are quadratic)."""
    c = np.asarray(e, float).copy()
    hlen = 1
    while hlen < len(c):
        c = c.reshape(-1, 2, hlen)
        a, b = c[:, 0, :].copy(), c[:, 1, :].copy()
        c[:, 0, :], c[:, 1, :] = a + b, a - b
        c = c.reshape(-1)
        hlen *= 2
    c /= len(c)
    masks = np.arange(len(c))
    pop = np.array([bin(m).count("1") for m in masks])
    scale = max(1.0, float(np.abs(c).max()))
    if np.any(np.abs(c[pop > 2]) > tol * scale):
        raise ValueError("energy has terms of order > 2; cannot be written with RZ/RZZ only")
    h = np.array([c[1 << i] for i in range(n)])
    h[np.abs(h) <= tol * scale] = 0.0
    J = {}
    for m in masks[(pop == 2) & (np.abs(c) > tol * scale)]:
        i, j = [q for q in range(n) if (m >> q) & 1]
        J[(i, j)] = float(c[m])
    return h, J


def _mpsx_qubits(mpsx):
    glob, nxt = [], mpsx.inst.N
    for reg in mpsx.regs:
        gl = []
        for it in reg.item_of_local:
            gl.append(int(it) if it >= 0 else nxt)
            if it < 0:
                nxt += 1
        glob.append(gl)
    return glob


def mpsx_layers_circuit(mpsx, theta, p):
    """M1/M2 phase + mixer layers on N + slack qubits (no state preparation)."""
    from qiskit import QuantumCircuit
    inst = mpsx.inst
    h, J = qubo_to_ising(inst.lin, inst.W)
    glob = _mpsx_qubits(mpsx)
    qc = QuantumCircuit(mpsx.n_qubits)
    th = np.asarray(theta).reshape(p, mpsx.per_layer)
    for l in range(p):
        g = th[l, 0] / mpsx.span
        for i in range(inst.N):
            if h[i] != 0.0:
                qc.rz(2 * g * h[i], i)            # exp(-i g h Z) = RZ(2 g h)
        for (i, j), v in J.items():
            qc.rzz(2 * g * v, i, j)
        for r, (edges, gl) in enumerate(zip(mpsx.reg_edges, glob)):
            off = sum(len(E) for E in mpsx.reg_edges[:r])
            for t, (a, b) in enumerate(edges):
                beta = th[l, 1 + off + t] if mpsx.multi else th[l, 1 + r]
                qc.rxx(beta, gl[a], gl[b])
                qc.ryy(beta, gl[a], gl[b])
    return qc


def penalty_layers_circuit(meth, theta, p):
    """M3/M4 layers: exact Ising form of the penalised training energy, then RX(2 beta) on every qubit."""
    from qiskit import QuantumCircuit
    n = meth.n_qubits
    h, J = ising_from_diagonal(meth.e_train / meth.span, n)
    qc = QuantumCircuit(n)
    th = np.asarray(theta).reshape(p, 2)
    for l in range(p):
        g = th[l, 0]
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2 * g * h[i], i)
        for (i, j), v in J.items():
            qc.rzz(2 * g * v, i, j)
        for q in range(n):
            qc.rx(2 * th[l, 1], q)                # numpy engine: exp(-i beta X) = RX(2 beta)
    return qc


def mapc_structure(inst):
    """M5 gate structure (same order as methods.MAPC): pairs, chain, slack edges, S."""
    N = inst.N
    S = N - int(inst.k.sum())
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if inst.W[i, j] != 0.0]
    chain = [(i, i + 1) for i in range(N - 1)]
    slack_edges = [(N + s, r) for s in range(S) for r in range(N)]
    return S, pairs, chain, slack_edges


def mapc_layers_circuit(inst, theta, p):
    """M5 layers built from the instance alone (no subspace simulation needed, so usable above the memory cap)."""
    from qiskit import QuantumCircuit
    N = inst.N
    S, pairs, chain, slack_edges = mapc_structure(inst)
    per_layer = N + len(pairs) + len(chain) + len(slack_edges)
    th = np.asarray(theta).reshape(p, per_layer)
    qc = QuantumCircuit(N + S)
    for l in range(p):
        t = th[l]
        a = 0
        for i in range(N):
            qc.rz(t[a], i); a += 1                # exp(-i t z_i / 2)
        for (i, j) in pairs:
            qc.rzz(t[a], i, j); a += 1            # exp(-i t z_i z_j / 2)
        for (i, j) in chain:
            qc.rxx(t[a], i, j); qc.ryy(t[a], i, j); a += 1
        for (sq, r) in slack_edges:
            qc.rxx(t[a], sq, r); qc.ryy(t[a], sq, r); a += 1
    return qc


def mapc_seed_state(inst):
    N = inst.N
    S = N - int(inst.k.sum())
    seed = 0
    for g in range(inst.G):
        for i in inst.items_of(g)[: inst.k[g]]:
            seed |= 1 << int(i)
    for s in range(S):
        seed |= 1 << (N + s)
    return seed, N + S


def initial_state(name, meth=None, inst=None):
    """Ideal initial statevector (full 2^n space) for each method."""
    if name in ("M1", "M2"):
        from .noise import FullSpaceMPSX
        return FullSpaceMPSX(meth).initial()
    if name in ("M3", "M4"):
        d = 2 ** meth.n_qubits
        return np.full(d, 1 / np.sqrt(d), dtype=complex)
    if name == "M5":
        seed, n = mapc_seed_state(inst)
        v = np.zeros(2 ** n, dtype=complex)
        v[seed] = 1.0
        return v
    raise ValueError(name)


def layers_circuit(name, meth, inst, theta, p):
    if name in ("M1", "M2"):
        return mpsx_layers_circuit(meth, theta, p)
    if name in ("M3", "M4"):
        return penalty_layers_circuit(meth, theta, p)
    if name == "M5":
        return mapc_layers_circuit(inst, theta, p)
    raise ValueError(name)


def qiskit_probabilities(name, meth, inst, theta, p):
    """Noiseless Qiskit statevector probabilities over the full register (little-endian index)."""
    from qiskit.quantum_info import Statevector
    sv = Statevector(initial_state(name, meth, inst)).evolve(layers_circuit(name, meth, inst, theta, p))
    return np.abs(np.asarray(sv.data)) ** 2


def transpile_layers(qc, basis=BASIS, optimization_level=1, seed=0):
    """Transpile without a coupling map; levels > 1 are refused because they may permute qubits."""
    from qiskit import transpile
    if optimization_level > 1:
        raise ValueError("optimization_level must be 0 or 1 (qubit order must be preserved)")
    t = transpile(qc, basis_gates=list(basis), optimization_level=optimization_level, seed_transpiler=seed)
    lay = getattr(t, "layout", None)
    if lay is not None and getattr(lay, "final_layout", None) is not None:
        raise ValueError("transpilation permuted the qubits; refusing to use the circuit")
    return t


def resource_counts(qc, basis=BASIS, optimization_level=1, seed=0):
    t = transpile_layers(qc, basis, optimization_level, seed)
    ops = dict(t.count_ops())
    two = int(ops.get("cx", 0))
    one = int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x")))
    return {"depth": int(t.depth()), "two_qubit": two, "one_qubit": one, "ops": {k: int(v) for k, v in ops.items()}}


def check_against_numpy(mpsx, theta, p, atol=1e-8):
    """M1/M2: Qiskit statevector (ideal Dicke preparation + layers) vs the numpy engine, item marginals."""
    from .noise import FullSpaceMPSX
    fm = FullSpaceMPSX(mpsx)
    sv = qiskit_probabilities(mpsx.name, mpsx, mpsx.inst, theta, p)
    q_marg = np.bincount(fm.xint, weights=sv, minlength=2 ** mpsx.inst.N)
    xint, prob, _ = mpsx.dist(theta, p)
    n_marg = np.bincount(xint, weights=prob, minlength=2 ** mpsx.inst.N)
    d = float(np.abs(q_marg - n_marg).max())
    return d < atol, d
