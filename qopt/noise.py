"""
Simulated noise sensitivity (a noisy-simulator robustness test, not hardware validation).

Monte Carlo Pauli trajectories on the full statevector:
  * after every one-qubit rotation (RZ of the phase separator, RX of the X mixer): depolarising with prob p1
  * after every two-qubit gate (RZZ of the phase separator on coupled pairs, each XY edge): depolarising with prob p2
  * readout: each measured bit flipped with prob p_ro
The ideal state preparation (Dicke / |+>) is assumed noiseless and this is stated in the output.
"""
import numpy as np

from .sim import FullSpace

P1 = ["X", "Y", "Z"]
P2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]


MPSX_IDS = ("M1", "M2")      # structural XY-register methods (single- and multi-angle)


class FullSpaceMPSX:
    """Full-space realisation of an MPSX (M1/M2) method, for noise trajectories and cross-checks."""

    def __init__(self, mpsx):
        self.m = mpsx
        inst = mpsx.inst
        N = inst.N
        self.global_of = []      # per register: local position -> global qubit
        nxt = N
        for reg in mpsx.regs:
            gl = []
            for it in reg.item_of_local:
                if it >= 0:
                    gl.append(int(it))
                else:
                    gl.append(nxt); nxt += 1
            self.global_of.append(gl)
        self.n = nxt
        self.fs = FullSpace(self.n)
        idx = self.fs.idx
        self.xint = idx & ((1 << N) - 1)
        self.f = mpsx.exact.f_all[self.xint]
        ok = np.ones(self.fs.dim, bool)
        for reg, gl in zip(mpsx.regs, self.global_of):
            w = np.zeros(self.fs.dim, np.int16)
            for q in gl:
                w += ((idx >> q) & 1).astype(np.int16)
            ok &= w == reg.w
        self.weight_ok = ok
        self.pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if inst.W[i, j] != 0.0]

    def initial(self):
        psi = self.weight_ok.astype(complex)
        return psi / np.linalg.norm(psi)


def _depol1(fs, psi, q, p, rng):
    if p > 0 and rng.random() < p:
        fs.pauli(psi, q, P1[rng.integers(3)])


def _depol2(fs, psi, a, b, p, rng):
    if p > 0 and rng.random() < p:
        pa, pb = P2[rng.integers(15)]
        fs.pauli(psi, a, pa)
        fs.pauli(psi, b, pb)


def trajectory_mpsx(fm, theta, p_layers, p1, p2, rng):
    m = fm.m
    th = np.asarray(theta).reshape(p_layers, m.per_layer)
    fs = fm.fs
    psi = fm.initial()
    N = m.inst.N
    for l in range(p_layers):
        psi *= np.exp(-1j * th[l, 0] * fm.f / m.span)
        for q in range(N):
            _depol1(fs, psi, q, p1, rng)
        for (i, j) in fm.pairs:
            _depol2(fs, psi, i, j, p2, rng)
        for g, (edges, gl) in enumerate(zip(m.reg_edges, fm.global_of)):
            off = sum(len(E) for E in m.reg_edges[:g])
            for t, (a, b) in enumerate(edges):
                beta = th[l, 1 + off + t] if m.multi else th[l, 1 + g]
                fs.xy(psi, gl[a], gl[b], beta)
                _depol2(fs, psi, gl[a], gl[b], p2, rng)
    return psi


def trajectory_penalty(meth, theta, p_layers, p1, p2, rng):
    th = np.asarray(theta).reshape(p_layers, 2)
    fs = meth.fs
    inst = meth.inst
    n = meth.n_qubits
    # qubits coupled by the penalty/objective Ising terms: all pairs inside a group's item+slack set
    groups_q = []
    off = inst.N
    for g in range(inst.G):
        q = list(inst.items_of(g))
        if meth.kind == "slack":
            q += list(range(off, off + meth.slack_bits[g]))
            off += meth.slack_bits[g]
        groups_q.append(q)
    pairs = [(a, b) for q in groups_q for i, a in enumerate(q) for b in q[i + 1:]]
    psi = np.full(fs.dim, 1 / np.sqrt(fs.dim), dtype=complex)
    for l in range(p_layers):
        psi *= np.exp(-1j * th[l, 0] * meth.e_train / meth.span)
        for q in range(n):
            _depol1(fs, psi, q, p1, rng)
        for (a, b) in pairs:
            _depol2(fs, psi, a, b, p2, rng)
        fs.rx_all(psi, th[l, 1])
        for q in range(n):
            _depol1(fs, psi, q, p1, rng)
    return psi


def noisy_distribution(kind, obj, theta, p_layers, p1, p2, trajectories, seed):
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(trajectories):
        psi = trajectory_mpsx(obj, theta, p_layers, p1, p2, rng) if kind in MPSX_IDS else \
            trajectory_penalty(obj, theta, p_layers, p1, p2, rng)
        pr = np.abs(psi) ** 2
        acc = pr if acc is None else acc + pr
    return acc / trajectories


def sample_with_readout(prob, n_qubits, shots, p_ro, seed):
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(prob), size=shots, p=prob / prob.sum()).astype(np.int64)
    if p_ro > 0:
        flips = (rng.random((shots, n_qubits)) < p_ro).astype(np.int64)
        mask = (flips << np.arange(n_qubits)).sum(1)
        draws ^= mask
    return draws
