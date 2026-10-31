"""
Statevector engines (numpy only).

Gate conventions (identical to Qiskit):
  RX(2b)      = exp(-i b X)                          (QAOA X mixer exp(-i b sum X))
  XY(b)       = RXX(b) RYY(b) = exp(-i b (XX+YY)/2)  acts on {|01>,|10>} as [[cos b, -i sin b], [-i sin b, cos b]]
  RZ(t)       = exp(-i t Z / 2);  RZZ(t) = exp(-i t ZZ / 2)
  phase(g,E)  = exp(-i g E) on the computational basis

Qubit q of a full-space state is bit q of the basis index (little-endian, as Qiskit).

Two engines:
  * FullSpace: 2^n amplitudes (used for X-mixer methods, noise trajectories and cross-checks)
  * Register:  fixed-Hamming-weight subspace of one register (used for XY methods; exact, much smaller)
"""
from itertools import combinations

import numpy as np


# ------------------------------------------------------------------ full space
class FullSpace:
    def __init__(self, n):
        self.n = n
        self.dim = 2 ** n
        self.idx = np.arange(self.dim, dtype=np.int64)
        self._edge_cache = {}

    def bit(self, q):
        return ((self.idx >> q) & 1).astype(np.int8)

    def rx_all(self, psi, beta, qubits=None):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in (range(self.n) if qubits is None else qubits):
            v = psi.reshape(-1, 2, 2 ** q)
            v0, v1 = v[:, 0, :], v[:, 1, :]
            t = v0.copy()
            v0 *= c
            v0 += s * v1
            v1 *= c
            v1 += s * t
        return psi

    def _edge(self, i, j):
        key = (i, j)
        if key not in self._edge_cache:
            bi, bj = (self.idx >> i) & 1, (self.idx >> j) & 1
            I = self.idx[(bi == 1) & (bj == 0)]
            J = I ^ (1 << i) ^ (1 << j)
            self._edge_cache[key] = (I, J)
        return self._edge_cache[key]

    def xy(self, psi, i, j, beta):
        I, J = self._edge(i, j)
        a, b = psi[I].copy(), psi[J].copy()
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi[I] = c * a + s * b
        psi[J] = s * a + c * b
        return psi

    # Pauli operators (noise trajectories)
    def pauli(self, psi, q, which):
        if which == "I":
            return psi
        v = psi.reshape(-1, 2, 2 ** q)
        if which in ("X", "Y"):
            v[:] = v[:, ::-1, :].copy()
            if which == "Y":            # Y|0> = i|1>, Y|1> = -i|0> (applied after the swap)
                v[:, 0, :] *= -1j       # Y|1> = -i|0>
                v[:, 1, :] *= 1j        # Y|0> =  i|1>
        elif which == "Z":
            v[:, 1, :] *= -1
        return psi


# ------------------------------------------------------------------ register subspace
class Register:
    """Fixed-weight subspace of the qubits `qubits` (global ids), weight w.
    `item_of_local[j]` = global item id of local qubit j, or -1 for a slack qubit."""

    def __init__(self, item_of_local, weight):
        self.m = len(item_of_local)
        self.item_of_local = np.asarray(item_of_local, int)
        self.w = int(weight)
        states = [sum(1 << j for j in c) for c in combinations(range(self.m), self.w)]
        self.states = np.array(sorted(states), dtype=np.int64)
        self.dim = len(self.states)
        self.pos = {s: k for k, s in enumerate(self.states)}
        loc_bits = ((self.states[:, None] >> np.arange(self.m)) & 1).astype(np.int8)
        self.loc_bits = loc_bits
        self.z = (1 - 2 * loc_bits).astype(np.int8)
        xint = np.zeros(self.dim, dtype=np.int64)
        for j, it in enumerate(self.item_of_local):
            if it >= 0:
                xint |= loc_bits[:, j].astype(np.int64) << it
        self.xint = xint
        self._edge_cache = {}

    def edge(self, a, b):
        key = (a, b)
        if key not in self._edge_cache:
            sel = ((self.states >> a) & 1 == 1) & ((self.states >> b) & 1 == 0)
            I = np.flatnonzero(sel)
            J = np.array([self.pos[s ^ (1 << a) ^ (1 << b)] for s in self.states[I]], dtype=np.int64)
            self._edge_cache[key] = (I, J)
        return self._edge_cache[key]

    def xy(self, psi, a, b, beta):
        I, J = self.edge(a, b)
        if len(I) == 0:
            return psi
        x, y = psi[I].copy(), psi[J].copy()
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi[I] = c * x + s * y
        psi[J] = s * x + c * y
        return psi

    def dicke(self):
        return np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)

    def basis(self, local_state):
        psi = np.zeros(self.dim, dtype=complex)
        psi[self.pos[local_state]] = 1.0
        return psi


def ring_edges(m):
    """Ring on m local qubits, ordered even edges then odd edges (parity ordering)."""
    if m < 2:
        return []
    if m == 2:
        return [(0, 1)]
    edges = [(i, (i + 1) % m) for i in range(m)]
    return edges[0::2] + edges[1::2]


def product_distribution(regs, amps, cap=5_000_000):
    """Joint item-space distribution of independent registers: returns (xint, prob)."""
    xint = np.zeros(1, dtype=np.int64)
    prob = np.ones(1)
    size = 1
    for r, a in zip(regs, amps):
        size *= r.dim
        if size > cap:
            raise MemoryError(f"joint register space {size} > cap {cap}")
        pr = np.abs(a) ** 2
        xint = (xint[:, None] | r.xint[None, :]).ravel()
        prob = (prob[:, None] * pr[None, :]).ravel()
    return xint, prob


def cvar(energies, probs, alpha, order=None):
    """CVaR_alpha (lower tail, minimisation) of a discrete distribution (Barkoutsos et al. 2020).
    `order` = precomputed argsort of `energies` (energies are fixed during training)."""
    if alpha >= 1.0:
        return float(np.dot(probs, energies) / max(probs.sum(), 1e-300))
    o = np.argsort(energies, kind="stable") if order is None else order
    e, pr = energies[o], probs[o]
    c = np.cumsum(pr)
    k = int(np.searchsorted(c, alpha))
    k = min(k, len(e) - 1)
    head = c[k - 1] if k > 0 else 0.0
    return float((np.dot(pr[:k], e[:k]) + (alpha - head) * e[k]) / alpha)
