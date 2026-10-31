"""
Problem instance: multi-group cardinality-constrained QUBO.

    minimise  f(x) = lin . x + x^T W x          x in {0,1}^N
    subject to count_g(x) = sum_{i in group g} x_i >= k_g   for every group g

W is symmetric and has non-zero entries only inside a group, so f is separable by group:
f(x) = sum_g f_g(x_g). Both the steel case and the synthetic suite use this form.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instance:
    name: str
    group_names: list
    item_group: np.ndarray          # (N,) int, group index of each item
    k: np.ndarray                   # (G,) int, count lower bound per group
    lin: np.ndarray                 # (N,) linear coefficients (weights and normalisation applied)
    W: np.ndarray                   # (N,N) symmetric quadratic coefficients (weights applied)
    cost_usd: np.ndarray            # (N,) raw cost per item (US$/month), for reporting
    emis_kg: np.ndarray             # (N,) raw emissions per item (kg CO2e/month)
    Q_raw: np.ndarray               # (N,N) raw risk matrix (before weight / R_ref)
    refs: dict                      # C_ref, E_ref, R_ref
    weights: dict                   # w_cost, w_emit, w_risk
    labels: list = field(default_factory=list)   # item labels, e.g. "H1#0"
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.lin)

    @property
    def G(self):
        return len(self.group_names)

    def items_of(self, g):
        return np.flatnonzero(self.item_group == g)

    def check(self):
        N = self.N
        assert self.W.shape == (N, N) and np.allclose(self.W, self.W.T), "W must be symmetric"
        for i in range(N):
            for j in range(N):
                if self.item_group[i] != self.item_group[j] and self.W[i, j] != 0:
                    raise ValueError("W couples different groups; separability assumption violated")
        for g in range(self.G):
            n_g = len(self.items_of(g))
            if not (0 <= self.k[g] <= n_g):
                raise ValueError(f"k[{g}]={self.k[g]} outside [0, {n_g}]")
        return True

    # --- evaluation on explicit bit vectors -------------------------------------------
    def f(self, X):
        X = np.atleast_2d(X).astype(float)
        return X @ self.lin + np.einsum("si,ij,sj->s", X, self.W, X)

    def counts(self, X):
        X = np.atleast_2d(X)
        return np.stack([X[:, self.items_of(g)].sum(1) for g in range(self.G)], axis=1)

    def feasible(self, X):
        return np.all(self.counts(X) >= self.k, axis=1)

    def components(self, x):
        """Raw cost (US$), emissions (kg) and risk for one bit vector."""
        x = np.asarray(x, float)
        return float(x @ self.cost_usd), float(x @ self.emis_kg), float(x @ self.Q_raw @ x)


def penalty_scale(inst):
    """M = largest possible increase of f from adding one item: lin_i + W_ii + 2 sum_j |W_ij|.
    With penalty A*M*deficit^2 and A > 1, omitting a required item never lowers the penalised energy."""
    off = 2.0 * (np.abs(inst.W).sum(1) - np.abs(np.diag(inst.W)))
    return float(np.max(inst.lin + np.diag(inst.W) + off))


def bits_of(xint, N):
    xint = np.asarray(xint, dtype=np.int64)
    return ((xint[..., None] >> np.arange(N)) & 1).astype(np.int8)


def energy_table(inst, N_cap=24):
    """f(x) and feasibility for all 2^N item bitstrings (bit i = item i)."""
    N = inst.N
    if N > N_cap:
        raise MemoryError(f"N = {N} > cap {N_cap} for full enumeration")
    idx = np.arange(2 ** N, dtype=np.int64)
    f = np.zeros(2 ** N)
    bit = [((idx >> i) & 1).astype(np.float64) for i in range(N)]
    for i in range(N):
        f += (inst.lin[i] + inst.W[i, i]) * bit[i]
    for i in range(N):
        for j in range(i + 1, N):
            if inst.W[i, j] != 0.0:
                f += 2.0 * inst.W[i, j] * bit[i] * bit[j]
    feas = np.ones(2 ** N, bool)
    for g in range(inst.G):
        c = np.zeros(2 ** N, np.int16)
        for i in inst.items_of(g):
            c += bit[i].astype(np.int16)
        feas &= c >= inst.k[g]
    return f, feas


@dataclass
class Exact:
    f_all: np.ndarray
    feas_all: np.ndarray
    f_opt: float
    f_worst: float
    opt_set: np.ndarray        # all optimal xints (degenerate blocks give several)
    n_feasible: int


def solve_exact(inst, tol=1e-9):
    f_all, feas = energy_table(inst)
    ff = np.where(feas, f_all, np.inf)
    f_opt = float(ff.min())
    f_worst = float(np.where(feas, f_all, -np.inf).max())
    opt_set = np.flatnonzero(feas & (f_all <= f_opt + tol * max(1.0, abs(f_opt))))
    return Exact(f_all, feas, f_opt, f_worst, opt_set, int(feas.sum()))


def build_from_raw(name, group_names, item_group, k, cost_usd, emis_kg, Q_raw, weights, labels=None, meta=None):
    """Normalise with all-items-on references and apply weights."""
    item_group = np.asarray(item_group, int)
    cost_usd = np.asarray(cost_usd, float)
    emis_kg = np.asarray(emis_kg, float)
    Q_raw = np.asarray(Q_raw, float)
    C_ref, E_ref = cost_usd.sum(), emis_kg.sum()
    R_ref = float(np.ones(len(cost_usd)) @ Q_raw @ np.ones(len(cost_usd)))
    wc, we, wr = weights["w_cost"], weights["w_emit"], weights.get("w_risk", 0.0)
    lin = wc * cost_usd / C_ref + we * emis_kg / E_ref
    W = (wr / R_ref) * Q_raw if R_ref > 0 else np.zeros_like(Q_raw)
    W = 0.5 * (W + W.T)
    inst = Instance(name, list(group_names), item_group, np.asarray(k, int), lin, W, cost_usd, emis_kg, Q_raw,
                    {"C_ref": C_ref, "E_ref": E_ref, "R_ref": R_ref},
                    {"w_cost": wc, "w_emit": we, "w_risk": wr}, labels or [], meta or {})
    inst.check()
    return inst


def synthetic_instance(sizes, seed, spec):
    """Synthetic multi-group instance (declared synthetic; ranges from the steel scale)."""
    rng = np.random.default_rng(seed)
    item_group = np.concatenate([[g] * n for g, n in enumerate(sizes)])
    N = len(item_group)
    k = np.array([rng.integers(max(1, n - spec["k_spread"]), n) for n in sizes])   # in [max(1, n-k_spread), n-1]
    lo, hi = spec["cost_usd_per_item"]
    cost = rng.uniform(lo, hi, N)
    lo, hi = spec["emis_kg_per_item"]
    emis = rng.uniform(lo, hi, N)
    Q = np.zeros((N, N))
    start = 0
    for n in sizes:
        B = rng.normal(0, 1, (n, n))
        Q[start:start + n, start:start + n] = B @ B.T / n
        start += n
    labels = [f"g{item_group[i]}#{i}" for i in range(N)]
    return build_from_raw(f"synth{tuple(sizes)}_s{seed}", [f"g{g}" for g in range(len(sizes))], item_group, k,
                          cost, emis, Q, spec["weights"], labels, {"seed": seed, "sizes": list(sizes)})
