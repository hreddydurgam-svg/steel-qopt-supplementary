"""
Quantum methods (parameter file section 'methods').

  M1  MPSX-QAOA (proposed), multi-angle: one XY register per group (items + s_g = n_g - k_g slack qubits,
      weight n_g), Dicke initial state per register, phase exp(-i gamma f/span), XY ring mixer per register with
      one beta per XY edge. Every reachable state satisfies count_g >= k_g (structural; no penalty).
  M2  MPSX-QAOA, single-angle variant: as M1 with one beta per register.
  M3  Standard QAOA: |+>^n, X mixer, phase from f + quadratic penalty with binary slack per group.
  M4  QAOA with unbalanced penalisation (Montanez-Barrera et al. 2024): X mixer, no slack qubits.
  M5  Multi-angle particle-conserving ansatz (MA-PC), adapted: one global register (weight N, S = N - sum k slacks),
      free RZ / RZZ angles, XY chain + slack-to-every-item XY; per-group deficits penalised in the CVaR energy.
Initial-angle ranges are set from methods.common / methods.M5 by make_method().

Each method exposes: n_params(p), init(rng, p=1), dist(theta, p) -> (xint, prob, e_train) and metadata.
"""
import numpy as np

from .instance import penalty_scale
from .sim import FullSpace, Register, ring_edges, product_distribution


def _span(exact):
    s = exact.f_worst - exact.f_opt
    return s if s > 0 else 1.0


def interp(theta_p, p):
    """INTERP (Zhou et al. 2020): (p, m) -> (p+1, m)."""
    th = np.asarray(theta_p).reshape(p, -1)
    out = np.zeros((p + 1, th.shape[1]))
    for i in range(1, p + 2):                       # 1-based i = 1..p+1
        left = th[i - 2] if i - 2 >= 0 else 0.0
        right = th[i - 1] if i - 1 < p else 0.0
        out[i - 1] = (i - 1) / p * left + (p - i + 1) / p * right
    return out.ravel()


# ------------------------------------------------------------------ M1 / M2
class MPSX:
    init_gamma, init_beta = (0.0, np.pi), (0.0, np.pi / 2)        # overwritten by make_method from the JSON
    shot_seed_stride = 1000

    def __init__(self, inst, exact, multi_angle=False):
        self.name = "M1" if multi_angle else "M2"
        self.inst, self.exact = inst, exact
        self.span = _span(exact)
        self.multi = multi_angle
        self.regs, self.reg_E, self.reg_edges = [], [], []
        for g in range(inst.G):
            items = list(inst.items_of(g))
            s = len(items) - int(inst.k[g])
            reg = Register(items + [-1] * s, len(items))
            # register energy: f restricted to the group's items (slack qubits carry no cost)
            bits = reg.loc_bits[:, :len(items)].astype(float)
            lin = inst.lin[items]
            Wg = inst.W[np.ix_(items, items)]
            e = bits @ lin + np.einsum("si,ij,sj->s", bits, Wg, bits)
            self.regs.append(reg)
            self.reg_E.append(e)
            self.reg_edges.append(ring_edges(reg.m))
        self.n_qubits = sum(r.m for r in self.regs)
        # joint training energy (all states feasible) and joint item xint, computed once
        xint = np.zeros(1, dtype=np.int64)
        e_tot = np.zeros(1)
        for r, e in zip(self.regs, self.reg_E):
            xint = (xint[:, None] | r.xint[None, :]).ravel()
            e_tot = (e_tot[:, None] + e[None, :]).ravel()
        self.xint, self.e_train = xint, e_tot
        self.n_edges = sum(len(E) for E in self.reg_edges)
        self.per_layer = 1 + (self.n_edges if multi_angle else inst.G)

    def n_params(self, p):
        return p * self.per_layer

    def init(self, rng, p=1):
        th = []
        for _ in range(p):
            th += [rng.uniform(*self.init_gamma)] + list(rng.uniform(*self.init_beta, self.per_layer - 1))
        return np.array(th)

    def amplitudes(self, theta, p):
        th = np.asarray(theta).reshape(p, self.per_layer)
        amps = []
        for g, (reg, e, edges) in enumerate(zip(self.regs, self.reg_E, self.reg_edges)):
            psi = reg.dicke()
            off = sum(len(E) for E in self.reg_edges[:g])
            for l in range(p):
                psi *= np.exp(-1j * th[l, 0] * e / self.span)
                for t, (a, b) in enumerate(edges):
                    beta = th[l, 1 + off + t] if self.multi else th[l, 1 + g]
                    psi = reg.xy(psi, a, b, beta)
            amps.append(psi)
        return amps

    def dist(self, theta, p):
        amps = self.amplitudes(theta, p)
        _, prob = product_distribution(self.regs, amps)
        return self.xint, prob, self.e_train


# ------------------------------------------------------------------ M3 / M4 (full space, X mixer)
class PenaltyQAOA:
    init_gamma, init_beta = (0.0, np.pi), (0.0, np.pi / 2)        # overwritten by make_method from the JSON
    shot_seed_stride = 1000

    def __init__(self, inst, exact, kind, A=None, lam=None, qubit_cap=21):
        self.name = {"slack": "M3", "unbalanced": "M4"}[kind]
        self.inst, self.exact, self.kind = inst, exact, kind
        self.span = _span(exact)
        N = inst.N
        self.slack_bits = []
        if kind == "slack":
            for g in range(inst.G):
                rng_ = len(inst.items_of(g)) - int(inst.k[g])
                self.slack_bits.append(int(np.ceil(np.log2(rng_ + 1))) if rng_ > 0 else 0)
        n = N + sum(self.slack_bits)
        if n > qubit_cap:
            raise MemoryError(f"{self.name}: {n} qubits > cap {qubit_cap}")
        self.n_qubits = n
        self.fs = FullSpace(n)
        idx = self.fs.idx
        self.xint = idx & ((1 << N) - 1)
        f_items = exact.f_all[self.xint]
        pen = np.zeros(self.fs.dim)
        off = N
        for g in range(inst.G):
            items = inst.items_of(g)
            n_g = len(items)
            cnt = np.zeros(self.fs.dim)
            for i in items:
                cnt += (idx >> i) & 1
            if kind == "slack":
                s = np.zeros(self.fs.dim)
                for b in range(self.slack_bits[g]):
                    s += ((idx >> (off + b)) & 1) * (1 << b)
                off += self.slack_bits[g]
                pen += A * (cnt - inst.k[g] - s) ** 2
            else:
                h = cnt - inst.k[g]
                pen += -lam[0] * h + lam[1] * h ** 2
        self.M = penalty_scale(inst)
        self.e_train = f_items + self.M * pen
        self.params = {"A": A} if kind == "slack" else {"lambda1": lam[0], "lambda2": lam[1]}
        self.per_layer = 2

    def n_params(self, p):
        return 2 * p

    def init(self, rng, p=1):
        th = []
        for _ in range(p):
            th += [rng.uniform(*self.init_gamma), rng.uniform(*self.init_beta)]
        return np.array(th)

    def state(self, theta, p):
        th = np.asarray(theta).reshape(p, 2)
        psi = np.full(self.fs.dim, 1 / np.sqrt(self.fs.dim), dtype=complex)
        for l in range(p):
            psi *= np.exp(-1j * th[l, 0] * self.e_train / self.span)
            psi = self.fs.rx_all(psi, th[l, 1])
        return psi

    def dist(self, theta, p):
        return self.xint, np.abs(self.state(theta, p)) ** 2, self.e_train


# ------------------------------------------------------------------ M5 (MA-PC, adapted)
class MAPC:
    init_halfwidth = 0.01                                             # overwritten by make_method from the JSON
    shot_seed_stride = 1000

    def __init__(self, inst, exact, A=10.0, dim_cap=60000):
        self.name = "M5"
        self.inst, self.exact = inst, exact
        self.span = _span(exact)
        N = inst.N
        S = N - int(inst.k.sum())
        from math import comb
        if comb(N + S, N) > dim_cap:
            raise MemoryError(f"M5: register dimension {comb(N + S, N)} > cap {dim_cap}")
        self.S = S
        self.reg = Register(list(range(N)) + [-1] * S, N)
        self.n_qubits = N + S
        self.pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if inst.W[i, j] != 0.0]
        self.chain = [(i, i + 1) for i in range(N - 1)]
        self.slack_edges = [(N + s, r) for s in range(S) for r in range(N)]
        self.per_layer = N + len(self.pairs) + len(self.chain) + len(self.slack_edges)
        X = self.reg.loc_bits[:, :N]
        f = exact.f_all[self.reg.xint]
        deficit = np.zeros(self.reg.dim)
        for g in range(inst.G):
            items = inst.items_of(g)
            deficit += np.maximum(0, inst.k[g] - X[:, items].sum(1)) ** 2
        self.xint = self.reg.xint
        self.e_train = f + penalty_scale(inst) * A * deficit
        self.zz = np.stack([self.reg.z[:, i] * self.reg.z[:, j] for i, j in self.pairs], 1).astype(float) \
            if self.pairs else np.zeros((self.reg.dim, 0))
        self.zi = self.reg.z[:, :N].astype(float)
        # seed state: a feasible basis state (first k_g items of each group + all slacks)
        seed = 0
        for g in range(inst.G):
            for i in inst.items_of(g)[: inst.k[g]]:
                seed |= 1 << int(i)
        for s in range(S):
            seed |= 1 << (N + s)
        self.seed_state = seed
        self.params = {"A": A, "S": S}

    def n_params(self, p):
        return p * self.per_layer

    def init(self, rng, p=1):
        return rng.uniform(-self.init_halfwidth, self.init_halfwidth, self.n_params(p))

    def state(self, theta, p):
        N = self.inst.N
        th = np.asarray(theta).reshape(p, self.per_layer)
        psi = self.reg.basis(self.seed_state)
        for l in range(p):
            t = th[l]
            a = 0
            ph = self.zi @ t[a:a + N] / 2.0; a += N
            if self.pairs:
                ph = ph + self.zz @ t[a:a + len(self.pairs)] / 2.0
            a += len(self.pairs)
            psi *= np.exp(-1j * ph)
            for (i, j) in self.chain:
                psi = self.reg.xy(psi, i, j, t[a]); a += 1
            for (sq, r) in self.slack_edges:
                psi = self.reg.xy(psi, sq, r, t[a]); a += 1
        return psi

    def dist(self, theta, p):
        return self.xint, np.abs(self.state(theta, p)) ** 2, self.e_train


def make_method(name, inst, exact, cfg, tuned=None):
    tuned = tuned or {}
    m = cfg["methods"]
    cap = cfg["synthetic_suite"]["full_space_qubit_cap"]
    c = m["common"]
    gam = tuple(np.pi * v for v in c["init_gamma_range_over_pi"])
    bet = tuple(np.pi * v for v in c["init_beta_range_over_pi"])

    def need(key):
        if key not in tuned:
            raise ValueError(f"{name}: penalty parameters not supplied (tuned or fixed values from the JSON are required)")
        return tuned[key]
    if name in ("M1", "M2"):
        meth = MPSX(inst, exact, name == "M1")
    elif name == "M3":
        meth = PenaltyQAOA(inst, exact, "slack", A=need("M3")["A"], qubit_cap=cap)
    elif name == "M4":
        t = need("M4")
        meth = PenaltyQAOA(inst, exact, "unbalanced", lam=(t["lambda1"], t["lambda2"]), qubit_cap=cap)
    elif name == "M5":
        meth = MAPC(inst, exact, A=m["M5"]["penalty_A"], dim_cap=m["M5"]["subspace_dim_cap"])
        meth.init_halfwidth = m["M5"]["init_halfwidth"]
        meth.shot_seed_stride = c["shot_seed_stride"]
        return meth
    else:
        raise ValueError(name)
    meth.init_gamma, meth.init_beta = gam, bet
    meth.shot_seed_stride = c["shot_seed_stride"]
    return meth
