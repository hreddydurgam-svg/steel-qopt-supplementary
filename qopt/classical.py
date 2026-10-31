"""
Classical reference solvers on the same Instance.
  exact      : enumeration (qopt.instance.solve_exact)
  milp       : PuLP/CBC on the McCormick linearisation of the QUBO (MILP); optional
  greedy     : per group, add the item with the smallest marginal objective until count = k_g
  SA / tabu  : single-bit-flip search on f + A*M*sum_g deficit_g^2 (M = penalty_scale) with a fixed evaluation budget
"""
import time

import numpy as np


def penalised(inst, x, A, M):
    """f + A*M*sum_g deficit_g^2, M = penalty_scale(inst)."""
    x = np.asarray(x, float)
    d = 0.0
    for g in range(inst.G):
        d += max(0, inst.k[g] - x[inst.items_of(g)].sum()) ** 2
    return float(inst.f(x)[0] + A * M * d)


def greedy(inst):
    x = np.zeros(inst.N)
    for g in range(inst.G):
        items = list(inst.items_of(g))
        for _ in range(int(inst.k[g])):
            best, bi = None, None
            for i in items:
                if x[i] == 0:
                    y = x.copy(); y[i] = 1
                    v = inst.f(y)[0]
                    if best is None or v < best:
                        best, bi = v, i
            x[bi] = 1
    return x.astype(int)


def simulated_annealing(inst, span, evaluations, A, T0_rel, T_end_rel, seed):
    from .instance import penalty_scale
    M = penalty_scale(inst)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, inst.N)
    e = penalised(inst, x, A, M)
    best_x, best_e = x.copy(), e
    T0, T1 = T0_rel * span, T_end_rel * span
    n_ev = 1
    while n_ev < evaluations:
        T = T0 * (T1 / T0) ** (n_ev / evaluations)
        i = rng.integers(inst.N)
        y = x.copy(); y[i] ^= 1
        ey = penalised(inst, y, A, M); n_ev += 1
        if ey <= e or rng.random() < np.exp(-(ey - e) / T):
            x, e = y, ey
            if e < best_e:
                best_x, best_e = x.copy(), e
    return best_x, n_ev


def tabu_search(inst, span, evaluations, A, tenure, seed):
    from .instance import penalty_scale
    M = penalty_scale(inst)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, inst.N)
    best_x, best_e = x.copy(), penalised(inst, x, A, M)
    tabu, n_ev = [], 1
    while n_ev + inst.N <= evaluations:
        cand = []
        for i in range(inst.N):
            if i in tabu:
                continue
            y = x.copy(); y[i] ^= 1
            cand.append((penalised(inst, y, A, M), i, y)); n_ev += 1
        if not cand:
            break
        e, i, x = min(cand, key=lambda c: c[0])
        tabu.append(i)
        if len(tabu) > tenure:
            tabu.pop(0)
        if e < best_e:
            best_x, best_e = x.copy(), e
    return best_x, n_ev


def milp(inst, msg=False):
    """McCormick-linearised QUBO solved with CBC. Returns (x, seconds, solver_version) or raises ImportError."""
    import pulp
    t0 = time.time()
    N = inst.N
    prob = pulp.LpProblem("steel_qubo_milp", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(N)]
    terms = [(inst.lin[i] + inst.W[i, i]) * x[i] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            w = inst.W[i, j]
            if w != 0.0:
                z = pulp.LpVariable(f"z{i}_{j}", lowBound=0, upBound=1)
                prob += z <= x[i]
                prob += z <= x[j]
                prob += z >= x[i] + x[j] - 1
                terms.append(2.0 * w * z)
    prob += pulp.lpSum(terms)
    for g in range(inst.G):
        prob += pulp.lpSum(x[i] for i in inst.items_of(g)) >= int(inst.k[g])
    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP status {pulp.LpStatus[prob.status]}")
    sol = np.array([int(round(v.varValue)) for v in x])
    return sol, time.time() - t0, getattr(pulp, "__version__", "unknown")
