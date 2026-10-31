"""Experiment drivers shared by the scripts."""
import json
import time

import numpy as np
import pandas as pd

from .classical import greedy, simulated_annealing, tabu_search
from .methods import make_method
from .train import run_chain


def uniform_baseline(exact, method):
    """P(optimum) of the uniform distribution over the method's reachable space (structural methods)."""
    if method.name in ("M1", "M2"):
        xint = method.xint
        return float(np.isin(xint, exact.opt_set).mean())
    return np.nan


def tune_penalties(insts_exacts, cfg, seeds, p, maxiter, shots, log=print, methods=("M3", "M4")):
    """Grid search on tuning instances (methods.tuning). Returns {'M3': {'A':..}, 'M4': {...}}, table."""
    rows = []
    best = {}
    m = cfg["methods"]
    grid = {"M3": [{"A": a} for a in m["M3"]["penalty_A_grid"]],
            "M4": [{"lambda1": a, "lambda2": b} for a in m["M4"]["lambda_grid"]["lambda1"] for b in m["M4"]["lambda_grid"]["lambda2"]]}
    for name, cands in grid.items():
        if name not in methods:
            continue
        scores = []
        for cand in cands:
            vals_opt, vals_feas = [], []
            for inst, ex in insts_exacts:
                try:
                    meth = make_method(name, inst, ex, cfg, {name: cand})
                except MemoryError as e:
                    log(f"  tuning {name} skipped: {e}")
                    vals_opt = None
                    break
                for s in seeds:
                    r = run_chain(meth, ex, s, p, maxiter, m["common"]["cvar_alpha"], shots, p_list=[p])[-1]
                    vals_opt.append(r["p_opt"]); vals_feas.append(r["p_feasible"])
            if vals_opt is None:
                break
            sc = (float(np.mean(vals_opt)), float(np.mean(vals_feas)))
            scores.append((sc, cand))
            rows.append({"method": name, **cand, "mean_p_opt": sc[0], "mean_p_feasible": sc[1]})
            log(f"  tuning {name} {cand}: mean P(opt) {sc[0]:.4f}  mean P(feasible) {sc[1]:.4f}")
        if scores:
            best[name] = max(scores, key=lambda t: t[0])[1]
    return best, pd.DataFrame(rows)


def run_quantum(inst, ex, cfg, methods, seeds, p_max, maxiter, shots, tuned, log=print):
    rows, skipped = [], []
    alpha = cfg["methods"]["common"]["cvar_alpha"]
    for name in methods:
        try:
            meth = make_method(name, inst, ex, cfg, tuned)
        except (MemoryError, ValueError) as e:
            skipped.append({"method": name, "reason": str(e)})
            log(f"  {name}: SKIPPED ({e})")
            continue
        ub = uniform_baseline(ex, meth)
        t0 = time.time()
        for s in seeds:
            for r in run_chain(meth, ex, s, p_max, maxiter, alpha, shots):
                theta = r["theta"]
                r = {k: v for k, v in r.items() if k not in ("theta", "history")}
                r.update({"theta_json": json.dumps([float(t) for t in theta]),
                          "method_params_json": json.dumps(getattr(meth, "params", {}), default=float),
                          "n_qubits": meth.n_qubits, "uniform_p_opt": ub, "instance": inst.name, "cvar_alpha": alpha,
                          "cvar_saturated": bool(r["cvar_final"] <= ex.f_opt + 1e-9 * max(1.0, abs(ex.f_opt)))})
                rows.append(r)
        log(f"  {name}: {meth.n_qubits} qubits, {len(seeds)} seeds x p<= {p_max}, {time.time() - t0:.1f}s")
    return pd.DataFrame(rows), skipped


def run_classical(inst, ex, cfg, seeds):
    c = cfg["classical"]
    span = ex.f_worst - ex.f_opt if ex.f_worst > ex.f_opt else 1.0
    tol = 1e-9 * max(1.0, abs(ex.f_opt))
    rows = []

    def rec(name, x, seed, nev, dt):
        x = np.asarray(x)
        fe = bool(inst.feasible(x)[0])
        fv = float(inst.f(x)[0])
        rows.append({"method": name, "seed": seed, "evaluations": nev, "seconds": dt, "feasible": fe, "f": fv,
                     "hit_opt": bool(fe and fv <= ex.f_opt + tol),
                     "ar": (ex.f_worst - fv) / span if fe else np.nan, "instance": inst.name})

    t = time.time(); x = greedy(inst); rec("greedy", x, -1, None, time.time() - t)
    for s in seeds:
        t = time.time()
        x, nev = simulated_annealing(inst, span, c["simulated_annealing"]["evaluations"], c["simulated_annealing"]["penalty_A"],
                                     c["simulated_annealing"]["T0_rel"], c["simulated_annealing"]["T_end_rel"], s)
        rec("SA", x, s, nev, time.time() - t)
        t = time.time()
        x, nev = tabu_search(inst, span, c["tabu"]["evaluations"], c["tabu"]["penalty_A"], c["tabu"]["tenure"], s)
        rec("tabu", x, s, nev, time.time() - t)
    return pd.DataFrame(rows)


def aggregate(df_q):
    if df_q.empty:
        return df_q
    g = df_q.groupby(["instance", "method", "p"])
    return g.agg(n_qubits=("n_qubits", "first"), n_params=("n_params", "first"), seeds=("seed", "count"),
                 p_opt_median=("p_opt", "median"), p_opt_q1=("p_opt", lambda v: v.quantile(0.25)),
                 p_opt_q3=("p_opt", lambda v: v.quantile(0.75)), p_feasible_median=("p_feasible", "median"),
                 ar_best_median=("ar_best_sampled", "median"), success_rate=("shots_hit_opt", "mean"),
                 nfev_median=("nfev", "median"), uniform_p_opt=("uniform_p_opt", "first"),
                 train_seconds_median=("train_seconds", "median")).reset_index()
