"""
Publication figures from the result CSVs (no re-computation of results).

Reads results/<section>_<profile>/ (and the alpha = 1 folders) written by the run_*.py scripts and writes, for every figure,
  results/figures_<profile>/<name>.pdf   (vector, fonts embedded as TrueType)
  results/figures_<profile>/<name>.png   (600 dpi)
  results/figures_<profile>/<name>.tiff  (600 dpi, LZW)
  results/figures_<profile>/<name>_data.csv (the exact numbers plotted)
plus figures_<profile>.zip containing all of them.

Checks: every result folder used must carry the SHA-256 of the current parameter file; otherwise the script stops.
A missing result folder is reported and its figure is skipped.
Usage:  python make_figures.py --profile quick|paper
"""
import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import ROOT, parse_profile
from qopt.plotting import COLOR, LABEL, MARKER

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

args = parse_profile()
p = load()
cfg = p.raw
PROF = args.profile
RES = ROOT / "results"
OUT = RES / f"figures_{PROF}"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style (journal: 8 pt text, 1-column 3.5 in, 2-column 7.2 in)
ONE, TWO = 3.5, 7.2
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
GREY = "#7a7a7a"
made, skipped = [], []


def panel_label(ax, letter):
    t = ax.get_title()
    ax.set_title(f"({letter})  {t}" if t else f"({letter})", loc="left", fontsize=8)
    ax.set_title("")                                   # clear the centred title (moved to the left with the label)


ZORDER = {"M1": 6, "M2": 5, "M3": 4, "M4": 3, "M5": 2}   # proposed method drawn on top


def log_floor(values):
    """Lower plotting limit for a log axis: one decade below the smallest positive value (zeros are drawn there)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    return 10 ** np.floor(np.log10(v.min()) - 1) if len(v) else 1e-6


def mark_zeros(ax, x, y, floor, color):
    """Draw exact zeros at the floor of a log axis as open downward triangles (value 0, not data at the floor)."""
    x, y = np.asarray(x), np.asarray(y, float)
    z = y <= 0
    if z.any():
        ax.scatter(x[z], np.full(z.sum(), floor), marker="v", s=22, facecolors="white", edgecolors=color, linewidths=0.9,
                   zorder=7, clip_on=False)
    return z.any()


def folder(name):
    d = RES / name
    if not d.exists():
        return None
    sj = d / "summary.json"
    if not sj.exists():
        print(f"  NOTE: {d.name} has no summary.json yet (run incomplete or still running) - its figure is skipped")
        return None
    h = json.loads(sj.read_text()).get("params_sha256")
    if h != p.sha256:
        sys.exit(f"STOP: {d} was produced with parameter file {h}, current is {p.sha256}. "
                 "Delete or move that folder and re-run its script.")
    return d


def save(fig, name, data):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    data.to_csv(OUT / f"{name}_data.csv", index=False)
    made.append(name)
    print(f"  wrote {name}.pdf/.png/.tiff (+ _data.csv)")


def methods_in(df):
    return [m for m in ["M1", "M2", "M3", "M4", "M5"] if m in set(df["method"])]


# ------------------------------------------------------------------ Fig 1: steel case, quantum methods vs depth (3.1)
def fig_steel(suffix="", name="Fig1_steel_quantum_vs_depth"):
    d = folder(f"3_1_steel_case_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"{name} (3.1{' ' + suffix if suffix else ''})")
    s = pd.read_csv(d / "quantum_summary.csv")
    runs = pd.read_csv(d / "quantum_runs.csv")
    alpha = float(runs["cvar_alpha"].iloc[0])
    sat = runs.groupby(["method", "p"])["cvar_saturated"].mean().rename("cvar_saturated_frac").reset_index()
    s = s.merge(sat, on=["method", "p"], how="left")
    fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.5))
    for m in methods_in(s):
        h = s[s["method"] == m].sort_values("p")
        yerr = np.vstack([h["p_opt_median"] - h["p_opt_q1"], h["p_opt_q3"] - h["p_opt_median"]])
        axs[0].errorbar(h["p"], h["p_opt_median"], yerr=yerr, color=COLOR[m], marker=MARKER[m], capsize=2, label=LABEL[m],
                        zorder=ZORDER[m])
        axs[1].plot(h["p"], h["p_feasible_median"], color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=ZORDER[m])
    u = s["uniform_p_opt"].dropna()
    if len(u):
        axs[0].axhline(u.iloc[0], color=GREY, linestyle="--", linewidth=0.9, label="uniform over MPSX reachable states")
    axs[0].set(xlabel="QAOA depth p", ylabel="P(optimum), median and IQR over seeds", title=f"training objective CVaR, α = {alpha:g}")
    axs[1].set(xlabel="QAOA depth p", ylabel="P(feasible), median over seeds", ylim=(0, 1.05), title=f"training objective CVaR, α = {alpha:g}")
    for a, l in zip(axs, "ab"):
        a.xaxis.set_major_locator(MaxNLocator(integer=True)); panel_label(a, l)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=3)
    save(fig, name, s)


# ------------------------------------------------------------------ Fig 2: gradient variance (3.2a)
def fig_gradvar():
    d = folder(f"3_2_trainability_{PROF}")
    if d is None:
        return skipped.append("Fig2-4 (3.2)")
    g = pd.read_csv(d / "gradient_variance.csv")
    pmax = int(g["p"].max())
    inits = list(dict.fromkeys(g["init"]))
    fig, axs = plt.subplots(1, len(inits), figsize=(TWO, 2.5), squeeze=False)
    Ns = sorted(g["N_items"].unique())
    rng = cfg["trainability"]["gradient_variance"]["inits"]
    for a, kind, l in zip(axs[0], inits, "abcd"):
        for m in methods_in(g):
            h = g[(g["method"] == m) & (g["init"] == kind) & (g["p"] == pmax)].set_index("N_items")["var"].reindex(Ns)
            a.plot(Ns, h.values, color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=ZORDER[m])   # gap = not run
        a.set_yscale("log")
        a.set_xticks(Ns)
        lo, hi = rng[kind]
        rtxt = "[0, 2π)" if abs(hi - 2 * np.pi) < 1e-9 and lo == 0 else f"[{lo:g}, {hi:g}]"
        a.set(xlabel="items N", ylabel=f"mean Var[∂E/∂β], last layer (p = {pmax})", title=f"initial angles uniform in {rtxt}")
        panel_label(a, l)
    gaps = [f"{m} at N = {n}" for m in methods_in(g) for n in Ns
            if g[(g["method"] == m) & (g["N_items"] == n) & (g["p"] == pmax)].empty]
    if gaps:
        axs[0][0].plot([], [], " ", label="gap = not simulated (memory cap): " + ", ".join(gaps))
    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=2)
    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=2)
    save(fig, "Fig2_gradient_variance", g.drop(columns=["var_per_param", "param_indices"], errors="ignore"))

    # Fig 3: DLA dimension
    dl = pd.read_csv(d / "dla_dimensions.csv").dropna(subset=["dla_dim"])
    cases = list(dict.fromkeys(dl["case"]))
    ms = methods_in(dl)
    fig, a = plt.subplots(figsize=(ONE, 2.4))
    w = 0.8 / len(ms)
    x = np.arange(len(cases))
    for j, m in enumerate(ms):
        h = dl[dl["method"] == m].set_index("case").reindex(cases)
        bars = a.bar(x + (j - (len(ms) - 1) / 2) * w, h["fraction_of_u(D)"], w * 0.92, color=COLOR[m], label=LABEL[m])
        for b, (dim, D) in zip(bars, zip(h["dla_dim"], h["space_dim"])):
            if not np.isnan(dim):
                a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{int(dim)}/{int(D) ** 2}",
                       ha="center", va="bottom", fontsize=5.5, rotation=90)
    a.set_xticks(x, [f"groups {c}" for c in cases])
    a.set(ylabel="DLA dimension / dim u(D)", ylim=(0, 1.35))
    allrows = pd.read_csv(d / "dla_dimensions.csv")
    miss = allrows[allrows["dla_dim"].isna()]
    if len(miss):
        a.text(0.0, 1.02, "not computed: " + "; ".join(f"{r.method} {r.case} ({r.note})" for r in miss.itertuples()),
               transform=a.transAxes, fontsize=6, color=GREY)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "Fig3_dla_dimension", dl)

    # Fig 4: depth sweep
    ds = pd.read_csv(d / "depth_sweep.csv")
    fig, a = plt.subplots(figsize=(ONE, 2.4))
    rows = []
    fl = log_floor(ds["p_opt"])
    zeros = False
    for m in methods_in(ds):
        h = ds[ds["method"] == m].groupby("p")["p_opt"]
        med, q1, q3 = h.median(), h.quantile(0.25), h.quantile(0.75)
        pos = med > 0
        a.errorbar(med.index[pos], med.values[pos], yerr=np.vstack([(med - q1.clip(lower=fl))[pos], (q3 - med)[pos]]),
                   color=COLOR[m], marker=MARKER[m], capsize=2, label=LABEL[m], zorder=ZORDER[m])
        zeros |= mark_zeros(a, med.index, med.values, fl, COLOR[m])
        rows.append(pd.DataFrame({"method": m, "p": med.index, "median": med.values, "q1": q1.values, "q3": q3.values}))
    a.set_yscale("log")
    a.set_ylim(bottom=fl)
    if zeros:
        a.scatter([], [], marker="v", s=22, facecolors="white", edgecolors="#222222", label="median = 0 (drawn at axis floor)")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    sz = cfg["trainability"]["depth_sweep"]["instance"]["sizes"]
    a.set(xlabel="QAOA depth p", ylabel="P(optimum), median and IQR over seeds", title=f"synthetic instance, groups {sz}; training CVaR α = {cfg['methods']['common']['cvar_alpha']:g}")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    save(fig, "Fig4_depth_sweep", pd.concat(rows))


# ------------------------------------------------------------------ Fig 5: simulated noise (3.3)
def fig_noise(suffix="", name="Fig5_noise_sensitivity"):
    d = folder(f"3_3_noise_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"{name} (3.3{' ' + suffix if suffix else ''})")
    n = pd.read_csv(d / "noise_sweep.csv")
    alpha = json.loads((d / "summary.json").read_text()).get("cvar_alpha", cfg["methods"]["common"]["cvar_alpha"])
    ro = cfg["noise"]["readout_flip"]
    n = n[(n["readout"] == ro) | (n["p2"] == 0)].sort_values("p2")
    p2s = sorted(n["p2"].unique())
    pos = {v: i for i, v in enumerate(p2s)}
    fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.5))
    for m in methods_in(n):
        h = n[n["method"] == m]
        x = h["p2"].map(pos)
        axs[0].plot(x, h["frac_feasible"], color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=ZORDER[m])
        axs[1].plot(x, h["frac_optimal"], color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=ZORDER[m])
        if h["postselect_frac_optimal"].notna().any():
            axs[1].plot(x, h["postselect_frac_optimal"], color=COLOR[m], marker=MARKER[m], linestyle="--",
                        markerfacecolor="white", label=f"{m} after symmetry post-selection")
    for a, yl, l in zip(axs, ["fraction of feasible shots", "fraction of optimal shots"], "ab"):
        a.set_xticks(range(len(p2s)), [f"{v:g}" for v in p2s])
        a.set(xlabel="two-qubit depolarising rate", ylabel=yl,
              title=f"readout flip {ro:g} for rates > 0")
        panel_label(a, l)
    axs[0].set_ylim(0, 1.05)
    axs[1].legend(loc="upper center", bbox_to_anchor=(-0.1, -0.22), ncol=3)
    fig.suptitle(f"Pauli-trajectory noise; circuits trained with CVaR α = {alpha:g}", fontsize=8, y=1.02)
    save(fig, name, n)


# ------------------------------------------------------------------ Fig 6: stress scenarios (3.4)
def fig_stress(suffix="", name="Fig6_stress_scenarios"):
    d = folder(f"3_4_disruptions_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"{name} (3.4{' ' + suffix if suffix else ''})")
    alpha = json.loads((d / "summary.json").read_text()).get("cvar_alpha", cfg["methods"]["common"]["cvar_alpha"])
    s = pd.read_csv(d / "scenarios.csv")
    routes = list(cfg["routes"])
    ok = s[s["exact_route_counts"].notna()]
    bad = s[s["exact_route_counts"].isna()]["scenario"].tolist()
    ex = [ast.literal_eval(v) for v in ok["exact_route_counts"]]
    q = [ast.literal_eval(v) for v in ok["best_route_counts"]]
    ramp = ["#1c5cab", "#3987e5", "#86b6ef", "#b7d3f6", "#d6e6fb"]
    fig, a = plt.subplots(figsize=(TWO, 2.4))
    x = np.arange(len(ok))
    w = 0.8 / len(routes)
    rows = []
    for j, r in enumerate(routes):
        xs = x + (j - (len(routes) - 1) / 2) * w
        a.bar(xs, [c[r] for c in ex], w * 0.9, color=ramp[j], edgecolor="#1c5cab", linewidth=0.4, label=f"{r} (exact)")
        a.scatter(xs, [c[r] for c in q], s=10, color="black", marker="_", linewidths=1.2, zorder=3,
                  label=f"{ok['method'].iloc[0]} best sampled" if j == 0 else None)
        rows += [{"scenario": sc, "route": r, "exact": e[r], "quantum": qq[r]} for sc, e, qq in zip(ok["scenario"], ex, q)]
    a.set_xticks(x, [sc.replace("_", " ") for sc in ok["scenario"]], rotation=12)
    a.yaxis.set_major_locator(MaxNLocator(integer=True))
    a.set(ylabel="blocks selected per route")
    if bad:
        a.text(1.0, 1.02, "no certified plan: " + ", ".join(bad), transform=a.transAxes, ha="right", fontsize=7, color=GREY)
    a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    a.set_title(f"{ok['method'].iloc[0]} trained with CVaR α = {alpha:g}", loc="left", fontsize=8)
    save(fig, name, pd.DataFrame(rows))


# ------------------------------------------------------------------ Fig 7: chance-constraint validation (3.5)
def fig_chance(suffix="", name="Fig7_chance_constraint_validation"):
    d = folder(f"3_5_chance_constraint_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"{name} (3.5{' ' + suffix if suffix else ''})")
    alpha = json.loads((d / "summary.json").read_text()).get("cvar_alpha", cfg["methods"]["common"]["cvar_alpha"])
    v = pd.read_csv(d / "solutions_validation.csv")
    eps = {c["id"]: c["epsilon"] for c in cfg["risk"]["chance_constraint"]["validation_cases"]}
    cases = list(dict.fromkeys(v["case"]))
    sols = list(dict.fromkeys(v["solution"]))
    shade = {s: c for s, c in zip(sols, ["#222222", "#7a7a7a", "#c3c2b7", COLOR["M1"]])}
    mk = {s: c for s, c in zip(sols, ["o", "s", "^", MARKER["M1"]])}
    fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.6))
    x = np.arange(len(cases))
    w = 0.8 / len(sols)
    for j, s in enumerate(sols):
        h = v[v["solution"] == s].set_index("case").reindex(cases)
        xs = x + (j - (len(sols) - 1) / 2) * w
        axs[0].errorbar(xs, h["violation_rate"], yerr=[np.zeros(len(h)), h["cp95_upper"] - h["violation_rate"]],
                        fmt=mk[s], color=shade[s], capsize=2, label=s.replace("_", " "))
        okr = h["regret_median"].notna() & (h["violation_rate"] <= h.index.map(eps).values)
        axs[1].errorbar(xs[okr], h["regret_median"][okr],
                        yerr=[h["regret_median"][okr] - h["regret_q1"][okr], h["regret_q3"][okr] - h["regret_median"][okr]],
                        fmt=mk[s], color=shade[s], capsize=2)
    for i, c in enumerate(cases):
        axs[0].hlines(eps[c], i - 0.45, i + 0.45, color="#c0392b", linewidth=1.0, label="ε" if i == 0 else None)
    axs[0].set_yscale("log")
    axs[0].set(ylabel="violation rate (validation), with one-sided\n95% Clopper–Pearson upper bound")
    axs[1].set(ylabel="regret vs ex-post optimum\n(median and IQR over met scenarios)",
               title="shown only for solutions with violation rate ≤ ε")
    labels = [c.replace("_sigma", "\nσ=").replace("_eps", ", ε=").replace("student_t4", "Student-t ν=4").replace("gaussian", "Gaussian") for c in cases]
    for a, l in zip(axs, "ab"):
        a.set_xticks(x, labels)
        panel_label(a, l)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=5)
    fig.suptitle(f"M1 trained with CVaR α = {alpha:g}", fontsize=8, y=1.02)
    save(fig, name, v)


# ------------------------------------------------------------------ Fig 8: cost-emissions trade-off (3.6)
def fig_pareto(suffix="", name="Fig8_cost_emissions_tradeoff"):
    d = folder(f"3_6_pareto_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"{name} (3.6{' ' + suffix if suffix else ''})")
    alpha = json.loads((d / "summary.json").read_text()).get("cvar_alpha", cfg["methods"]["common"]["cvar_alpha"])
    c = pd.read_csv(d / "all_feasible_configurations.csv")
    sw = pd.read_csv(d / "weight_sweep.csv")
    cx, ey = c["cost_usd"] / 1e6, c["emis_kg"] / 1e6
    fig, a = plt.subplots(figsize=(ONE, 2.8))
    a.scatter(cx, ey, s=14, color="#c3c2b7", label="feasible configurations", zorder=2)
    nd = c[c["non_dominated"]].sort_values("cost_usd")
    a.plot(nd["cost_usd"] / 1e6, nd["emis_kg"] / 1e6, color="#222222", marker="o", markersize=4, linewidth=1,
           label="non-dominated (cost, emissions)", zorder=3)
    nd3 = c[c["non_dominated_cost_emis_risk"] & ~c["non_dominated"]]
    a.scatter(nd3["cost_usd"] / 1e6, nd3["emis_kg"] / 1e6, s=26, facecolors="none", edgecolors="#222222", marker="D",
              linewidths=0.8, label="non-dominated only with risk", zorder=3)
    ue = sorted(set(sw["exact_config"]))
    a.scatter(cx[ue], ey[ue], s=70, facecolors="none", edgecolors="#222222", marker="s", linewidths=1.0,
              label="weighted-sum optima (exact)", zorder=4)
    m = sw["method"].iloc[0]
    uq = sorted(set(sw["q_modal_config"]))
    a.scatter(cx[uq], ey[uq], s=16, color=COLOR[m], marker=MARKER[m], label=f"{m} modal best sampled", zorder=5)
    a.set(xlabel="cost (US$ million per month)", ylabel="upstream transport emissions\n(kt CO₂e per month)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    a.set_title(f"{m} trained with CVaR α = {alpha:g}", fontsize=8)
    save(fig, name, c)


# ------------------------------------------------------------------ Fig 9: Benchmark B (each alpha)
def fig_suite(suffix, tag):
    d = folder(f"B_synthetic_suite_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"Fig9{tag} (Benchmark B{suffix})")
    q = pd.read_csv(d / "quantum_runs.csv")
    alpha = float(q["cvar_alpha"].iloc[0])
    per = q.groupby(["size", "instance", "method", "p"]).agg(p_opt=("p_opt", "median"), p_feas=("p_feasible", "median"),
                                                             uni=("uniform_p_opt", "first")).reset_index()
    sizes = list(dict.fromkeys(per["size"]))
    zeros = False
    fig, axs = plt.subplots(2, len(sizes), figsize=(TWO, 4.4), squeeze=False, sharex=True)
    rows = []
    for j, sz in enumerate(sizes):
        for m in methods_in(per):
            h = per[(per["size"] == sz) & (per["method"] == m)]
            if h.empty:
                continue
            fl = log_floor(per[per["size"] == sz]["p_opt"])
            for i, col in enumerate(["p_opt", "p_feas"]):
                g = h.groupby("p")[col]
                med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
                if col == "p_opt":
                    pos = med > 0
                    axs[i][j].plot(med.index[pos], med.values[pos], color=COLOR[m], marker=MARKER[m], label=LABEL[m],
                                   zorder=ZORDER[m])
                    axs[i][j].fill_between(med.index, q1.clip(lower=fl).values, q3.clip(lower=fl).values, color=COLOR[m],
                                           alpha=0.15, linewidth=0)
                    zeros |= mark_zeros(axs[i][j], med.index, med.values, fl, COLOR[m])
                else:
                    axs[i][j].plot(med.index, med.values, color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=ZORDER[m])
                    axs[i][j].fill_between(med.index, q1.values, q3.values, color=COLOR[m], alpha=0.15, linewidth=0)
                rows.append(pd.DataFrame({"size": sz, "method": m, "metric": col, "p": med.index, "median": med.values,
                                          "q1": q1.values, "q3": q3.values, "instances": g.count().values}))
        u = per[(per["size"] == sz) & (per["method"] == "M1")]["uni"].median()
        if not np.isnan(u):
            axs[0][j].axhline(u, color=GREY, linestyle="--", linewidth=0.9, label="uniform over MPSX reachable states")
        n_inst = per[per["size"] == sz]["instance"].nunique()
        n_m = per[per["size"] == sz].groupby("method")["instance"].nunique()
        notes = [f"{m} on {int(v)}" for m, v in n_m.items() if v < n_inst]
        notes += [f"{m} not run" for m in methods_in(per) if m not in n_m.index]
        axs[0][j].set_title(f"groups {sz}, {n_inst} instances" + (f"\n({'; '.join(notes)})" if notes else ""))
        axs[0][j].set_yscale("log")
        axs[0][j].set_ylim(log_floor(per[per["size"] == sz]["p_opt"]), 1.5)
        axs[1][j].set_ylim(0, 1.05)
        axs[1][j].set_xlabel("QAOA depth p")
        axs[1][j].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0][0].set_ylabel("P(optimum)\nmedian, IQR over instances")
    axs[1][0].set_ylabel("P(feasible)\nmedian, IQR over instances")
    for k, a in enumerate(axs.ravel()):
        panel_label(a, "abcdefgh"[k])
    if zeros:
        axs[0][0].scatter([], [], marker="v", s=22, facecolors="white", edgecolors="#222222", label="median = 0 (drawn at axis floor)")
    hd, lb = axs[0][0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.suptitle(f"CVaR α = {alpha:g}", fontsize=8, y=1.0)
    save(fig, f"Fig9{tag}_synthetic_suite_alpha{alpha:g}", pd.concat(rows))


# ------------------------------------------------------------------ Fig 10, Fig 11, Tables S1-S2: Qiskit companion run
def fig_qiskit():
    d = folder(f"C_qiskit_{PROF}")
    if d is None:
        return skipped.append("Fig10-11, Tables S1-S2 (run_C_qiskit.py)")
    d1 = folder(f"C_qiskit_{PROF}_alpha1")
    cv = pd.concat([pd.read_csv(x / "cross_validation.csv") for x in [d, d1] if x is not None])
    okc = cv[cv["status"].isin(["PASS", "FAIL"])]
    if len(okc):
        t1 = okc.groupby(["source", "method"]).agg(circuits=("status", "size"), failures=("status", lambda s: int((s == "FAIL").sum())),
                                                   max_abs_diff_p_opt=("abs_diff_p_opt", "max"),
                                                   max_abs_diff_p_feasible=("abs_diff_p_feasible", "max"),
                                                   max_qubits=("n_qubits", "max")).reset_index()
        for dd, lab in [(d, f"CVaR alpha {cfg['methods']['common']['cvar_alpha']:g}"), (d1, "CVaR alpha 1")]:
            ac = dd / "aer_circuit_check.csv" if dd is not None else None
            if ac is None or not ac.exists() or ac.stat().st_size <= 1:
                continue
            a = pd.read_csv(ac)
            t1 = pd.concat([t1, pd.DataFrame({"source": f"3_3 noise circuits, {lab} (transpiled, sent to Aer)", "method": a["method"],
                                              "circuits": 1, "failures": (a["status"] == "FAIL").astype(int),
                                              "max_abs_diff_p_opt": a["abs_diff"], "max_abs_diff_p_feasible": np.nan,
                                              "max_qubits": np.nan})])
        t1.to_csv(OUT / "TableS1_qiskit_cross_validation.csv", index=False)
        made.append("TableS1_qiskit_cross_validation")
        print("  wrote TableS1_qiskit_cross_validation.csv")
    rs = pd.read_csv(d / "circuit_resources_summary.csv")
    rs.to_csv(OUT / "TableS2_circuit_resources.csv", index=False)
    made.append("TableS2_circuit_resources")
    print("  wrote TableS2_circuit_resources.csv")
    r = pd.read_csv(d / "circuit_resources.csv")
    r = r[(r["status"] == "ok") & (r["p"] == 1) & (r["size"] != "steel")]
    if len(r):
        fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.5))
        Ns = sorted(r["N_items"].unique())
        rows = []
        for m in methods_in(r):
            h = r[r["method"] == m].groupby("N_items")
            for a, col in zip(axs, ["two_qubit", "depth"]):
                med = h[col].median().reindex(Ns)
                a.plot(Ns, med.values, color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=ZORDER[m])
            rows.append(h[["two_qubit", "depth", "n_qubits"]].median().reset_index().assign(method=m))
        for a, yl, l in zip(axs, ["CX gates per layer (transpiled)", "circuit depth per layer (transpiled)"], "ab"):
            a.set(xlabel="items N", ylabel=yl, title="median over instances, basis {cx, rz, sx, x}")
            a.set_xticks(Ns)
            panel_label(a, l)
        m12 = r[r["method"].isin(["M1", "M2"])].pivot_table(index=["instance", "p"], columns="method", values="two_qubit")
        if {"M1", "M2"} <= set(m12.columns) and (m12["M1"] == m12["M2"]).all():
            axs[0].text(0.02, 0.97, "M1 and M2 use identical gates (M2 ties the mixer angles)", transform=axs[0].transAxes,
                        fontsize=6, color=GREY, va="top")
        axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=2)
        save(fig, "Fig10_circuit_resources", pd.concat(rows))


def fig_aer(suffix="", name="Fig11_aer_vs_trajectory_noise"):
    d = folder(f"C_qiskit_{PROF}{suffix}")
    if d is None:
        return skipped.append(f"{name} (run_C_qiskit.py{' --alpha 1' if suffix else ''})")
    alpha = json.loads((d / "summary.json").read_text()).get("cvar_alpha", cfg["methods"]["common"]["cvar_alpha"])
    an = d / "aer_noise_sweep.csv"
    tr = RES / f"3_3_noise_{PROF}{suffix}" / "noise_sweep.csv"
    if an.exists() and an.stat().st_size > 1 and tr.exists():
        a_ = pd.read_csv(an)
        t_ = pd.read_csv(tr)
        ro = cfg["noise"]["readout_flip"]
        sel = lambda x: x[(x["readout"] == ro) | (x["p2"] == 0)].sort_values("p2")
        a_, t_ = sel(a_), sel(t_)
        p2s = sorted(a_["p2"].unique())
        pos = {v: i for i, v in enumerate(p2s)}
        fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.5))
        for m in methods_in(a_):
            ha, ht = a_[a_["method"] == m], t_[t_["method"] == m]
            for ax, col in zip(axs, ["frac_feasible", "frac_optimal"]):
                ax.plot(ha["p2"].map(pos), ha[col], color=COLOR[m], marker=MARKER[m], label=f"{m} Qiskit Aer", zorder=ZORDER[m])
                ax.plot(ht["p2"].map(pos), ht[col], color=COLOR[m], marker=MARKER[m], linestyle="--", markerfacecolor="white",
                        label=f"{m} Pauli trajectories")
        for ax, yl, l in zip(axs, ["fraction of feasible shots", "fraction of optimal shots"], "ab"):
            ax.set_xticks(range(len(p2s)), [f"{v:g}" for v in p2s])
            ax.set(xlabel="two-qubit depolarising rate", ylabel=yl, title=f"readout flip {ro:g} for rates > 0")
            panel_label(ax, l)
        axs[0].set_ylim(0, 1.05)
        axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=3)
        fig.suptitle(f"circuits trained with CVaR α = {alpha:g}; rate applied per native CX (Aer) "
                     "or per logical two-qubit gate (trajectories)", fontsize=8, y=1.02)
        save(fig, name, pd.concat([a_.assign(model="qiskit_aer"), t_.assign(model="pauli_trajectories")]))


# ------------------------------------------------------------------ Table S3: steel decision sections, both alphas
def table_steel_alpha():
    rows = []
    for suffix in ["", "_alpha1"]:
        d4, d5, d6 = (folder(f"{x}_{PROF}{suffix}") for x in ("3_4_disruptions", "3_5_chance_constraint", "3_6_pareto"))
        for d, sec in [(d4, "3.4"), (d5, "3.5"), (d6, "3.6")]:
            if d is None:
                continue
            al = json.loads((d / "summary.json").read_text()).get("cvar_alpha", cfg["methods"]["common"]["cvar_alpha"])
            if sec == "3.4":
                t = pd.read_csv(d / "scenarios.csv")
                t = t[t["exact_route_counts"].notna()]
                for _, r in t.iterrows():
                    rows.append({"section": sec, "cvar_alpha": al, "case": r["scenario"], "plan_matches_exact":
                                 r["exact_route_counts"] == r["best_route_counts"], "p_opt_median": r["p_opt_median"]})
            elif sec == "3.5":
                t = pd.read_csv(d / "solutions_validation.csv")
                for case, g in t.groupby("case", sort=False):
                    e = g[g["solution"] == "exact_safe_count"]["route_counts"].iloc[0]
                    m = g[g["solution"] == "M1_committed"]
                    rows.append({"section": sec, "cvar_alpha": al, "case": case, "plan_matches_exact": m["route_counts"].iloc[0] == e,
                                 "certified": bool(m["certified_at_eps"].iloc[0]), "violation_rate": m["violation_rate"].iloc[0]})
            else:
                t = pd.read_csv(d / "weight_sweep.csv")
                for _, r in t.iterrows():
                    rows.append({"section": sec, "cvar_alpha": al, "case": f"w_cost={r['w_cost']:g}, w_emit={r['w_emit']:g}",
                                 "plan_matches_exact": bool(r["q_matches_exact"]), "p_opt_median": r["q_p_opt_median"]})
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "TableS3_steel_sections_by_alpha.csv", index=False)
        made.append("TableS3_steel_sections_by_alpha")
        print("  wrote TableS3_steel_sections_by_alpha.csv")


print(f"Figures for profile '{PROF}' (parameter file {p.sha256[:12]}...) -> {OUT}")
fig_steel(); fig_steel("_alpha1", "Fig1b_steel_quantum_vs_depth_alpha1"); fig_gradvar(); fig_noise(); fig_noise("_alpha1", "Fig5b_noise_sensitivity_alpha1"); fig_stress(); fig_stress("_alpha1", "Fig6b_stress_scenarios_alpha1")
fig_chance(); fig_chance("_alpha1", "Fig7b_chance_constraint_validation_alpha1")
fig_pareto(); fig_pareto("_alpha1", "Fig8b_cost_emissions_tradeoff_alpha1"); table_steel_alpha()
fig_suite("", "a"); fig_suite("_alpha1", "b"); fig_qiskit(); fig_aer(); fig_aer("_alpha1", "Fig11b_aer_vs_trajectory_noise_alpha1")
zp = shutil.make_archive(str(RES / f"figures_{PROF}"), "zip", OUT)
print(f"\nMade {len(made)} figures; zip: {zp}")
if skipped:
    print("Skipped (result folder missing or incomplete; run its script first):", "; ".join(skipped))
