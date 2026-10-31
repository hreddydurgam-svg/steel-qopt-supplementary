"""
Cost and emission sensitivities declared in testbed_params_v18.json ('sensitivity' fields).

Each case changes one input, recomputes the per-tonne cost (TLC) and emissions of every route with the
same loader formulas (load_params_v18.load on a modified copy of the parameters), rebuilds the steel case
(chance-constraint calibration, safe counts k) and solves it exactly by enumeration. The quantum and
heuristic methods are not re-run.

The parameter file is not changed; each modified copy is written to a temporary folder and deleted.

Output: results/cost_sensitivity/route_values.csv, optimum.csv
Usage:  python cost_sensitivity.py
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from load_params_v18 import load
from qopt.common import banner, results_dir
from qopt.instance import solve_exact
from qopt.steel import counts_from_xint, steel_case

base = load()
raw0 = base.raw
fx = base.fx
out = results_dir("cost_sensitivity")
banner("COST AND EMISSION SENSITIVITIES (declared in the parameter file)", base)


def case_list():
    r = raw0["routes"]
    ef = raw0["emission_factors_g_tkm"]
    cases = [("base", "base case", lambda d: None)]
    for v in r["H1"]["sea_freight_usd"]["sensitivity"]:
        cases.append((f"sea_freight_{v:g}", f"H1 sea freight US${v:g}/t (base US${r['H1']['sea_freight_usd']['value']:g})",
                      lambda d, v=v: d["routes"]["H1"]["sea_freight_usd"].__setitem__("value", v)))
    v = r["H1"]["port_handling_inr"]["sensitivity_usd_total"]
    cases.append(("port_total_usd", f"H1 port handling total US${v:g}/t (base: wharfage Rs {r['H1']['port_handling_inr']['value']}/t)",
                  lambda d, v=v: d["routes"]["H1"]["port_handling_inr"].__setitem__("value", v * fx)))
    cases.append(("gst_included", "rail GST 5% included in rail cost (base: excluded)",
                  lambda d: d["rail_common"].__setitem__("gst_treatment", "include")))
    for v in r["H1"]["sea_distance_km"]["sensitivity"]:
        cases.append((f"sea_km_{v:g}", f"H1 sea distance {v:g} km (base {r['H1']['sea_distance_km']['value']:g})",
                      lambda d, v=v: d["routes"]["H1"]["sea_distance_km"].__setitem__("value", v)))
    for v in ef["sea"]["sensitivity"]:
        cases.append((f"sea_ef_{v:g}", f"sea emission factor {v:g} g/tkm (base {ef['sea']['value']:g})",
                      lambda d, v=v: d["emission_factors_g_tkm"]["sea"].__setitem__("value", v)))
    for v in ef["truck"]["sensitivity"]:
        cases.append((f"truck_ef_{v:g}", f"truck emission factor {v:g} g/tkm (base {ef['truck']['value']:g})",
                      lambda d, v=v: d["emission_factors_g_tkm"]["truck"].__setitem__("value", v)))
    for v in r["R1"]["hub_ef_kg_t"]["sensitivity"]:
        cases.append((f"R1_hub_{v:g}", f"R1 hub factor {v:g} kg/t (base {r['R1']['hub_ef_kg_t']['value']:g})",
                      lambda d, v=v: d["routes"]["R1"]["hub_ef_kg_t"].__setitem__("value", v)))
    return cases


rows, opt_rows = [], []
with tempfile.TemporaryDirectory() as tmp:
    for cid, desc, change in case_list():
        raw = copy.deepcopy(raw0)
        change(raw)
        path = Path(tmp) / f"{cid}.json"
        path.write_text(json.dumps(raw))
        p = load(path)
        for r in p.routes:
            rows.append({"case": cid, "route": r, "TLC_usd_per_t": p.tlc[r], "E_kg_per_t": p.emis[r],
                         "dTLC_usd_per_t": p.tlc[r] - base.tlc[r], "dE_kg_per_t": p.emis[r] - base.emis[r]})
        sc = steel_case(p)
        acc = sc["cal"]["accepted"]
        if sc["inst"] is None:
            opt_rows.append({"case": cid, "description": desc, "k": None, "optimal_route_counts": "no certified k"})
            print(f"{cid:<18} no certified safe-count k")
            continue
        inst, rm = sc["inst"], sc["rm"]
        ex = solve_exact(inst)
        counts = sorted({tuple(c) for c in counts_from_xint(rm, ex.opt_set).tolist()})
        x = (ex.opt_set[0] >> np.arange(inst.N)) & 1
        c, e, rk = inst.components(x)
        opt_rows.append({"case": cid, "description": desc, "k": acc["k"],
                         "optimal_route_counts": "; ".join(str(dict(zip(rm.routes, cc))) for cc in counts),
                         "f_opt": ex.f_opt, "cost_usd_per_month": c, "emissions_kg_per_month": e, "risk": rk})
        print(f"{cid:<18} k {acc['k']}  optimum {counts}  cost US${c:,.0f}/month  emissions {e:,.0f} kg/month")

rv = pd.DataFrame(rows)
rv.to_csv(out / "route_values.csv", index=False)
op = pd.DataFrame(opt_rows)
b = op.iloc[0]
op["optimum_changed_vs_base"] = op["optimal_route_counts"] != b["optimal_route_counts"]
op["k_changed_vs_base"] = op["k"].astype(str) != str(b["k"])
op.to_csv(out / "optimum.csv", index=False)
print("\nPer-tonne changes vs base (only routes that change):")
print(rv[(rv.dTLC_usd_per_t.abs() > 1e-12) | (rv.dE_kg_per_t.abs() > 1e-12)].round(4).to_string(index=False))
print("\nExact optimum per case:")
print(op[["case", "k", "optimal_route_counts", "optimum_changed_vs_base", "k_changed_vs_base"]].to_string(index=False))
single = [m for m in base.demand if sum(raw0["routes"][r]["material"] == m for r in base.routes) == 1]
print(f"\nMaterials with a single route ({', '.join(single)}): their block counts equal k, so changes to those routes "
      "can change the optimum only through k.")
print(f"\n-> {out}")
