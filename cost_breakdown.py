"""
Per-tonne cost and emission breakdown of every route, from testbed_params_v18.json.

Each row gives the inputs, the formula, the result in Rs/t and US$/t (or kg CO2e/t), the label
and source recorded in the parameter file, and numbered references (REFERENCES below). The
script checks that the components add up to the values used by the models (load_params_v18.load()).

"Checked" states whether the quoted value was found at the URL (checked Oct 2026).

Output: results/cost_breakdown/cost_breakdown.csv, emission_breakdown.csv, references.csv
Usage:  python cost_breakdown.py
"""
import pandas as pd

from load_params_v18 import G_PER_KG, load
from qopt.common import banner, results_dir

REFERENCES = {
    1: ("Board of Governors of the Federal Reserve System, Indian Rupees to U.S. Dollar Spot Exchange Rate "
        "(AEXINUS), annual, via FRED, Federal Reserve Bank of St. Louis",
        "https://fred.stlouisfed.org/series/AEXINUS",
        "yes: 2024 = 83.6566"),
    2: ("Department of Industry, Science and Resources (Australia), Resources and Energy Quarterly, December 2025, "
        "Table 4.2, p. 30",
        "https://www.industry.gov.au/sites/default/files/2025-12/resources-and-energy-quarterly-december-2025.pdf",
        "yes: hard coking coal FOB Australia East Coast, 2024, nominal US$242/t"),
    3: ("EY-Parthenon and Indian Steel Association, India's coking coal strategy: Building resilience through "
        "innovation, sustainability and policy, September 2025, p. 21",
        "https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/insights/mining-metals/doucments/"
        "ey-india-s-coking-coal-strategy.pdf",
        "yes: freight differentials and currency fluctuations 'can widen landed-cost bands by US$15-25 per ton'. "
        "Not a freight-rate quote; US$22/t is an assumption inside this range"),
    4: ("Karaikal Port Private Limited, Berthing Policy & Tariff Structure BPTS/KPPL/04, w.e.f. 1 Oct 2024, "
        "wharfage charges (coal)",
        "https://cdn.gac.com/prod/docs/INDIA-Karaikal_BPTS-Karaikal-Port-W.e.f-01st-Oct-2024.pdf",
        "yes: 'Coal M. Ton 40/-' (copy hosted by GAC, a port agent)"),
    5: ("Indian Railways, RailSAHAY freight calculator (FOIS), KIKP -> MCRD and BAY -> MCRD, BOXN, CC+8, "
        "accessed 3 Oct 2026",
        "https://www.fois.indianrail.gov.in/RailSAHAY/pages/FrgtCalc.jsp",
        "coal: yes, author's calculator printout 7 Oct 2026 (Coking Coal (Imported), train load class 145A, "
         "797.20 + 20.00 destination terminal charge, 327 km, 59 BOXN x 68.10 t, total incl. GST Rs 3,447,600); "
         "iron ore: yes, printout 7 Oct 2026 (Iron Ore, train load class 165, 1,113.10 + 166.97 surcharge (dynamic "
         "pricing) + 104.00 development and terminal charges, 609 km, total incl. GST Rs 5,839,107). "
         "The calculator states its freight is indicative"),
    6: ("Indian Bureau of Mines, Monthly Statistics of Mineral Production, February 2024 (average sale price "
        "extract), iron ore fines, Karnataka, 62% to below 65% Fe",
        "https://ibm.gov.in/writereaddata/files/1711621865660546e9d9b64ASP_Min_Feb_2024.pdf",
        "yes: Rs 5,389/t"),
    7: ("BigMint, Chennai HMS 80:20 scrap price assessment, 2 Aug 2024 (subscription)",
        "",
        "no: not publicly accessible; not cited; listed only because the parameter-file note names it. Cite [8]"),
    8: ("BigMint, India: Chennai scrap prices inch up by INR 200/t w-o-w, 26 Sep 2024",
        "https://bigmint.co/insights/detail/india-chennai-scrap-prices-inch-up-by-inr-200-t-w-o-w-26-sep-583877",
        "yes: Chennai HMS (80:20) INR 31,800/t"),
    9: ("NITI Aayog, RMI and RMI India, Fast Tracking Freight in India: A Roadmap for Clean and Cost-Effective "
        "Goods Transport, June 2021, Exhibit 3-1, p. 31",
        "https://www.niti.gov.in/sites/default/files/2021-06/FreightReportNationalLevel.pdf",
        "yes: road cost 3.6 INR/tonne-km (rail 1.6); endnote 79 cites Aritua et al. 2018 [10]. The year 2015 in the "
         "parameter-file note is not stated in [9] or [10]"),
    10: ("Aritua, B., Havenga, J., Simpson, Z., Chiew, E.W.L., Unlocking India's Logistics Potential: The Value of "
         "Disaggregated Macroscopic Freight Flow Analysis, Policy Research Working Paper 8337, World Bank, Feb 2018",
         "https://documents1.worldbank.org/curated/en/888791518530457570/pdf/WPS8337.pdf",
         "opened: the 3.6 INR/tonne-km figure is not stated in the paper; it is cited only through [9]"),
    11: ("Project OSRM, Open Source Routing Machine (public demo server), driving routes, 3 Oct 2026",
         "https://project-osrm.org",
         "yes, re-run 7 Oct 2026: city points Chennai 80.2707E 13.0827N, Vellore 79.1325E 12.9165N, Bengaluru "
         "77.5946E 12.9716N to plant 77.9200E 11.8115N (inside the OpenStreetMap 'JSW Steel Plant' area): "
         "337.2, 201.1, 168.8 km"),
    12: ("searoute (Python package) v1.6.0, shortest sea route, 3 Oct 2026",
         "https://pypi.org/project/searoute/",
         "yes, re-run 7 Oct 2026 with searoute 1.6.0, Hay Point Coal Terminal 149.2930E 21.2768S to Karaikal Port "
         "79.8457E 10.8388N (OpenStreetMap): 9,295.0 km (check_sea_distance.py)"),
    13: ("Smart Freight Centre, GLEC Framework for Logistics Emissions Accounting and Reporting v3.2, October 2025, "
         "Table 3, p. 96 (hubs) and Table 14, p. 107 (sea)",
         "https://smartfreightcentre.org/news/13311209 (PDF linked from this page)",
         "yes: Table 14 WTW with 15% DAF, bulk carrier VLSFO 60,000-99,999 dwt 6.1, 100,000-199,999 dwt 4.1; "
         "Table 3 medians, ambient: transshipment 1.2, storage + transshipment 2.7"),
    14: ("Smart Freight Centre and TCI-IIMB Supply Chain Sustainability Lab, India Default GHG Emission Values V1.0, "
         "May 2025",
         "https://smart-freight-centre-media.s3.amazonaws.com/documents/India_Default_GHG_Emission_Values_SFC_INDIA_v1_2.pdf",
         "yes: WTW rail 10.6, HCV-3 (GVW 30-50 t) 66.3, tractor-trailer (GVW 30-60 t) 55.1 g CO2e/tkm"),
    15: ("Kamboj, P., Tongia, R., Indian Railways and Coal: An Unsustainable Interdependency, Brookings India, "
         "July 2018, Section 2.1, Fig. 8, Appendix Table 17",
         "https://www.brookings.edu/wp-content/uploads/2018/07/Railways-and-coal.pdf",
         "yes: coal class 145, iron ore class 165, GST 5% on freight (supporting)"),
    16: ("Ministry of Finance, Economic Survey 2025-26, Statistical Appendix, Table 1.26 Operations of Indian Railways",
         "https://www.indiabudget.gov.in/economicsurvey/doc/stat/tab1.26.pdf",
         "yes: average rate 173.3 paise/tonne-km, average lead 601 km, 2024-25 provisional (supporting)"),
}
# references per component; R3/R4 scrap price is the Chennai price (assumption)
REFS = {"fx": [1], "rake payload": [5], "rail GST": [5, 15], "truck rate": [9, 10],
        ("H1", "purchase price"): [2], ("H1", "sea freight"): [3], ("H1", "port wharfage"): [4],
        ("H1", "rail freight"): [5, 15, 16], ("R1", "purchase price"): [6], ("R1", "rail freight"): [5, 15, 16],
        ("scrap", "purchase price"): [8], ("scrap", "road freight"): [9, 10, 11],
        "sea leg": [13, 12], "rail leg": [14, 5], "truck leg": [14, 11], "hub": [13]}


def tag(ids):
    return "".join(f"[{i}]" for i in ids)

p = load()
raw = p.raw
fx = p.fx
rc, rd = raw["rail_common"], raw["road_common"]
ef = raw["emission_factors_g_tkm"]
rake_t = rc["wagons_per_rake"] * rc["pcc_t_per_wagon"]
gst_div = (1 + rc["gst_rate"]) if rc["gst_treatment"] == "exclude" else 1.0
out = results_dir("cost_breakdown")
banner("COST AND EMISSION BREAKDOWN PER TONNE", p)

common = [
    ("fx", f"{fx} Rs per US$", raw["fx_inr_per_usd"]["label"], raw["fx_inr_per_usd"]["source"]),
    ("rake payload", f"{rc['wagons_per_rake']} wagons x {rc['pcc_t_per_wagon']} t = {rake_t:,.1f} t", rc["label"], rc["source"]),
    ("rail GST", f"{rc['gst_rate']:.0%}, treatment: {rc['gst_treatment']}", rc["gst_label"], f"{rc['gst_source']}; {rc['gst_reason']}"),
    ("truck rate", f"{rd['truck_rate_inr_tkm']} Rs/tkm", rd["label"], rd["source"]),
]
print("Common inputs")
for name, val, lab, src in common:
    print(f"  {name:<13} {val:<40} {tag(REFS[name]):<8} [{lab}] {src}")

rows, erows = [], []
for r, d in raw["routes"].items():
    comp = []
    if "price_usd" in d:
        v = d["price_usd"]
        comp.append(("purchase price", f"US$ {v['value']}", v["value"] * fx, v["value"], v["label"], v["source"]))
    else:
        v = d["price_inr"]
        comp.append(("purchase price", f"Rs {v['value']} / {fx}", v["value"], v["value"] / fx, v["label"],
                     v["source"] + ("; " + v["basis"] if "basis" in v else "")))
    if "sea_freight_usd" in d:
        v = d["sea_freight_usd"]
        comp.append(("sea freight", f"US$ {v['value']}", v["value"] * fx, v["value"], v["label"], v["source"]))
    if "port_handling_inr" in d:
        v = d["port_handling_inr"]
        comp.append(("port wharfage", f"Rs {v['value']} / {fx}", v["value"], v["value"] / fx, v["label"], v["source"]))
    if "rail" in d:
        v = d["rail"]
        inr = v["rake_total_inr_incl_gst"] / gst_div / rake_t
        comp.append(("rail freight", f"Rs {v['rake_total_inr_incl_gst']:,} / {gst_div:g} / {rake_t:,.1f} t "
                     f"({v['distance_km']} km, class {v['rate_class']})", inr, inr / fx, v["label"], v["source"]))
    if "road_km" in d:
        v = d["road_km"]
        inr = rd["truck_rate_inr_tkm"] * v["value"]
        comp.append(("road freight", f"{rd['truck_rate_inr_tkm']} Rs/tkm x {v['value']} km", inr, inr / fx,
                     f"{rd['label']} (rate), {v['label']} (km)", f"rate: {rd['source']}; km: {v['source']}"))
    total = sum(c[3] for c in comp)
    assert abs(total - p.tlc[r]) < 1e-9, (r, total, p.tlc[r])
    print(f"\n{r} {d['name']} ({d['material']})")
    for name, formula, inr, usd, lab, src in comp:
        refs = tag(REFS.get((r, name)) or REFS[(d["material"], name)])
        print(f"  {name:<15} {formula:<62} {inr:>10,.2f} Rs/t {usd:>9.4f} US$/t  {refs:<12} [{lab}] {src}")
        rows.append({"route": r, "material": d["material"], "component": name, "formula": formula,
                     "inr_per_t": inr, "usd_per_t": usd, "share_of_total": usd / total, "label": lab, "source": src,
                     "refs": refs})
    print(f"  {'TOTAL':<15} {'':<62} {total * fx:>10,.2f} Rs/t {total:>9.4f} US$/t  (= value used by the models)")
    rows.append({"route": r, "material": d["material"], "component": "TOTAL", "formula": "sum of components",
                 "inr_per_t": total * fx, "usd_per_t": total, "share_of_total": 1.0, "label": "D", "source": "",
                 "refs": ""})
    if "rail" in d:
        v = d["rail"]
        inr = v["rake_total_inr_incl_gst"] / gst_div / rake_t
        print(f"  (reference only, not used: base rate Rs {v['base_rate_inr_t']}/t; surcharges and terminal/development "
              f"charges Rs {inr - v['base_rate_inr_t']:.2f}/t, {inr / v['base_rate_inr_t'] - 1:.1%}; see [5])")

    # emissions
    e_tot = 0.0
    for mode, key in d["legs_ef"]:
        km = d
        for k in key.split("."):
            km = km[k]
        km = km["value"] if isinstance(km, dict) else km
        kg = ef[mode]["value"] * km / G_PER_KG
        e_tot += kg
        erows.append({"route": r, "component": f"{mode} leg", "formula": f"{ef[mode]['value']} g/tkm x {km} km / 1000",
                      "kgco2e_per_t": kg, "label": ef[mode]["label"], "source": ef[mode]["source"],
                      "refs": tag(REFS[f"{mode} leg"])})
    h = d["hub_ef_kg_t"]
    e_tot += h["value"]
    erows.append({"route": r, "component": "hub", "formula": f"{h['value']} kg/t", "kgco2e_per_t": h["value"],
                  "label": h["label"], "source": h["source"], "refs": tag(REFS["hub"]) if h["value"] else ""})
    assert abs(e_tot - p.emis[r]) < 1e-9, (r, e_tot, p.emis[r])
    erows.append({"route": r, "component": "TOTAL", "formula": "sum of components", "kgco2e_per_t": e_tot,
                  "label": "D", "source": "", "refs": ""})

pd.DataFrame(rows).to_csv(out / "cost_breakdown.csv", index=False)
edf = pd.DataFrame(erows)
edf.to_csv(out / "emission_breakdown.csv", index=False)
print("\nEmissions (kg CO2e/t)")
print(edf[["route", "component", "formula", "kgco2e_per_t", "refs"]].to_string(index=False))
pd.DataFrame([{"ref": k, "reference": c, "url": u, "checked": chk} for k, (c, u, chk) in REFERENCES.items()]
             ).to_csv(out / "references.csv", index=False)
print("\nReferences (checked = quoted value found at the URL, Oct 2026)")
for k, (c, u, chk) in REFERENCES.items():
    print(f"[{k}] {c}. {u or '(no public URL)'}\n     checked: {chk}")
print(f"\nAll route totals equal the model values. -> {out}")
