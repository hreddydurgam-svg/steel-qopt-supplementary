# Supplementary code: chance-constrained quantum-classical optimisation for upstream steel supply

Supplementary Information. Reproduces every computational result in Sections 3.1–3.8 of the manuscript: the steel case (3.1–3.6), the synthetic benchmark (3.7, "Benchmark B" in the code) and the Qiskit cross-check (3.8, `run_C_qiskit.py`).

- All inputs and settings are in `testbed_params_v18.json` (SHA-256 `5f80e9b0c819670e62678f2cfabfdcbeccf80ed8b31b44400f3aca635208eb2e`), read through `load_params_v18.py`. Every script prints the SHA-256 of this file; each `summary.json` records it with the Python and package versions.
- All quantum results are exact statevector simulations on a CPU (noiseless or with simulated noise).
- `SHA256SUMS.txt` lists the SHA-256 of every file in this folder.

## 0. Map to the manuscript

The code numbers its figures 1–11 (file names); the manuscript numbers them 2–16 (Figure 1 is the workflow diagram, not a data figure).

| Manuscript | File in `figures_paper_final/` (each also as `.pdf`, `.png` and `_data.csv`) | Result folder (`results/`) |
|---|---|---|
| Figure 2 | `Fig1_steel_quantum_vs_depth.tiff` | `3_1_steel_case_paper` |
| Figure 3 | `Fig1b_steel_quantum_vs_depth_alpha1.tiff` | `3_1_steel_case_paper_alpha1` |
| Figure 4 | `Fig2_gradient_variance.tiff` | `3_2_trainability_paper` |
| Figure 5 | `Fig3_dla_dimension.tiff` | `3_2_trainability_paper` |
| Figure 6 | `Fig4_depth_sweep.tiff` | `3_2_trainability_paper` |
| Figure 7 | `Fig5_noise_sensitivity.tiff` | `3_3_noise_paper` |
| Figure 8 | `Fig5b_noise_sensitivity_alpha1.tiff` | `3_3_noise_paper_alpha1` |
| Figure 9 | `Fig6_stress_scenarios.tiff` | `3_4_disruptions_paper` |
| Supplementary Figure S1 | `Fig6b_stress_scenarios_alpha1.tiff` | `3_4_disruptions_paper_alpha1` |
| Figure 10 | `Fig7_chance_constraint_validation.tiff` | `3_5_chance_constraint_paper` |
| Supplementary Figure S2 | `Fig7b_chance_constraint_validation_alpha1.tiff` | `3_5_chance_constraint_paper_alpha1` |
| Figure 11 | `Fig8_cost_emissions_tradeoff.tiff` | `3_6_pareto_paper` |
| Supplementary Figure S3 | `Fig8b_cost_emissions_tradeoff_alpha1.tiff` | `3_6_pareto_paper_alpha1` |
| Figure 12 | `Fig9a_synthetic_suite_alpha0.05.tiff` | `B_synthetic_suite_paper` |
| Figure 13 | `Fig9b_synthetic_suite_alpha1.tiff` | `B_synthetic_suite_paper_alpha1` |
| Figure 14 | `Fig10_circuit_resources.tiff` | `C_qiskit_paper` |
| Figure 15 | `Fig11_aer_vs_trajectory_noise.tiff` | `C_qiskit_paper`, `3_3_noise_paper` |
| Figure 16 | `Fig11b_aer_vs_trajectory_noise_alpha1.tiff` | `C_qiskit_paper_alpha1`, `3_3_noise_paper_alpha1` |
| Supplementary Table S1 | `TableS1_qiskit_cross_validation.csv` | `C_qiskit_paper`, `C_qiskit_paper_alpha1` (`cross_validation.csv`) |
| Supplementary Table S2 | `TableS2_circuit_resources.csv` | `C_qiskit_paper` (`circuit_resources_summary.csv`) |
| Supplementary Table S3 | `TableS3_steel_sections_by_alpha.csv` | `3_4`–`3_6` folders, both α |

| Manuscript table | Main result files |
|---|---|
| Table 1 (routes, cost, emissions) | `results/cost_breakdown/`, `testbed_params_v18.json` |
| Table 4 (certification) | `3_1_steel_case_paper/calibration_table.csv`, `3_5_chance_constraint_paper/calibration.csv` |
| Table 5 (steel case, P(optimum)) | `3_1_steel_case_paper/quantum_summary.csv`, `quantum_runs.csv`; `3_1_steel_case_paper_alpha1/quantum_summary.csv` |
| Table 6 (gradient variance) | `3_2_trainability_paper/gradient_variance.csv` |
| Table 7 (depth sweep) | `3_2_trainability_paper/depth_sweep.csv` |
| Table 8 (noise) | `3_3_noise_paper/noise_sweep.csv`, `3_3_noise_paper_alpha1/noise_sweep.csv` |
| Table 9 (stress scenarios) | `3_4_disruptions_paper/scenarios.csv`, `quantum_runs.csv` |
| Table 10 (validation) | `3_5_chance_constraint_paper/solutions_validation.csv`, `calibration.csv` |
| Table 11 (plans meeting the safe counts) | `3_6_pareto_paper/all_feasible_configurations.csv`, `weight_sweep.csv` |
| Table 12 (synthetic benchmark) | `B_synthetic_suite_paper/quantum_runs.csv`, `instances.csv`, `paired_tests_p_opt.csv` |
| Table 13 (heuristics) | `B_synthetic_suite_paper/classical_runs.csv`, `3_1_steel_case_paper/classical_runs.csv` |
| Table 14 (circuit resources) | `C_qiskit_paper/circuit_resources.csv` |

`sources/` holds the two RailSAHAY "Expected Freight Charges" printouts dated 7 Oct 2026 (coal KIKP → MCRD; iron ore BAY → MCRD), the only record of the rail inputs (Section 6, Rail). Other sources are cited with their URLs in `testbed_params_v18.json` and `results/cost_breakdown/references.csv`; their documents are not redistributed.

Not included: the resume caches (`results/*/_checkpoint/`, about 135 MB), the per-script preview plots (`results/*/fig_*.png`) and the earlier figure render `results/figures_paper/` (same plotted data as `figures_paper_final/`; figure titles differ). Re-running a script without its cache recomputes every unit.

## 1. Contents

| File | Purpose |
|---|---|
| `testbed_params_v18.json` | Inputs and settings. Data values are labelled Q (quoted), D (derived), A (assumption) or NF (not found), with source. |
| `load_params_v18.py` | Loader: per-tonne cost and emissions, blocks (qubits), lock checks (`python load_params_v18.py`). |
| `cost_sensitivity.py` | Cost and emission sensitivities declared in the parameter file: per-tonne values and the exact optimum of the steel case for each case. |
| `check_sea_distance.py` | Recomputes the Hay Point–Karaikal sea distance with searoute 1.6.0 (`pip install searoute==1.6.0`). |
| `cost_breakdown.py` | Per-tonne cost and emission breakdown of every route: inputs, formula, Rs/t, US$/t, label, source and numbered references (full title, URL, whether the value was checked) for each component; checks that the components sum to the values used by the models. |
| `run_00_selftest.py` | Engine checks; must end with `SELFTEST: PASS`. |
| `run_3_1_steel_case.py` | 3.1 Steel case: variables, chance-constraint calibration, worked example, MILP, heuristics, quantum methods |
| `run_3_2_trainability.py` | 3.2 Gradient variance, dynamical Lie algebra dimensions, depth sweep |
| `run_3_3_noise.py` | 3.3 Simulated depolarising and readout noise; symmetry post-selection (M1, M2) |
| `run_3_4_disruptions.py` | 3.4 Stress scenarios S1–S5 |
| `run_3_5_chance_constraint.py` | 3.5 Out-of-sample validation, regret, failures, paired tests |
| `run_3_6_pareto.py` | 3.6 Feasible configurations, non-dominated sets, weight sweep |
| `run_B_synthetic_suite.py` | Benchmark B: synthetic instances, paired tests |
| `run_C_qiskit.py` | Qiskit cross-validation, circuit resources, Qiskit Aer noise |
| `make_figures.py` | Figures (PDF, PNG and TIFF at 600 dpi) and plotted data (CSV) |
| `run_all.py` | Runs all steps of a profile in order, then the figures |
| `qopt/` | Library: instances, exact solver, steel model, simulators, methods, training, classical solvers, statistics, trainability, noise, Qiskit circuits |
| `requirements.txt` | Package versions used |
| `results/` | Result files of every section (`paper` profile, both α) |
| `figures_paper_final/` | Figures and plotted data used in the manuscript, Supplementary Tables S1–S3 |
| `sources/` | RailSAHAY freight printouts, 7 Oct 2026 |
| `SHA256SUMS.txt` | SHA-256 of every file |
| `LICENSE`, `LICENSE-DATA.md` | MIT (code), CC BY 4.0 (data); `sources/` not covered |

## 2. Methods

| ID | Method | Constraint handling | Angles per layer |
|---|---|---|---|
| **M1** | **MPSX-QAOA (proposed adaptation; see Provenance)**: one XY ring register per material with slack qubits, Dicke initial state, one mixer angle per XY edge | structural | 1 + XY edges |
| M2 | MPSX-QAOA, one mixer angle per register | structural | 1 + registers |
| M3 | QAOA, X mixer, quadratic penalty with binary slack | penalty | 2 |
| M4 | QAOA, unbalanced penalisation (Montañez-Barrera et al. 2024) | penalty | 2 |
| M5 | Particle-conserving multi-angle baseline constructed for this study (labelled MA-PC in the code), three materials | global count structural; per-material shortfall penalised | free |
| classical | enumeration (ground truth); MILP on the McCormick linearisation (PuLP/CBC); greedy; simulated annealing; tabu search | | |

- Sections 3.4–3.6 use M1 (`methods.headline`).
- Provenance of M1. Its components are established: XY ring mixers and Dicke initial states for Hamming-weight constraints (Hadfield et al. 2019, doi:10.3390/a12020034; Wang et al. 2020, doi:10.1103/PhysRevA.101.012320; Dicke-state preparation: Bärtschi and Eidenbenz 2019, doi:10.1007/978-3-030-25027-0_9); one angle per term (multi-angle QAOA, Herrman et al. 2022, doi:10.1038/s41598-022-10555-8; one angle per XY term on a single Dicke register with equality constraints, Kordonowy and Leipold 2026, doi:10.1038/s41534-026-01192-4); per-register Dicke states with intra-register XY mixers for equality constraints (Min et al. 2026, doi:10.1038/s41598-026-65365-z). M1 combines these and adds s_g = n_g − k_g slack qubits to each register, so that a register of conserved weight n_g encodes the inequality count_g ≥ k_g, inside a chance-constrained supply-selection objective with a risk term. This combination was not found in a non-systematic search (Oct 2026); it is an adaptation of existing components, not a new ansatz family.
- M5 is the authors' own construction, with no external template: one global register (all items plus S = N − Σk slack qubits, weight N), a free RZ angle per item and RZZ angle per coupled item pair, XY (RXX + RYY) exchange along an item chain and between every slack and every item (XY mixers: Hadfield et al. 2019), one angle per gate (multi-angle idea: Herrman et al. 2022), initial angles uniform in [−0.01, 0.01]. "MA-PC" is a code label, not a published name; "the original ansatz design" in `methods.M5.init_p1` refers to this construction.
- Training: CVaR_α, α = 0.05 (Barkoutsos et al. 2020) and α = 1 (expectation); COBYLA with budget max(maxiter, 2·n_params + 2); depth chain p = 1…p_max with INTERP warm starts (Zhou et al. 2020); 10,000 shots.
- `cvar_saturated` flags runs whose training CVaR is at or below f*.
- Penalty tuning (M3, M4): held-out instances (Benchmark B) or held-out seeds (steel case); penalties in units of M, the largest single-item contribution to the objective.
- Benchmark B coverage: M3 is not run at [5,5,5] (`synthetic_suite.method_exclusions`); M5 is not run when its register dimension exceeds `methods.M5.subspace_dim_cap` (5/20 instances at [4,4,4], 8/20 at [5,5,5]). Skips are recorded in `instances.csv`; paired tests use instances where both methods ran.
- Statistics: two-sided Wilcoxon signed-rank tests with Holm correction (Benchmark B); one-sided Clopper–Pearson bounds, Wilson intervals, percentile bootstrap and TOST (Section 3.5).

## 3. Running

Environment used for the reported results: Windows 11, Python 3.13.14 and the package versions in `requirements.txt`.

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python run_00_selftest.py
python load_params_v18.py
python cost_breakdown.py
python cost_sensitivity.py
python run_all.py --profile paper
```

- `python run_all.py --profile paper --status` lists completed steps. A step is complete when its folder holds `summary.json` with the current parameter-file hash.
- Individual scripts: `python <script> --profile quick|paper`; `--alpha 1` (scripts 3.1, 3.3–3.6, B, C) runs the α = 1 setting into a `_alpha1` folder; `--methods` (3.1, B) restricts the methods.
- `quick` is a short end-to-end check; `paper` is the full protocol.
- Every finished unit (training chain, tuning candidate, gradient estimate, noisy distribution, Qiskit check) is stored in `results/<section>_<profile>/_checkpoint/`; re-running a stopped script resumes from there with identical results.
- Bit-for-bit reproduction requires the same operating system and package builds. On another platform, COBYLA trajectories of individual seeds differ; exact optima, MILP and certification results do not change.

Paper-profile wall times on the reference machine (16 logical CPUs): 3.1, 91 min (α = 1: 14 min); 3.2, 24 min; 3.3–3.6, ≤ 2 min each; Benchmark B, about 8 h per α (sum of per-instance times in `instances.csv`; the runs were resumed from checkpoints); `run_C_qiskit.py`, about 10 min per α.

**Qiskit run** (`run_C_qiskit.py`):
- Cross-validation: saved trained angles are rebuilt as gate-level circuits (ideal state preparation) and simulated with `qiskit.quantum_info.Statevector`; P(optimum) and P(feasible) must match the numpy engine within `qiskit.cross_validation.tolerance` for circuits up to `qiskit.cross_validation.max_qubits` qubits.
- Resources: transpiled depth, CX and one-qubit counts per layer, basis {cx, rz, sx, x}; state preparation excluded.
- Aer noise: depolarising error per native CX (p2) and per sx/x (p1), readout flips; the depolarising parameter is set so that p is the Pauli-error probability. The trajectory model of Section 3.3 applies p2 per logical two-qubit operation.

## 4. Outputs

- `results/<section>_<profile>/` (and `_alpha1`): CSV tables, PNG figures and `summary.json` (parameter-file hash, profile, environment, wall time). In this folder only the `paper` profile is included, without the PNG previews.
- `results/figures_<profile>/`: Figures 1–11 (with 1b, 5b, 6b, 7b, 8b, 11b for α = 1) and Tables S1–S3, plus `figures_<profile>.zip`. The figures used in the manuscript are in `figures_paper_final/` (map in Section 0).
- `results/00_selftest/selftest.json`.
- `results/cost_breakdown/`: `cost_breakdown.csv`, `emission_breakdown.csv`, `references.csv`.
- `results/cost_sensitivity/`: `route_values.csv`, `optimum.csv`.

## 5. Scope

- Exhaustive enumeration solves every instance; no advantage over classical solvers is claimed.
- Noise results are simulated sensitivity scenarios, not hardware calibrations.
- Benchmark B instances are synthetic (`synthetic_suite`).
- The approximation ratio is computed over runs with at least one feasible sample.
- Cost and emission sensitivities (sea freight, port handling, rail GST, sea distance, emission factors, hub factor) are computed by `cost_sensitivity.py` for the exact optimum only; the quantum and heuristic methods are not re-run for them.

## 6. Source notes

**Prices and exchange rate**
- 83.6566 INR/USD: FRED AEXINUS, 2024 annual average.
- US$242/t: hard coking coal, FOB Australia East Coast, 2024, nominal (Resources and Energy Quarterly, Dec 2025, Table 4.2).
- Rs 5,389/t: Indian Bureau of Mines average sale price, Karnataka iron ore fines 62% to below 65% Fe, Feb 2024.
- Rs 31,800/t: Chennai HMS 80:20 steel scrap, BigMint public market report, 26 Sep 2024 (bigmint.co/insights/detail/india-chennai-scrap-prices-inch-up-by-inr-200-t-w-o-w-26-sep-583877); also used for Vellore and Bengaluru (no local price found).
- Rs 40/t: coal wharfage, Karaikal Port scale of rates BPTS/KPPL/04, effective 1 Oct 2024.
- US$22/t sea freight: assumption (not a freight-rate quote), within the US$15–25/t range by which freight differentials and currency fluctuations widen landed-cost bands (EY and Indian Steel Association, Sep 2025, p. 21).

**Plant quantities** (JSW Steel Salem Works, Environmental Statement for the year ending 31 Mar 2025, Part B)
- Coal: 673,291 t consumed at the coke oven plant.
- Iron ore: 682,162 t at the blast furnace plus 909,775 t iron ore fines at the sinter plant.
- Billets/Blooms: 1,073,347 t.
- Upper-bound sensitivity: 212,600 t "Iron Skull & Steel Scrap and Skull (includes purchased)".

**Scrap demand** = billets/blooms × (20.8 / 41.1) × 1.1236 / 12 (assumption).
- 20.8% EAF and 41.1% BOF: India crude steel by route, FY2024-25, provisional (Ministry of Steel Annual Report 2025-26, p. 17).
- 1.1236 t/t: metallic charge of a worked EAF example (Cappel 2021, "EAF Efficiency", AIST MENA), applied per tonne of billet.

**Emission factors**
- Sea: 6.1 g CO2e/tkm, bulk carrier 60,000–99,999 dwt, VLSFO, WTW with 15% distance adjustment factor (GLEC Framework v3.2, Table 14, p. 107); sensitivity 4.1 g/tkm, 100,000–199,999 dwt.
- Hubs: 2.7 kg CO2e/t storage + transshipment and 1.2 kg CO2e/t transshipment, ambient, median values (GLEC v3.2, Table 3, p. 96).
- Rail 10.6 and road 66.3 g CO2e/tkm WTW, road class HCV-3 (Smart Freight Centre, India Default GHG Emission Values V1.0, May 2025); road sensitivity 55.1 g CO2e/tkm, tractor-trailer.

**Road**
- 3.6 Rs/tkm truck rate: NITI Aayog, RMI and RMI India (Jun 2021), Exhibit 3-1, p. 31, which attributes it to Aritua et al. (2018, World Bank WPS8337); the figure is not stated in Aritua et al. itself. Not escalated to 2024; the year 2015 given in the parameter-file note is not stated in either source.
- Road distances: OSRM driving routes from city points Chennai (13.0827N, 80.2707E), Vellore (12.9165N, 79.1325E) and Bengaluru (12.9716N, 77.5946E) to the plant (11.8115N, 77.9200E, inside the OpenStreetMap "JSW Steel Plant" area); re-run 7 Oct 2026: 337.2, 201.1, 168.8 km.
- Sea distance: searoute 1.6.0, Hay Point Coal Terminal (21.2768S, 149.2930E) to Karaikal Port (10.8388N, 79.8457E), OpenStreetMap points: 9,295.0 km (`check_sea_distance.py`).

**Rail**
Source: Indian Railways freight calculator (RailSAHAY, fois.indianrail.gov.in/RailSAHAY/pages/FrgtCalc.jsp), "Expected Freight Charges" printouts dated 7 Oct 2026 (the same rake totals were first recorded on 3 Oct 2026 in the parameter file; no 3 Oct printout is kept); BOXN, 59 wagons, 68.10 t permissible carrying capacity per wagon (CC+8). The calculator states that its freight is indicative.
- Cost per tonne = rake total (incl. 5% GST) ÷ 1.05 ÷ (59 × 68.1 t).
- Coal, Karaikal Port Pvt Ltd siding (KIKP) → Mecheri Road (MCRD), commodity "Coking Coal (Imported)", 327 km, train load, class 145A: base rate Rs 797.20/t + destination terminal charge Rs 20.00/t = Rs 817.20/t; chargeable weight 4,017.89 t; Rs 3,283,427.88 + GST Rs 164,172 = Rs 3,447,600 per rake (calculator printout, 7 Oct 2026).
- Iron ore, Ballari Jn (BAY) → MCRD, commodity "Iron Ore", 609 km, train load, class 165: base rate Rs 1,113.10/t + surcharge Rs 166.97/t (15%, "dynamic pricing for commodities other than coal and coke") + other charges Rs 104.00/t (development charge, originating and destination terminal charges) = Rs 1,384.07/t; chargeable weight 4,017.89 t; Rs 5,561,054.85 + GST Rs 278,052 = Rs 5,839,107 per rake (calculator printout, 7 Oct 2026).
- Coal class: the calculator charges train load at class 145A and wagon load at class 145B (Rs 837.10/t); the testbed uses train load. Brookings (2018) reports coal at class 145.
- Share of route landed cost: coal 3.6%, iron ore 20.4%.
- Values are a single-date snapshot; rates vary with wagon type, season and tariff revisions. The network is illustrative.
- Corroboration: coal at class 145, iron ore at class 165, GST 5% on freight (Kamboj and Tongia, "Indian Railways and coal: an unsustainable interdependency", Brookings India, Jul 2018: Section 2.1, Fig. 8, Appendix Table 17). Indian Railways average freight earnings 173.3 paise/tkm at 601 km average lead, 2024-25 provisional (Economic Survey 2025-26, Statistical Appendix Table 1.26); the testbed rates, 2.50 (coal) and 2.27 (ore) Rs/tkm, are 1.44 and 1.31 times this all-commodity average.

**Assumptions**
- Values labelled A in the parameter file are assumptions, with reasons stated there. The plant, railhead and port define an illustrative sourcing network.
- Block pools (5, 4, 3, 3, 3) are a design choice (A): they leave 16 feasible count vectors at ε = 0.05; removing one block from any route except R1 still leaves 6.

**Reading the short source notes in `testbed_params_v18.json`** (the file is hash-locked to the results, so its notes are kept as run; where they differ, this section applies)
- `routes.R2.price_inr`: cite the public BigMint report of 26 Sep 2024 (above), not the 2 Aug 2024 assessment.
- `road_common`: the 3.6 Rs/tkm is from NITI Aayog, RMI and RMI India (2021), Exhibit 3-1, which attributes it to Aritua et al. (2018, World Bank Policy Research Working Paper 8337). Aritua et al. give road, rail and waterway rates as Indian national averages for 2015, in US$/tkm (their Figure 18, p. 32); the Rs value is the one printed by NITI Aayog et al. Not escalated.
- `routes.H1.sea_freight_usd`: the EY/ISA range refers to freight differentials and currency fluctuations; insurance is not mentioned.
- `routes.H1.rail` and `routes.R1.rail`: class and base rate are as printed by the calculator; the distance-slab bounds in the note are not shown on the calculator printout.
- `routes.H1.port_handling_inr`: cite section AA (wharfage) of BPTS/KPPL/04; page numbering differs between copies.
- `routes.H1.sea_distance_km.sensitivity` (9,435.2 km, +1.5%) has no recorded source; it is used only as an illustrative distance sensitivity.
- `_spec.sensitivity_policy`: the cost and emission sensitivities are now computed by `cost_sensitivity.py`; the scrap upper-bound demand and the cross-material correlation 0.3 remain declared only.
