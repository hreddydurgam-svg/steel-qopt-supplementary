"""
Run every step of one profile in order, skipping steps that are already complete. Safe to stop at any time:
re-running this command continues where it stopped (finished steps are skipped; an unfinished step reuses its
checkpointed units, so at most the unit in progress is recomputed).

  python run_all.py --profile quick|paper            # all steps
  python run_all.py --profile paper --status         # only show which steps are complete

A step is complete when its results folder holds summary.json written with the current parameter file.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from load_params_v18 import load

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="quick", choices=["quick", "paper"])
ap.add_argument("--status", action="store_true", help="only report which steps are complete")
args = ap.parse_args()
prof = args.profile
sha = load().sha256

STEPS = [
    ("3.1 steel case study", ["run_3_1_steel_case.py"], f"3_1_steel_case_{prof}"),
    ("3.1 steel case study, CVaR alpha 1 (M1, M2, M4, M5)",
     ["run_3_1_steel_case.py", "--alpha", "1", "--methods", "M1,M2,M4,M5"], f"3_1_steel_case_{prof}_alpha1"),
    ("3.4 stress scenarios", ["run_3_4_disruptions.py"], f"3_4_disruptions_{prof}"),
    ("3.4 stress scenarios, CVaR alpha 1", ["run_3_4_disruptions.py", "--alpha", "1"], f"3_4_disruptions_{prof}_alpha1"),
    ("3.5 chance constraint", ["run_3_5_chance_constraint.py"], f"3_5_chance_constraint_{prof}"),
    ("3.5 chance constraint, CVaR alpha 1", ["run_3_5_chance_constraint.py", "--alpha", "1"], f"3_5_chance_constraint_{prof}_alpha1"),
    ("3.6 cost-emissions trade-off", ["run_3_6_pareto.py"], f"3_6_pareto_{prof}"),
    ("3.6 cost-emissions trade-off, CVaR alpha 1", ["run_3_6_pareto.py", "--alpha", "1"], f"3_6_pareto_{prof}_alpha1"),
    ("3.3 simulated noise", ["run_3_3_noise.py"], f"3_3_noise_{prof}"),
    ("3.3 simulated noise, CVaR alpha 1", ["run_3_3_noise.py", "--alpha", "1"], f"3_3_noise_{prof}_alpha1"),
    ("3.2 trainability", ["run_3_2_trainability.py"], f"3_2_trainability_{prof}"),
    ("Benchmark B, CVaR alpha 0.05", ["run_B_synthetic_suite.py"], f"B_synthetic_suite_{prof}"),
    ("Benchmark B, CVaR alpha 1", ["run_B_synthetic_suite.py", "--alpha", "1"], f"B_synthetic_suite_{prof}_alpha1"),
    ("Qiskit cross-validation, resources, Aer noise", ["run_C_qiskit.py"], f"C_qiskit_{prof}"),
    ("Qiskit, CVaR alpha 1 (steel cross-validation, Aer noise)", ["run_C_qiskit.py", "--alpha", "1"], f"C_qiskit_{prof}_alpha1"),
]


def complete(folder):
    sj = ROOT / "results" / folder / "summary.json"
    if not sj.exists():
        return False
    try:
        return json.loads(sj.read_text()).get("params_sha256") == sha
    except (json.JSONDecodeError, OSError):
        return False


print(f"Profile '{prof}', parameter file {sha[:12]}...")
for i, (name, _, folder) in enumerate(STEPS, 1):
    print(f"  [{'done' if complete(folder) else 'todo'}] step {i}: {name}")
if args.status:
    sys.exit(0)

for i, (name, cmd, folder) in enumerate(STEPS, 1):
    if complete(folder):
        continue
    print(f"\n{'#' * 78}\n# step {i}/{len(STEPS)}: {name}   (started {time.strftime('%H:%M:%S')})\n{'#' * 78}", flush=True)
    rc = subprocess.call([sys.executable, *cmd, "--profile", prof], cwd=ROOT)
    if rc != 0 or not complete(folder):
        sys.exit(f"\nSTOPPED at step {i} ({name}), exit code {rc}. Read the messages above; "
                 f"re-running this command continues from here.")

print(f"\n{'#' * 78}\n# figures\n{'#' * 78}", flush=True)
rc = subprocess.call([sys.executable, "make_figures.py", "--profile", prof], cwd=ROOT)
print("\nALL STEPS COMPLETE" if rc == 0 else f"\nfigures exited with code {rc}")
sys.exit(rc)
