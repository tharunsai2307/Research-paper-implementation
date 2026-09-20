"""
Phase 12 — Master Orchestrator Script.

Runs all Phase 12 sub-experiments in sequence:
  1. Error Analysis, Calibration & Ablation (from existing Phase 4 records — no test re-run)
  2. Confidence Threshold Sensitivity Study (validation set only)
  3. Perturbation-Based Robustness Testing (validation set only)

Generates a unified Phase 12 summary JSON.

Outputs:
  outputs/phase_12/phase_12_summary.json
"""

import json
import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "phase_12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("Error Analysis + Calibration + Ablation", "scripts/phase_12_error_analysis.py"),
    ("Threshold Sensitivity Study",             "scripts/phase_12_threshold_study.py"),
    ("Perturbation Robustness Testing",          "scripts/phase_12_robustness_test.py"),
]


def run_script(name, script_rel):
    script = BASE_DIR / script_rel
    print(f"\n{'='*60}")
    print(f"[PHASE 12] Running: {name}")
    print(f"  Script: {script}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
        text=True,
        cwd=str(BASE_DIR),
    )
    elapsed = round(time.time() - t0, 1)
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"\n[PHASE 12] {name}: {status} (elapsed {elapsed}s)")
    return {"name": name, "script": script_rel, "status": status,
            "returncode": result.returncode, "elapsed_s": elapsed}


def build_summary(run_results):
    all_pass = all(r["status"] == "PASS" for r in run_results)
    summary = {
        "meta": {
            "phase": "Phase 12 — Comprehensive Model Ablation, Calibration, Robustness & Error Analysis",
            "project": "A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection "
                       "and Plantation Health Monitoring Using Mobile Applications",
            "integrity_rule": "No new training; no test-set re-evaluation; no fabricated results.",
            "source_phases": ["Phase 3 (training)", "Phase 4 (test evaluation)", "Phase 11 (annotation workspace)"],
        },
        "sub_experiments": run_results,
        "release_gate": "PASS" if all_pass else "FAIL",
        "outputs": {
            "error_analysis": "outputs/phase_12/error_analysis/",
            "threshold_study": "outputs/phase_12/threshold_study/",
            "robustness": "outputs/phase_12/robustness/",
            "figures": "outputs/phase_12/figures/",
        },
    }
    return summary


def main():
    print("\n" + "="*60)
    print("PHASE 12 — MASTER ORCHESTRATOR")
    print("="*60)
    print("Constraints:")
    print("  - No model retraining")
    print("  - Test set evaluated ONCE in Phase 4; NOT re-evaluated here")
    print("  - Validation set used for threshold sweep and robustness tests")
    print("  - Phase 4 records used for error taxonomy and calibration")
    print("="*60)

    run_results = []
    for name, script in SCRIPTS:
        r = run_script(name, script)
        run_results.append(r)

    summary = build_summary(run_results)
    summary_path = OUT_DIR / "phase_12_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("PHASE 12 — MASTER SUMMARY")
    print("="*60)
    for r in run_results:
        print(f"  [{r['status']:4s}] {r['name']} ({r['elapsed_s']}s)")
    print(f"\nRelease Gate: {summary['release_gate']}")
    print(f"Summary JSON: {summary_path}")
    print("="*60)

    if summary["release_gate"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
