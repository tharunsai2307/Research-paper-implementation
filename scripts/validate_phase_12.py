"""
Phase 12 — Release Gate Validator.

Runs all Phase 12 checks plus regressions for Phases 3–11.
Exits 0 on full PASS, 1 on any FAIL.

Usage:
  python scripts/validate_phase_12.py
"""

import sys
import json
import csv
import subprocess
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE3_JSON = BASE_DIR / "outputs" / "training" / "phase_3_results.json"
PHASE4_JSON = BASE_DIR / "outputs" / "evaluation" / "phase_4_test_results.json"
PHASE12_DIR = BASE_DIR / "outputs" / "phase_12"
ERROR_DIR = PHASE12_DIR / "error_analysis"
FIG_DIR = PHASE12_DIR / "figures"

CLASS_NAMES = ["bud root dropping", "bud rot", "gray leaf spot", "leaf rot", "stembleeding"]

FROZEN_P4 = {
    "EXP-001_baseline_yolov8n": {"precision": 0.816, "recall": 0.4,   "mAP50": 0.6343, "mAP50_95": 0.3663},
    "EXP-002_imgsz512":         {"precision": 0.9044, "recall": 0.4,  "mAP50": 0.6862, "mAP50_95": 0.2471},
}

CHECKS = []
PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, ok, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    tag = f"  [{status}]"
    msg = f"{tag} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    CHECKS.append({"name": name, "status": status, "detail": detail})


def run():
    print("=" * 65)
    print("PHASE 12 RELEASE GATE VALIDATOR")
    print("=" * 65)
    t0 = time.time()

    # ── 1. Phase 3/4 regressions ──────────────────────────────────────
    print("\n[REGRESSION] Phase 3 & 4 metrics")
    try:
        with open(PHASE3_JSON) as f:
            p3 = json.load(f)
        exp_ids = {e["experiment_id"] for e in p3["completed_experiments"]}
        check("Phase 3 EXP-001 present", "EXP-001_baseline_yolov8n" in exp_ids)
        check("Phase 3 EXP-002 present", "EXP-002_imgsz512" in exp_ids)
    except Exception as e:
        check("Phase 3 JSON readable", False, str(e))

    try:
        with open(PHASE4_JSON) as f:
            p4 = json.load(f)
        evals = p4["model_evaluations"]
        check("Phase 4 test images = 15", p4["meta"]["test_images"] == 15)
        for eid, expected in FROZEN_P4.items():
            if eid in evals:
                actual = evals[eid]["overall_metrics"]
                for k, v in expected.items():
                    ok = abs(actual.get(k, -9999) - v) < 0.001
                    check(f"Phase 4 {eid}.{k} frozen", ok, f"expected {v}, got {actual.get(k)}")
    except Exception as e:
        check("Phase 4 JSON readable", False, str(e))

    # ── 2. Phase 12 outputs ───────────────────────────────────────────
    print("\n[PHASE 12] Error analysis outputs")
    check("Phase 12 output dir exists", PHASE12_DIR.exists())
    check("Phase 12 error_analysis dir exists", ERROR_DIR.exists())
    check("Phase 12 figures dir exists", FIG_DIR.exists())

    # Error taxonomy
    tax_path = ERROR_DIR / "per_class_error_taxonomy.json"
    check("per_class_error_taxonomy.json exists", tax_path.exists())
    if tax_path.exists():
        try:
            with open(tax_path) as f:
                tax = json.load(f)
            taxonomy = tax.get("taxonomy", {})
            for eid in ["EXP-001_baseline_yolov8n", "EXP-002_imgsz512"]:
                check(f"Taxonomy has {eid}", eid in taxonomy)
                if eid in taxonomy:
                    per_class = taxonomy[eid]["per_class"]
                    for cn in CLASS_NAMES:
                        check(f"Taxonomy {eid} has class '{cn}'", cn in per_class)
                        if cn in per_class:
                            cc = per_class[cn]
                            check(f"  {cn} counts non-negative",
                                  all(cc.get(k, 0) >= 0 for k in ["tp", "fp", "fn", "tn"]))
        except Exception as e:
            check("Error taxonomy parseable", False, str(e))

    # Calibration
    cal_path = ERROR_DIR / "calibration_analysis.json"
    check("calibration_analysis.json exists", cal_path.exists())
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                cal_data = json.load(f)
            for eid in ["EXP-001_baseline_yolov8n", "EXP-002_imgsz512"]:
                cd = cal_data.get("calibration", {}).get(eid, {})
                check(f"Calibration has {eid}", bool(cd))
                ece = cd.get("ece")
                if ece is not None:
                    check(f"ECE {eid} in [0,1]", 0.0 <= ece <= 1.0, f"ECE={ece:.4f}")
        except Exception as e:
            check("Calibration JSON parseable", False, str(e))

    # Ablation
    abl_path = ERROR_DIR / "ablation_summary.json"
    check("ablation_summary.json exists", abl_path.exists())
    if abl_path.exists():
        try:
            with open(abl_path) as f:
                abl = json.load(f)
            exp_ids = {e["experiment_id"] for e in abl.get("experiments", [])}
            check("Ablation has EXP-001", "EXP-001_baseline_yolov8n" in exp_ids)
            check("Ablation has EXP-002", "EXP-002_imgsz512" in exp_ids)
        except Exception as e:
            check("Ablation JSON parseable", False, str(e))

    # Figures
    print("\n[PHASE 12] Figures")
    for fig in ["error_taxonomy.png", "calibration_reliability.png", "ablation_comparison.png"]:
        p = FIG_DIR / fig
        exists = p.exists()
        check(f"Figure: {fig}", exists)
        if exists:
            check(f"  {fig} non-trivial size", p.stat().st_size > 1000)

    # Threshold study (conditional)
    print("\n[PHASE 12] Threshold study (conditional)")
    thr_path = PHASE12_DIR / "threshold_study" / "threshold_sweep_results.json"
    if thr_path.exists():
        try:
            with open(thr_path) as f:
                thr = json.load(f)
            check("Threshold study: dataset = val only",
                  "Validation set" in thr.get("meta", {}).get("dataset", ""))
            check("Threshold study: optimal_thresholds present",
                  "optimal_thresholds" in thr)
            for eid in ["EXP-001_baseline_yolov8n", "EXP-002_imgsz512"]:
                check(f"Threshold study has {eid}", eid in thr)
        except Exception as e:
            check("Threshold study JSON parseable", False, str(e))
    else:
        print("  [SKIP] Threshold study not yet completed (still running or pending)")

    # Robustness (conditional)
    print("\n[PHASE 12] Robustness study (conditional)")
    rob_path = PHASE12_DIR / "robustness" / "robustness_results.json"
    if rob_path.exists():
        try:
            with open(rob_path) as f:
                rob = json.load(f)
            for eid in ["EXP-001_baseline_yolov8n", "EXP-002_imgsz512"]:
                check(f"Robustness has {eid}", eid in rob)
                if eid in rob:
                    check(f"Robustness {eid} has baseline", "baseline" in rob[eid])
        except Exception as e:
            check("Robustness JSON parseable", False, str(e))
    else:
        print("  [SKIP] Robustness study not yet completed (still running or pending)")

    # Research integrity
    print("\n[RESEARCH INTEGRITY]")
    seg_dir = BASE_DIR / "data" / "external" / "phase_9_segmentation"
    manifest = seg_dir / "metadata" / "annotation_manifest.csv"
    if manifest.exists():
        try:
            with open(manifest, newline="") as f:
                rows = list(csv.DictReader(f))
            accepted = [r for r in rows if r.get("status", "").strip().upper() == "ACCEPTED"]
            check("Segmentation: accepted masks still 0 (training gate intact)", len(accepted) == 0,
                  f"accepted={len(accepted)}")
        except Exception as e:
            check("Annotation manifest readable", False, str(e))
    else:
        check("Annotation manifest present", False, "file not found")

    # Prior phase validators
    print("\n[PRIOR PHASES] Regression validators")
    validators = [
        "scripts/validate_health_monitoring.py",
        "scripts/validate_phase_10.py",
        "scripts/validate_phase_11.py",
    ]
    for v in validators:
        vpath = BASE_DIR / v
        if vpath.exists():
            result = subprocess.run(
                [sys.executable, str(vpath)],
                capture_output=True, text=True, cwd=str(BASE_DIR)
            )
            ok = result.returncode == 0
            check(f"Prior validator: {v}", ok,
                  "" if ok else result.stdout[-200:].strip())
        else:
            check(f"Validator exists: {v}", False, "file not found")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    total = PASS_COUNT + FAIL_COUNT
    print("\n" + "=" * 65)
    print(f"PHASE 12 RELEASE GATE — {PASS_COUNT}/{total} PASS in {elapsed}s")
    if FAIL_COUNT == 0:
        print("Final release gate: PASS")
    else:
        print(f"Final release gate: FAIL ({FAIL_COUNT} failures)")
    print("=" * 65)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    passed = run()
    sys.exit(0 if passed else 1)
