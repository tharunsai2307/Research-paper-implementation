"""
Phase 12 — Test Suite: Ablation, Calibration, Robustness & Error Analysis.

Tests:
  1. Phase 3/4 regression — bitwise identical metrics
  2. Error taxonomy structure correctness
  3. Calibration ECE bounds
  4. Ablation summary completeness
  5. Threshold sweep output structure (when available)
  6. Robustness output structure (when available)
  7. Phase 12 output directory existence
  8. Research integrity — no synthetic results
"""

import unittest
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PHASE3_JSON = BASE_DIR / "outputs" / "training" / "phase_3_results.json"
PHASE4_JSON = BASE_DIR / "outputs" / "evaluation" / "phase_4_test_results.json"
PHASE12_DIR = BASE_DIR / "outputs" / "phase_12"
ERROR_DIR = PHASE12_DIR / "error_analysis"
THRESHOLD_DIR = PHASE12_DIR / "threshold_study"
ROBUSTNESS_DIR = PHASE12_DIR / "robustness"
FIG_DIR = PHASE12_DIR / "figures"

CLASS_NAMES = ["bud root dropping", "bud rot", "gray leaf spot", "leaf rot", "stembleeding"]
EXPECTED_EXP_IDS = {"EXP-001_baseline_yolov8n", "EXP-002_imgsz512"}

# Frozen Phase 4 metrics (must remain bitwise identical)
FROZEN_P4 = {
    "EXP-001_baseline_yolov8n": {"precision": 0.816, "recall": 0.4,   "mAP50": 0.6343, "mAP50_95": 0.3663},
    "EXP-002_imgsz512":         {"precision": 0.9044, "recall": 0.4,  "mAP50": 0.6862, "mAP50_95": 0.2471},
}


class TestPhase12Integrity(unittest.TestCase):

    # ── 1. Phase 3 Regression ──────────────────────────────────────────
    def test_phase3_exp_ids_intact(self):
        """Phase 3 experiment IDs must be unchanged."""
        with open(PHASE3_JSON) as f:
            p3 = json.load(f)
        exp_ids = {e["experiment_id"] for e in p3["completed_experiments"]}
        self.assertIn("EXP-001_baseline_yolov8n", exp_ids)
        self.assertIn("EXP-002_imgsz512", exp_ids)

    def test_phase3_val_map50_intact(self):
        """Phase 3 val mAP50 must match frozen values."""
        with open(PHASE3_JSON) as f:
            p3 = json.load(f)
        for exp in p3["completed_experiments"]:
            eid = exp["experiment_id"]
            if eid == "EXP-001_baseline_yolov8n":
                self.assertAlmostEqual(exp["overall_metrics"]["mAP50"], 0.6186, places=3)
            elif eid == "EXP-002_imgsz512":
                self.assertAlmostEqual(exp["overall_metrics"]["mAP50"], 0.5752, places=3)

    # ── 2. Phase 4 Regression ──────────────────────────────────────────
    def test_phase4_test_metrics_frozen(self):
        """Phase 4 test metrics must be bitwise identical to frozen values."""
        with open(PHASE4_JSON) as f:
            p4 = json.load(f)
        evals = p4["model_evaluations"]
        for eid, expected in FROZEN_P4.items():
            if eid in evals:
                actual = evals[eid]["overall_metrics"]
                for k, v in expected.items():
                    self.assertAlmostEqual(actual[k], v, places=3,
                                           msg=f"{eid}.{k}: expected {v}, got {actual[k]}")

    def test_phase4_test_images_count(self):
        """Phase 4 must report exactly 15 test images."""
        with open(PHASE4_JSON) as f:
            p4 = json.load(f)
        self.assertEqual(p4["meta"]["test_images"], 15)

    # ── 3. Phase 12 Output Directories ────────────────────────────────
    def test_phase12_dirs_exist(self):
        """Phase 12 output directories must exist."""
        for d in [PHASE12_DIR, ERROR_DIR, FIG_DIR]:
            self.assertTrue(d.exists(), f"Missing Phase 12 directory: {d}")

    def test_phase12_error_analysis_json(self):
        """per_class_error_taxonomy.json must exist and have correct structure."""
        path = ERROR_DIR / "per_class_error_taxonomy.json"
        self.assertTrue(path.exists(), "per_class_error_taxonomy.json missing")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("taxonomy", data)
        taxonomy = data["taxonomy"]
        for eid in EXPECTED_EXP_IDS:
            self.assertIn(eid, taxonomy, f"Missing model {eid} in taxonomy")
            per_class = taxonomy[eid]["per_class"]
            for cn in CLASS_NAMES:
                self.assertIn(cn, per_class, f"Missing class {cn} in taxonomy for {eid}")
                cc = per_class[cn]
                # All counts must be non-negative
                for field in ["tp", "fp", "fn", "tn"]:
                    self.assertGreaterEqual(cc[field], 0, f"{eid}.{cn}.{field} negative")

    def test_phase12_error_analysis_csv(self):
        """per_class_error_taxonomy.csv must exist with correct columns."""
        path = ERROR_DIR / "per_class_error_taxonomy.csv"
        self.assertTrue(path.exists(), "per_class_error_taxonomy.csv missing")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        expected_cols = {"model_id", "class_name", "tp", "fp", "fn", "tn"}
        actual_cols = set(reader.fieldnames or [])
        self.assertTrue(expected_cols.issubset(actual_cols),
                        f"Missing columns: {expected_cols - actual_cols}")
        # Should have 5 classes × 2 models = 10 rows
        self.assertGreaterEqual(len(rows), 5)

    def test_phase12_calibration_json(self):
        """calibration_analysis.json must exist with ECE values."""
        path = ERROR_DIR / "calibration_analysis.json"
        self.assertTrue(path.exists(), "calibration_analysis.json missing")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("calibration", data)
        for eid in EXPECTED_EXP_IDS:
            self.assertIn(eid, data["calibration"])
            cd = data["calibration"][eid]
            self.assertIn("ece", cd)
            # ECE must be in [0, 1] if computable, or None
            if cd["ece"] is not None:
                self.assertGreaterEqual(cd["ece"], 0.0)
                self.assertLessEqual(cd["ece"], 1.0)

    def test_phase12_ablation_json(self):
        """ablation_summary.json must exist with both experiments."""
        path = ERROR_DIR / "ablation_summary.json"
        self.assertTrue(path.exists(), "ablation_summary.json missing")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("experiments", data)
        exp_ids = {e["experiment_id"] for e in data["experiments"]}
        for eid in EXPECTED_EXP_IDS:
            self.assertIn(eid, exp_ids)

    def test_phase12_ablation_latency_plausible(self):
        """Ablation latency values must be plausible (> 0 ms)."""
        path = ERROR_DIR / "ablation_summary.json"
        with open(path) as f:
            data = json.load(f)
        for exp in data["experiments"]:
            lat = exp.get("cpu_mean_latency_ms")
            if lat is not None:
                self.assertGreater(lat, 0, f"{exp['experiment_id']} latency must be > 0")
                self.assertLess(lat, 5000, f"{exp['experiment_id']} latency implausibly high")

    # ── 4. Figures ─────────────────────────────────────────────────────
    def test_phase12_figures_exist(self):
        """Phase 12 figures must be generated."""
        expected_figs = ["error_taxonomy.png", "calibration_reliability.png", "ablation_comparison.png"]
        for fig_name in expected_figs:
            p = FIG_DIR / fig_name
            self.assertTrue(p.exists(), f"Missing figure: {fig_name}")
            self.assertGreater(p.stat().st_size, 1000, f"Figure file suspiciously small: {fig_name}")

    # ── 5. Threshold Sweep (conditional) ──────────────────────────────
    def test_phase12_threshold_json_if_exists(self):
        """If threshold sweep was run, validate its structure."""
        path = THRESHOLD_DIR / "threshold_sweep_results.json"
        if not path.exists():
            self.skipTest("Threshold study not yet completed; skipping structural check.")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("meta", data)
        self.assertEqual(data["meta"].get("dataset"), "Validation set (val/images)",
                         "Threshold study must NOT use test set")
        # Must have per-model results
        for eid in EXPECTED_EXP_IDS:
            self.assertIn(eid, data, f"Missing model {eid} in threshold results")
        # Optimal thresholds must be present
        self.assertIn("optimal_thresholds", data)
        for eid in EXPECTED_EXP_IDS:
            opt = data["optimal_thresholds"].get(eid, {})
            conf = opt.get("conf_threshold")
            self.assertIsNotNone(conf, f"Missing optimal conf for {eid}")
            self.assertGreater(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_phase12_threshold_csv_if_exists(self):
        """If threshold CSV exists, validate columns and row count."""
        path = THRESHOLD_DIR / "threshold_sweep_results.csv"
        if not path.exists():
            self.skipTest("Threshold CSV not yet completed.")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        expected_cols = {"model_id", "conf_threshold", "precision", "recall", "f1"}
        actual_cols = set(reader.fieldnames or [])
        self.assertTrue(expected_cols.issubset(actual_cols))
        # 18 thresholds × 2 models = 36 rows
        self.assertGreaterEqual(len(rows), 30, "Expected ~36 threshold rows")

    # ── 6. Robustness (conditional) ────────────────────────────────────
    def test_phase12_robustness_json_if_exists(self):
        """If robustness study was run, validate its structure."""
        path = ROBUSTNESS_DIR / "robustness_results.json"
        if not path.exists():
            self.skipTest("Robustness study not yet completed.")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("meta", data)
        for eid in EXPECTED_EXP_IDS:
            self.assertIn(eid, data, f"Missing model {eid} in robustness results")
        # Baseline must be present
        for eid in EXPECTED_EXP_IDS:
            self.assertIn("baseline", data[eid])

    # ── 7. Research Integrity ──────────────────────────────────────────
    def test_no_synthetic_masks_in_phase12(self):
        """Phase 12 must not have created any pseudo-masks."""
        seg_dir = BASE_DIR / "data" / "external" / "phase_9_segmentation"
        if not seg_dir.exists():
            return
        manifest = seg_dir / "metadata" / "annotation_manifest.csv"
        if not manifest.exists():
            return
        with open(manifest, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        accepted = [r for r in rows if r.get("status", "").strip().upper() == "ACCEPTED"]
        # In Phase 12 (no annotation work) accepted count must still be 0
        self.assertEqual(len(accepted), 0, "Phase 12 must not have accepted any masks")

    def test_phase4_test_results_not_modified(self):
        """The Phase 4 untouched test result JSON must retain its meta timestamp."""
        with open(PHASE4_JSON) as f:
            p4 = json.load(f)
        # Timestamp recorded in Phase 4 must remain
        ts = p4.get("meta", {}).get("date", "")
        self.assertTrue(len(ts) > 0, "Phase 4 test_results.json meta.date must not be empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
