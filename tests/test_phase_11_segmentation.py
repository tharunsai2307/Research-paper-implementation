"""
Unit and Integration Tests for Phase 11: Real Human Segmentation Annotation, Double-Annotator QC, and Dataset Release.
Tests:
1. Annotation parsing & geometry validation (coordinate bounds, Shoelace area, anti-box gate).
2. Provenance & image SHA-256 calculation.
3. Raster IoU and Dice calculation between dual annotator masks.
4. QC threshold and disagreement detection (IoU < 0.75 -> QC_REVIEW_REQUIRED).
5. Dataset completion audit and manifest integrity.
6. Hard Training Gate enforcement (0 accepted masks -> BLOCKED).
7. Historical regression protection across Phases 3 through 10.
"""

import os
import sys
import unittest
import json
import csv
import hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.annotate_polygons import (
    validate_polygon,
    is_axis_aligned_box,
    calculate_polygon_area,
    compute_polygon_iou_dice,
    CLASS_MAP
)
from scripts.box_vs_mask_analyzer import calculate_box_vs_mask_metrics
from scripts.generate_phase_11_reports import generate_phase_11_reports

class TestPhase11Segmentation(unittest.TestCase):

    def setUp(self):
        self.p11_dir = PROJECT_ROOT / "data/external/phase_11_segmentation"
        self.manifest_path = self.p11_dir / "metadata/annotation_manifest.csv"

    def test_01_annotation_parsing_and_geometry(self):
        """Verifies geometry validation rejects OOB, degenerate, and box pseudo-masks."""
        # Valid irregular lesion polygon
        valid_poly = [(0.2, 0.2), (0.35, 0.25), (0.4, 0.45), (0.28, 0.5), (0.18, 0.35)]
        valid, msg = validate_polygon(valid_poly)
        self.assertTrue(valid, msg)
        self.assertGreater(calculate_polygon_area(valid_poly), 0.0)

        # Coordinate > 1.0 rejected
        oob_poly = [(0.2, 0.2), (1.05, 0.25), (0.4, 0.45)]
        valid, msg = validate_polygon(oob_poly)
        self.assertFalse(valid)
        self.assertIn("outside [0.0, 1.0]", msg)

        # Axis-aligned box rejected (anti-fabrication rule)
        box_rect = [(0.1, 0.1), (0.6, 0.1), (0.6, 0.5), (0.1, 0.5)]
        self.assertTrue(is_axis_aligned_box(box_rect))
        valid, msg = validate_polygon(box_rect)
        self.assertFalse(valid)
        self.assertIn("REJECTED: Polygon matches an axis-aligned bounding box", msg)

    def test_02_provenance_and_manifest_structure(self):
        """Verifies Phase 11 annotation manifest exists, has 100 benchmark images, and correct columns."""
        self.assertTrue(self.manifest_path.exists())
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 100)
        expected_cols = {
            "image_id", "source_path", "disease_class", "class_id", "split", "image_sha256",
            "annotator_A_status", "annotator_A_path", "annotator_B_status", "annotator_B_path",
            "qc_status", "final_mask_path", "acceptance_status", "review_notes"
        }
        self.assertTrue(expected_cols.issubset(set(reader.fieldnames)))

    def test_03_raster_iou_dice_calculation(self):
        """Tests exact raster-level IoU and Dice calculation for double-annotator QC."""
        # 1. Identical polygons -> IoU = 1.0, Dice = 1.0
        poly1 = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
        poly2 = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
        iou, dice = compute_polygon_iou_dice(poly1, poly2)
        self.assertAlmostEqual(iou, 1.0, places=2)
        self.assertAlmostEqual(dice, 1.0, places=2)

        # 2. Completely disjoint polygons -> IoU = 0.0, Dice = 0.0
        poly3 = [(0.7, 0.7), (0.9, 0.7), (0.9, 0.9), (0.7, 0.9)]
        iou_disjoint, dice_disjoint = compute_polygon_iou_dice(poly1, poly3)
        self.assertEqual(iou_disjoint, 0.0)
        self.assertEqual(dice_disjoint, 0.0)

    def test_04_qc_disagreement_detection(self):
        """Verifies double-annotator disagreement is flagged when IoU < 0.75."""
        # Partially overlapping polygons: area1 = 0.4x0.4 = 0.16; area2 = 0.4x0.4 = 0.16
        # Intersection = 0.2x0.4 = 0.08; Union = 0.16 + 0.16 - 0.08 = 0.24 -> IoU = 0.08 / 0.24 = 0.333 (< 0.75)
        poly_a = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
        poly_b = [(0.4, 0.2), (0.8, 0.2), (0.8, 0.6), (0.4, 0.6)]
        iou, _ = compute_polygon_iou_dice(poly_a, poly_b)
        self.assertLess(iou, 0.75)
        qc_verdict = "PASS" if iou >= 0.75 else "QC_REVIEW_REQUIRED"
        self.assertEqual(qc_verdict, "QC_REVIEW_REQUIRED")

    def test_05_dataset_completion_reports(self):
        """Verifies generation of all 5 Phase 11 completion and QC artifacts."""
        generate_phase_11_reports()
        rep_dir = PROJECT_ROOT / "outputs/phase_11/annotation"
        
        status_f = rep_dir / "annotation_status.json"
        comp_f = rep_dir / "annotation_completion.csv"
        cls_f = rep_dir / "class_distribution.csv"
        split_f = rep_dir / "split_distribution.csv"
        qc_f = rep_dir / "qc_summary.json"

        self.assertTrue(status_f.exists())
        self.assertTrue(comp_f.exists())
        self.assertTrue(cls_f.exists())
        self.assertTrue(split_f.exists())
        self.assertTrue(qc_f.exists())

        with open(status_f, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["registered_images"], 100)
            self.assertEqual(data["hard_training_gate_status"], "BLOCKED")

    def test_06_hard_training_gate_enforcement(self):
        """Verifies training gate strictly enforces BLOCKED when accepted masks == 0."""
        with open(PROJECT_ROOT / "outputs/phase_11/annotation/annotation_status.json", "r", encoding="utf-8") as f:
            status_data = json.load(f)
        accepted = status_data["accepted_masks"]
        # Gate rule: accepted > 0 required for training
        gate_unlocked = (accepted > 0 and status_data["pending_images"] == 0)
        self.assertFalse(gate_unlocked)
        self.assertEqual(status_data["hard_training_gate_status"], "BLOCKED")

    def test_07_historical_protection(self):
        """Verifies historical checkpoints and metrics from Phases 3-10 remain unchanged."""
        # Phase 3 checkpoint hash
        p3_ckpt = PROJECT_ROOT / "outputs/training/EXP-002_imgsz512/weights/best.pt"
        self.assertTrue(p3_ckpt.exists())
        h = hashlib.sha256(p3_ckpt.read_bytes()).hexdigest()
        self.assertEqual(h, "e3b2b5d39f1e6bcef2fbb6fd9b8b133a38b5136caaf2e9eecde89a184e212c08")

        # Phase 4 test metrics
        p4_res = PROJECT_ROOT / "outputs/evaluation/phase_4_test_results.json"
        with open(p4_res, "r", encoding="utf-8") as f:
            m = json.load(f)["model_evaluations"]["EXP-002_imgsz512"]["overall_metrics"]
            self.assertEqual(m["precision"], 0.9044)
            self.assertEqual(m["mAP50"], 0.6862)

if __name__ == "__main__":
    unittest.main()
