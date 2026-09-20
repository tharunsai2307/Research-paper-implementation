"""
Unit and Integration Tests for Phase 10: Real Mask Annotation Execution & YOLOv8-Seg Experiments.
Tests:
1. Mask status audit: 100 registered, 0 completed, 100 pending, training gate BLOCKED.
2. Polygon validation rules: geometry, bounds, Shoelace area, anti-bounding-box check.
3. Dataset YAML configuration: 5 valid disease classes, valid format.
4. Split integrity: 63 train, 27 val, 10 test with zero cross-split leakage.
5. Mask/image correspondence: all manifest images verified on disk.
6. Severity calculation & Box-vs-Mask calculation: gamma ratio math and edge cases.
7. Mobile API compatibility: segmentation status returns PENDING_REAL_MASK_DATA without breaking Phase 6.
"""

import os
import sys
import unittest
import json
import csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mask_status import audit_mask_status
from scripts.annotate_polygons import (
    validate_polygon,
    is_axis_aligned_box,
    calculate_polygon_area,
    CLASS_MAP
)
from scripts.box_vs_mask_analyzer import calculate_box_vs_mask_metrics
from src.segmentation.interface import FallbackBoundingBoxSegmentationProxy

class TestPhase10Segmentation(unittest.TestCase):

    def setUp(self):
        self.seg_dir = PROJECT_ROOT / "data/external/phase_9_segmentation"
        self.manifest_path = self.seg_dir / "metadata/annotation_manifest.csv"

    def test_01_mask_status_audit(self):
        """Verifies automated audit detects 100 registered, 0 completed, 100 pending images."""
        summary = audit_mask_status()
        self.assertEqual(summary["total_registered_images"], 100)
        self.assertEqual(summary["completed_annotations"], 0)
        self.assertEqual(summary["pending_annotations"], 100)
        self.assertEqual(summary["hard_training_gate_status"], "BLOCKED")
        self.assertEqual(summary["orphan_annotations"], 0)
        self.assertEqual(summary["orphan_masks"], 0)

    def test_02_polygon_validation_rules(self):
        """Tests polygon validation, coordinate bounds, and anti-bounding-box gate."""
        # 1. Valid non-trivial polygon
        valid_poly = [(0.1, 0.1), (0.2, 0.4), (0.35, 0.5), (0.3, 0.2), (0.15, 0.15)]
        valid, msg = validate_polygon(valid_poly)
        self.assertTrue(valid, msg)
        self.assertGreater(calculate_polygon_area(valid_poly), 0.0)

        # 2. Insufficient vertices (< 3)
        invalid_few = [(0.1, 0.1), (0.2, 0.2)]
        valid, msg = validate_polygon(invalid_few)
        self.assertFalse(valid)
        self.assertIn("minimum 3 required", msg)

        # 3. Out of bounds coordinates (> 1.0)
        invalid_oob = [(0.1, 0.1), (1.2, 0.4), (0.3, 0.2)]
        valid, msg = validate_polygon(invalid_oob)
        self.assertFalse(valid)
        self.assertIn("outside [0.0, 1.0]", msg)

        # 4. Anti-Bounding Box: axis-aligned 4-point rectangle must be REJECTED
        rect_box = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.6), (0.1, 0.6)]
        self.assertTrue(is_axis_aligned_box(rect_box))
        valid, msg = validate_polygon(rect_box)
        self.assertFalse(valid)
        self.assertIn("REJECTED: Polygon matches an axis-aligned bounding box", msg)

    def test_03_dataset_yaml_configuration(self):
        """Verifies configs/coconut_segmentation.yaml matches Phase 10 disease ontology."""
        yaml_path = PROJECT_ROOT / "configs/coconut_segmentation.yaml"
        self.assertTrue(yaml_path.exists(), "configs/coconut_segmentation.yaml is missing!")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("nc: 5", content)
        for class_id, class_name in CLASS_MAP.items():
            self.assertIn(f"{class_id}: {class_name}", content)
        self.assertIn("hard_gate_status: BLOCKED_PENDING_MANUAL_MASKS", content)

    def test_04_split_integrity(self):
        """Verifies segmentation split integrity: 63 train, 27 val, 10 test with zero leakage."""
        splits_dir = self.seg_dir / "splits"
        train_p = splits_dir / "train.txt"
        val_p = splits_dir / "val.txt"
        test_p = splits_dir / "test.txt"

        self.assertTrue(train_p.exists())
        self.assertTrue(val_p.exists())
        self.assertTrue(test_p.exists())

        with open(train_p, "r", encoding="utf-8") as f:
            train_set = set(l.strip() for l in f if l.strip())
        with open(val_p, "r", encoding="utf-8") as f:
            val_set = set(l.strip() for l in f if l.strip())
        with open(test_p, "r", encoding="utf-8") as f:
            test_set = set(l.strip() for l in f if l.strip())

        self.assertEqual(len(train_set), 63)
        self.assertEqual(len(val_set), 27)
        self.assertEqual(len(test_set), 10)
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 100)

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_05_mask_image_correspondence(self):
        """Verifies all registered images in manifest physically exist in images/."""
        self.assertTrue(self.manifest_path.exists())
        images_dir = self.seg_dir / "images"

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 100)
        for row in rows:
            img_file = images_dir / row["image_id"]
            self.assertTrue(img_file.exists(), f"Image {row['image_id']} missing from disk!")

    def test_06_box_vs_mask_calculation(self):
        """Verifies empirical gamma formula calculation and zero-division stability."""
        # Box: xc=0.5, yc=0.5, w=0.4, h=0.4 -> area = 0.16
        boxes = [[0.5, 0.5, 0.4, 0.4]]
        # Mask polygon with area = 0.08
        mask_poly = [[(0.3, 0.3), (0.7, 0.3), (0.7, 0.5), (0.3, 0.5)]]  # 0.4 * 0.2 = 0.08
        res = calculate_box_vs_mask_metrics(boxes, mask_poly)
        self.assertEqual(res["status"], "COMPUTED")
        self.assertAlmostEqual(res["bbox_area"], 0.16, places=4)
        self.assertAlmostEqual(res["mask_area"], 0.08, places=4)
        self.assertAlmostEqual(res["gamma"], 2.0, places=4)

        # Empty case
        res_empty = calculate_box_vs_mask_metrics([], [])
        self.assertEqual(res_empty["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(res_empty["gamma"])

    def test_07_api_compatibility_and_severity(self):
        """Verifies segmentation fallback interface conforms to Step 20 schema contract."""
        proxy = FallbackBoundingBoxSegmentationProxy(operational_threshold=0.25)
        mock_img = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_detections = [{
            "confidence": 0.85,
            "bounding_box": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        }]
        out = proxy.segment_image(mock_img, mock_detections)
        self.assertEqual(out["status"], "PENDING_REAL_MASK_DATA")
        self.assertEqual(out["segmentation_status"], "PENDING_REAL_MASK_DATA")
        self.assertFalse(out["mask_available"])
        self.assertGreater(out["image_space_severity_ratio"], 0.0)

if __name__ == "__main__":
    unittest.main()
