"""
Phase 9 Unit Test Suite:
Validates:
1. Annotation manifest integrity and image presence (100 benchmark images).
2. Split file definitions and absence of cross-split leakage.
3. Bounding box vs mask comparison calculation logic and inflation factor.
4. Segmentation validator script passes with 100% compliance.
5. Anti-fabrication check: Zero synthetic masks in annotations.
"""

import unittest
import csv
from pathlib import Path
from src.segmentation.box_vs_mask_analysis import compute_box_vs_mask_comparison
from scripts.validate_segmentation_annotations import validate_segmentation_workspace

class TestPhase9Segmentation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path("data/external/phase_9_segmentation")
        cls.manifest_p = cls.base_dir / "metadata/annotation_manifest.csv"

    def test_01_manifest_image_pool_completeness(self):
        """Test manifest contains exactly 100 benchmark images and all exist on disk."""
        self.assertTrue(self.manifest_p.exists())
        with open(self.manifest_p, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        
        self.assertEqual(len(records), 100)
        
        # Verify 20 images per class
        from collections import Counter
        class_counts = Counter(r["disease_class"] for r in records)
        self.assertEqual(len(class_counts), 5)
        for c, count in class_counts.items():
            self.assertEqual(count, 20, f"Class {c} does not have exactly 20 images")

        # Verify all images exist on disk
        for r in records:
            img_path = self.base_dir / "images" / r["image_id"]
            self.assertTrue(img_path.exists(), f"Missing image {r['image_id']}")

    def test_02_split_leakage_free_guarantee(self):
        """Test train, val, and test splits partition the 100 images with zero overlap."""
        splits_dir = self.base_dir / "splits"
        train_p = splits_dir / "train.txt"
        val_p = splits_dir / "val.txt"
        test_p = splits_dir / "test.txt"

        self.assertTrue(train_p.exists())
        self.assertTrue(val_p.exists())
        self.assertTrue(test_p.exists())

        train_imgs = set(line.strip() for line in open(train_p) if line.strip())
        val_imgs = set(line.strip() for line in open(val_p) if line.strip())
        test_imgs = set(line.strip() for line in open(test_p) if line.strip())

        self.assertEqual(len(train_imgs), 63)
        self.assertEqual(len(val_imgs), 27)
        self.assertEqual(len(test_imgs), 10)
        self.assertEqual(len(train_imgs.union(val_imgs).union(test_imgs)), 100)

        # Zero leakage
        self.assertEqual(len(train_imgs.intersection(val_imgs)), 0)
        self.assertEqual(len(train_imgs.intersection(test_imgs)), 0)
        self.assertEqual(len(val_imgs.intersection(test_imgs)), 0)

    def test_03_box_vs_mask_comparison_metrics(self):
        """Test calculation of bounding box to mask inflation factor."""
        # 100x100 bounding box (area = 10,000 px) in 1000x1000 image
        box = [100.0, 100.0, 200.0, 200.0]
        # Inscribed triangle polygon inside box (base 100, height 100, area = 5,000 px)
        poly = [[100.0, 100.0], [200.0, 100.0], [150.0, 200.0]]

        res = compute_box_vs_mask_comparison(
            boxes=[box],
            polygon_points=poly,
            image_width=1000,
            image_height=1000
        )
        self.assertEqual(res["status"], "MEASURED")
        self.assertEqual(res["total_bounding_box_area_px"], 10000.0)
        self.assertEqual(res["mask_area_px"], 5000.0)
        # Inflation factor 10,000 / 5,000 = 2.0
        self.assertAlmostEqual(res["box_to_mask_inflation_factor"], 2.0)

    def test_04_box_vs_mask_pending_state(self):
        """Test pending polygon gracefully returns status PENDING_REAL_MASK_DATA."""
        box = [50.0, 50.0, 150.0, 150.0]
        res = compute_box_vs_mask_comparison(
            boxes=[box],
            polygon_points=None,
            image_width=500,
            image_height=500
        )
        self.assertEqual(res["status"], "PENDING_REAL_MASK_DATA")
        self.assertIsNone(res["mask_area_px"])

    def test_05_segmentation_workspace_validator(self):
        """Test validate_segmentation_workspace executes and returns PASS."""
        res = validate_segmentation_workspace()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["total_images"], 100)
        self.assertEqual(res["gate_status"], "BLOCKED_PENDING_MANUAL_MASKS")

if __name__ == "__main__":
    unittest.main()
