"""
Comprehensive Unit Test Suite for Phase 5 Plantation Health Monitoring Engine.

Specifically tests all 12 items required by Step 21:
1. IoU (test_iou)
2. Bounding-box area (test_bounding_box_area)
3. Overlap handling (test_overlap_handling)
4. Disease counting (test_disease_counting)
5. Image-level aggregation (test_image_level_aggregation)
6. Disease distribution (test_disease_distribution)
7. Health-index calculation (test_health_index_calculation)
8. Empty detections (test_empty_detections)
9. Low-confidence detections (test_low_confidence_detections)
10. Invalid boxes (test_invalid_boxes)
11. Multiple diseases in one image (test_multiple_diseases_in_one_image)
12. Repeated observations (test_repeated_observations)
"""

import unittest
import numpy as np
from src.health_monitoring.schemas import (
    CLASS_NAMES,
    ObservationStatus,
    DEFAULT_OPERATIONAL_THRESHOLD,
    DEFAULT_CANDIDATE_THRESHOLD,
    validate_detection_record
)
from src.health_monitoring.severity_proxy import (
    box_area,
    box_iou,
    calculate_union_area,
    calculate_relative_affected_area_proxy
)
from src.health_monitoring.disease_distribution import calculate_disease_distribution
from src.health_monitoring.plantation_health import PlantationHealthAggregator

class TestHealthMonitoringEngine(unittest.TestCase):

    # 1. IoU
    def test_iou(self):
        """Test Intersection over Union (IoU) computation."""
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(box_iou(b1, b2), 1.0)

        b3 = [20.0, 20.0, 30.0, 30.0]
        self.assertEqual(box_iou(b1, b3), 0.0)

        b4 = [5.0, 0.0, 15.0, 10.0]
        self.assertAlmostEqual(box_iou(b1, b4), 50.0 / 150.0)

    # 2. Bounding-box area
    def test_bounding_box_area(self):
        """Test bounding-box area calculation."""
        b = [10.0, 20.0, 30.0, 50.0]  # w=20, h=30 -> area=600
        self.assertEqual(box_area(b), 600.0)
        self.assertEqual(box_area([10.0, 10.0, 10.0, 20.0]), 0.0)
        self.assertEqual(box_area([30.0, 50.0, 10.0, 20.0]), 0.0)

    # 3. Overlap handling
    def test_overlap_handling(self):
        """Test exact 2D sweep-line union area calculation eliminates double counting."""
        b1 = [0.0, 0.0, 10.0, 10.0]  # 100
        b2 = [20.0, 20.0, 30.0, 30.0]  # 100
        self.assertAlmostEqual(calculate_union_area([b1, b2]), 200.0)

        # Completely identical boxes -> union area equals single box area
        b3 = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(calculate_union_area([b1, b3]), 100.0)

        # Nested box
        b_outer = [0.0, 0.0, 20.0, 20.0]  # 400
        b_inner = [5.0, 5.0, 15.0, 15.0]  # 100
        self.assertAlmostEqual(calculate_union_area([b_outer, b_inner]), 400.0)

    # 4. Disease counting
    def test_disease_counting(self):
        """Test precise lesion detection instance counting across records."""
        records = [
            {
                "image_id": "img1.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detections": [
                    {"disease_class": "leaf rot", "class_id": 3, "confidence": 0.8},
                    {"disease_class": "leaf rot", "class_id": 3, "confidence": 0.75}
                ]
            },
            {
                "image_id": "img2.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detections": [
                    {"disease_class": "leaf rot", "class_id": 3, "confidence": 0.9}
                ]
            }
        ]
        dist = calculate_disease_distribution(records)
        self.assertEqual(dist["total_accepted_detections"], 3)
        self.assertEqual(dist["class_distributions"]["leaf rot"]["detection_count"], 3)

    # 5. Image-level aggregation
    def test_image_level_aggregation(self):
        """Test that image-level records aggregate correctly into plantation-level observation totals."""
        records = [
            {
                "image_id": "img1.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detection_count": 2,
                "relative_affected_area_proxy": 0.15,
                "accepted_detections": [{"disease_class": "bud rot", "class_id": 1, "confidence": 0.8}]
            },
            {
                "image_id": "img2.jpg",
                "observation_status": ObservationStatus.NO_DISEASE_DETECTED,
                "accepted_detection_count": 0,
                "relative_affected_area_proxy": 0.0,
                "accepted_detections": []
            },
            {
                "image_id": "img3.jpg",
                "observation_status": ObservationStatus.INVALID_INPUT,
                "accepted_detection_count": 0,
                "relative_affected_area_proxy": 0.0,
                "accepted_detections": []
            }
        ]
        aggregator = PlantationHealthAggregator()
        report = aggregator.generate_plantation_report(records)

        counts = report["observation_counts"]
        self.assertEqual(counts["total_images_submitted"], 3)
        self.assertEqual(counts["successfully_processed_images"], 2)
        self.assertEqual(counts["processing_failures"], 1)
        self.assertEqual(counts["disease_positive_images"], 1)
        self.assertEqual(counts["disease_negative_images"], 1)

    # 6. Disease distribution
    def test_disease_distribution(self):
        """Test calculation of disease proportion and decoupled image-affected counts."""
        records = [
            {
                "image_id": "img1.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detections": [
                    {"disease_class": "gray leaf spot", "class_id": 2, "confidence": 0.8},
                    {"disease_class": "gray leaf spot", "class_id": 2, "confidence": 0.85}
                ]
            },
            {
                "image_id": "img2.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detections": [
                    {"disease_class": "stembleeding", "class_id": 4, "confidence": 0.9}
                ]
            }
        ]
        dist = calculate_disease_distribution(records)
        gls = dist["class_distributions"]["gray leaf spot"]
        sb = dist["class_distributions"]["stembleeding"]

        # Gray leaf spot: 2 detections in 1 image
        self.assertEqual(gls["detection_count"], 2)
        self.assertEqual(gls["unique_images_affected"], 1)
        self.assertAlmostEqual(gls["percentage_of_total_detections"], 66.67, places=1)
        self.assertAlmostEqual(gls["percentage_of_images_affected"], 50.0, places=1)

        # Stembleeding: 1 detection in 1 image
        self.assertEqual(sb["detection_count"], 1)
        self.assertEqual(sb["unique_images_affected"], 1)
        self.assertAlmostEqual(sb["percentage_of_total_detections"], 33.33, places=1)
        self.assertAlmostEqual(sb["percentage_of_images_affected"], 50.0, places=1)

    # 7. Health-index calculation
    def test_health_index_calculation(self):
        """Test Plantation Health Index formula correctness and bounds [0, 100]."""
        aggregator = PlantationHealthAggregator(weight_incidence=0.85, weight_area_proxy=0.15)
        
        # 1 positive out of 4 (incidence = 0.25), mean area proxy = 0.05
        records = [
            {
                "image_id": "pos.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detection_count": 1,
                "relative_affected_area_proxy": 0.20,
                "accepted_detections": [{"disease_class": "leaf rot", "class_id": 3, "confidence": 0.8}]
            }
        ] + [
            {
                "image_id": f"neg_{i}.jpg",
                "observation_status": ObservationStatus.NO_DISEASE_DETECTED,
                "accepted_detection_count": 0,
                "relative_affected_area_proxy": 0.0,
                "accepted_detections": []
            }
            for i in range(3)
        ]
        report = aggregator.generate_plantation_report(records)
        phi_inc = report["plantation_health_index"]["phi_incidence_only_score"]
        phi_comp = report["plantation_health_index"]["phi_composite_score"]

        # Incidence: 100 * (1 - 0.25) = 75.0
        self.assertEqual(phi_inc, 75.0)
        # Composite: Mean area = (0.20 + 0 + 0 + 0)/4 = 0.05.
        # Penalty = 0.85 * 0.25 + 0.15 * 0.05 = 0.2125 + 0.0075 = 0.22.
        # PHI_comp = 100 * (1 - 0.22) = 78.0
        self.assertEqual(phi_comp, 78.0)

    # 8. Empty detections
    def test_empty_detections(self):
        """Test edge case where zero detections occur (100% healthy block)."""
        aggregator = PlantationHealthAggregator()
        records = [
            {
                "image_id": f"healthy_{i}.jpg",
                "observation_status": ObservationStatus.NO_DISEASE_DETECTED,
                "accepted_detection_count": 0,
                "relative_affected_area_proxy": 0.0,
                "accepted_detections": []
            }
            for i in range(5)
        ]
        report = aggregator.generate_plantation_report(records)
        self.assertEqual(report["observation_counts"]["disease_positive_images"], 0)
        self.assertEqual(report["observation_rates"]["disease_positive_image_percentage"], 0.0)
        self.assertEqual(report["plantation_health_index"]["phi_composite_score"], 100.0)
        self.assertEqual(report["plantation_health_index"]["phi_incidence_only_score"], 100.0)
        self.assertEqual(report["plantation_health_index"]["health_tier"], "EXCELLENT / MINIMAL_DISEASE_PRESSURE")

    # 9. Low-confidence detections
    def test_low_confidence_detections(self):
        """Test that sub-threshold detections do not inflate confirmed disease counts."""
        records = [
            {
                "image_id": "candidate.jpg",
                "observation_status": ObservationStatus.LOW_CONFIDENCE_CANDIDATE,
                "accepted_detection_count": 0,
                "candidate_detection_count": 1,
                "relative_affected_area_proxy": 0.0,
                "accepted_detections": [],
                "candidate_detections": [{"disease_class": "bud rot", "class_id": 1, "confidence": 0.15}]
            }
        ]
        aggregator = PlantationHealthAggregator()
        report = aggregator.generate_plantation_report(records)
        self.assertEqual(report["observation_counts"]["disease_positive_images"], 0)
        self.assertEqual(report["observation_counts"]["low_confidence_candidate_images"], 1)
        self.assertEqual(report["plantation_health_index"]["phi_incidence_only_score"], 100.0)

    # 10. Invalid boxes
    def test_invalid_boxes(self):
        """Test schema validation rejects inverted, negative, or malformed bounding boxes."""
        # Negative coord
        self.assertFalse(validate_detection_record({
            "class_id": 0, "disease_class": "bud root dropping", "confidence": 0.5,
            "bounding_box": {"x1": -5.0, "y1": 10.0, "x2": 20.0, "y2": 20.0}
        }))
        # Inverted x2 < x1
        self.assertFalse(validate_detection_record({
            "class_id": 0, "disease_class": "bud root dropping", "confidence": 0.5,
            "bounding_box": {"x1": 50.0, "y1": 10.0, "x2": 20.0, "y2": 20.0}
        }))
        # Invalid confidence > 1.0
        self.assertFalse(validate_detection_record({
            "class_id": 0, "disease_class": "bud root dropping", "confidence": 1.05,
            "bounding_box": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 20.0}
        }))

    # 11. Multiple diseases in one image
    def test_multiple_diseases_in_one_image(self):
        """Test proper multi-class handling in a single image observation."""
        records = [
            {
                "image_id": "comorbid_palm.jpg",
                "observation_status": ObservationStatus.DISEASE_DETECTED,
                "accepted_detections": [
                    {"disease_class": "bud rot", "class_id": 1, "confidence": 0.75},
                    {"disease_class": "leaf rot", "class_id": 3, "confidence": 0.82}
                ]
            }
        ]
        dist = calculate_disease_distribution(records)
        self.assertEqual(dist["total_accepted_detections"], 2)
        self.assertEqual(dist["class_distributions"]["bud rot"]["unique_images_affected"], 1)
        self.assertEqual(dist["class_distributions"]["leaf rot"]["unique_images_affected"], 1)
        self.assertEqual(dist["class_distributions"]["stembleeding"]["unique_images_affected"], 0)

    # 12. Repeated observations
    def test_repeated_observations(self):
        """Test multi-visit temporal data tracking across successive sessions."""
        aggregator = PlantationHealthAggregator()
        records_t1 = [{
            "image_id": "tree_1.jpg", "plantation_id": "BLOCK_A",
            "observation_status": ObservationStatus.NO_DISEASE_DETECTED,
            "accepted_detection_count": 0, "relative_affected_area_proxy": 0.0, "accepted_detections": []
        }]
        records_t2 = [{
            "image_id": "tree_1.jpg", "plantation_id": "BLOCK_A",
            "observation_status": ObservationStatus.DISEASE_DETECTED,
            "accepted_detection_count": 1, "relative_affected_area_proxy": 0.12,
            "accepted_detections": [{"disease_class": "stembleeding", "class_id": 4, "confidence": 0.88}]
        }]

        report_t1 = aggregator.generate_plantation_report(records_t1, plantation_id="BLOCK_A", observation_session_id="VISIT_1")
        report_t2 = aggregator.generate_plantation_report(records_t2, plantation_id="BLOCK_A", observation_session_id="VISIT_2")

        self.assertEqual(report_t1["plantation_health_index"]["phi_composite_score"], 100.0)
        self.assertLess(report_t2["plantation_health_index"]["phi_composite_score"], 100.0)
        self.assertEqual(report_t1["meta"]["observation_session_id"], "VISIT_1")
        self.assertEqual(report_t2["meta"]["observation_session_id"], "VISIT_2")

    # 13. Validator accounting invariants and regression protection
    def test_validator_accounting_invariants(self):
        """Test validator dynamically computes total checks and enforces accounting invariants."""
        from scripts.validate_health_monitoring import HealthMonitoringValidator
        import json
        from pathlib import Path
        import copy

        records_path = Path("outputs/health_monitoring/image_health_records.json")
        report_path = Path("outputs/health_monitoring/plantation_health_report.json")
        
        validator = HealthMonitoringValidator(records_path, report_path)
        total_registered = len(validator.checks)
        self.assertEqual(total_registered, 11)

        with open(records_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Baseline: Valid data
        summary_valid = validator.validate_data(records, report)
        self.assertEqual(summary_valid["total_checks"], total_registered)
        self.assertEqual(summary_valid["passed_checks"], total_registered)
        self.assertEqual(summary_valid["failed_checks"], 0)
        self.assertTrue(summary_valid["is_valid"])
        
        # Structural Invariant Assertions
        self.assertEqual(summary_valid["passed_checks"] + summary_valid["failed_checks"], summary_valid["total_checks"])
        self.assertLessEqual(summary_valid["passed_checks"], summary_valid["total_checks"])
        self.assertGreaterEqual(summary_valid["failed_checks"], 0)

        # Fault Injection: Corrupted record with invalid confidence > 1.0
        corrupted_records = copy.deepcopy(records)
        # Add an out-of-bounds detection
        if not corrupted_records[0].get("accepted_detections"):
            corrupted_records[0]["accepted_detections"] = []
        corrupted_records[0]["accepted_detections"].append({
            "class_id": 3,
            "disease_class": "leaf rot",
            "confidence": 1.75,  # Out-of-bounds confidence
            "bounding_box": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}
        })
        
        summary_corrupted = validator.validate_data(corrupted_records, report)
        self.assertFalse(summary_corrupted["is_valid"])
        self.assertGreater(summary_corrupted["failed_checks"], 0)
        self.assertLess(summary_corrupted["passed_checks"], summary_corrupted["total_checks"])
        
        # Accounting invariants MUST still hold strictly even on failure
        self.assertEqual(summary_corrupted["passed_checks"] + summary_corrupted["failed_checks"], summary_corrupted["total_checks"])
        self.assertLessEqual(summary_corrupted["passed_checks"], summary_corrupted["total_checks"])
        self.assertGreaterEqual(summary_corrupted["failed_checks"], 0)

if __name__ == "__main__":
    unittest.main()
