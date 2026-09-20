"""
Unit Test Suite for Phase 7:
Covers:
1. Polygon geometry and Shoelace area calculation
2. Segmentation mask severity calculations and canopy denominator
3. Fallback segmentation proxy execution
4. Monotonic chronological observation sequence sorting and feature matrix extraction
5. Baseline forecasting models (LOCF and Linear Slope Projection)
6. PyTorch LSTM and GRU tensor forward pass (using synthetic testing fixtures)
7. Progression Alert Engine decision logic across all four alert tiers
"""

import unittest
import numpy as np
import torch
from datetime import datetime

from src.segmentation.schemas import PolygonAnnotation, calculate_segmented_severity_estimate
from src.segmentation.interface import FallbackBoundingBoxSegmentationProxy
from src.temporal.features import (
    PalmTemporalObservation,
    TemporalSequenceExtractor,
    BaselineProgressionForecaster
)
from src.temporal.models import CoconutProgressionLSTM, CoconutProgressionGRU
from src.alerts.engine import ProgressionAlertEngine, AlertLevel, ProgressionTrajectory

class TestPhase7Extensions(unittest.TestCase):

    # -------------------------------------------------------------
    # 1. Segmentation Geometry & Schemas
    # -------------------------------------------------------------
    def test_polygon_geometry_shoelace(self):
        """Test Shoelace formula computes exact polygon area."""
        # Right triangle with base 10, height 20 -> Area = 100.0
        triangle_pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 20.0)]
        poly = PolygonAnnotation(triangle_pts, class_id=3, class_name="leaf rot")
        self.assertAlmostEqual(poly.calculate_polygon_area(), 100.0)
        self.assertEqual(poly.num_vertices, 3)
        self.assertEqual(poly.bounding_box(), (0.0, 0.0, 10.0, 20.0))

        # Square with side 20 -> Area = 400.0
        square_pts = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        poly_sq = PolygonAnnotation(square_pts, class_id=1, class_name="bud rot")
        self.assertAlmostEqual(poly_sq.calculate_polygon_area(), 400.0)

    def test_polygon_invalid_vertices(self):
        """Test polygon rejects fewer than 3 vertices."""
        with self.assertRaises(ValueError):
            PolygonAnnotation([(0.0, 0.0), (10.0, 10.0)], class_id=0, class_name="bud root dropping")

    def test_segmented_severity_canopy_denominator(self):
        """Test segmented severity calculation with full-frame vs canopy mask."""
        h, w = 100, 100
        lesion_mask = np.zeros((h, w), dtype=np.uint8)
        lesion_mask[10:30, 10:30] = 1 # 400 pixels
        
        # 1. Full frame denominator (10,000 pixels) -> 4%
        res_full = calculate_segmented_severity_estimate(lesion_mask)
        self.assertEqual(res_full["lesion_pixels"], 400)
        self.assertAlmostEqual(res_full["image_space_severity_ratio"], 0.04)
        self.assertAlmostEqual(res_full["image_space_severity_percentage"], 4.0)

        # 2. Canopy mask denominator (2,000 pixels) -> 20%
        canopy_mask = np.zeros((h, w), dtype=np.uint8)
        canopy_mask[0:50, 0:40] = 1 # 2000 pixels
        res_canopy = calculate_segmented_severity_estimate(lesion_mask, canopy_mask)
        self.assertEqual(res_canopy["denominator_type"], "VISIBLE_CANOPY_PIXELS")
        self.assertAlmostEqual(res_canopy["image_space_severity_ratio"], 0.20)
        self.assertAlmostEqual(res_canopy["image_space_severity_percentage"], 20.0)

    def test_fallback_segmentation_proxy(self):
        """Test FallbackBoundingBoxSegmentationProxy safely uses Phase 5 sweep-line union."""
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        proxy = FallbackBoundingBoxSegmentationProxy(operational_threshold=0.25)
        detections = [
            {"confidence": 0.8, "bounding_box": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}},
            {"confidence": 0.15, "bounding_box": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}} # candidate ignored
        ]
        res = proxy.segment_image(dummy_img, detections)
        self.assertEqual(res["segmentation_status"], "PENDING_REAL_MASK_DATA")
        self.assertEqual(res["fallback_used"], "Phase_5_Sweep_Line_Bounding_Box_Proxy")
        # 100x100 box in 500x500 img -> 10,000 / 250,000 = 0.04 (4.0%)
        self.assertAlmostEqual(res["image_space_severity_ratio"], 0.04)

    # -------------------------------------------------------------
    # 2. Temporal Sequences & Baselines
    # -------------------------------------------------------------
    def test_temporal_sorting_and_features(self):
        """Test strictly monotonic chronological sorting and feature matrix extraction."""
        obs1 = PalmTemporalObservation("obs1", "T001", "2026-06-15T08:00:00Z", severity_ratio=0.02, primary_disease="leaf rot")
        obs2 = PalmTemporalObservation("obs2", "T001", "2026-06-01T08:00:00Z", severity_ratio=0.01, primary_disease="leaf rot")
        obs3 = PalmTemporalObservation("obs3", "T001", "2026-06-22T08:00:00Z", severity_ratio=0.04, primary_disease="leaf rot")

        # Pass out of order
        raw_list = [obs1, obs2, obs3]
        sorted_list = TemporalSequenceExtractor.sort_observations(raw_list)
        self.assertEqual(sorted_list[0].observation_id, "obs2")
        self.assertEqual(sorted_list[1].observation_id, "obs1")
        self.assertEqual(sorted_list[2].observation_id, "obs3")

        feats = TemporalSequenceExtractor.construct_feature_matrix(raw_list)
        self.assertEqual(len(feats), 3)
        self.assertEqual(len(feats[0]), 10) # 10 dimensions
        # Check delta days between obs2 (June 1) and obs1 (June 15) -> 14 days
        self.assertAlmostEqual(feats[1][3], 14.0)

    def test_baseline_locf_forecast(self):
        """Test Last Observation Carried Forward (LOCF) baseline."""
        history = [
            PalmTemporalObservation("o1", "T002", "2026-07-01T08:00:00Z", 0.05),
            PalmTemporalObservation("o2", "T002", "2026-07-08T08:00:00Z", 0.08)
        ]
        res = BaselineProgressionForecaster.forecast_locf(history, horizon_days=7)
        self.assertEqual(res["baseline_model"], "Last_Observation_Carried_Forward_LOCF")
        self.assertEqual(res["predicted_severity"], 0.08)
        self.assertEqual(res["trajectory_status"], "STABLE")

    def test_baseline_linear_slope_forecast(self):
        """Test Linear Velocity Projection baseline."""
        history = [
            PalmTemporalObservation("o1", "T003", "2026-07-01T08:00:00Z", 0.02),
            PalmTemporalObservation("o2", "T003", "2026-07-08T08:00:00Z", 0.06) # +0.04 over 7 days
        ]
        # In 7 days, projects an additional +0.04 -> 0.10
        res = BaselineProgressionForecaster.forecast_linear(history, horizon_days=7)
        self.assertEqual(res["baseline_model"], "Empirical_Linear_Slope_Projection")
        self.assertAlmostEqual(res["predicted_severity"], 0.10, places=4)
        self.assertEqual(res["trajectory_status"], "RAPID_EXPANSION")

    # -------------------------------------------------------------
    # 3. Recurrent Models (Synthetic Test Fixtures)
    # -------------------------------------------------------------
    def test_lstm_and_gru_forward_contracts(self):
        """Verify PyTorch LSTM and GRU input/output tensor contracts with synthetic tensors."""
        batch_size = 4
        seq_len = 5
        input_dim = 10

        synth_x = torch.randn(batch_size, seq_len, input_dim)

        lstm = CoconutProgressionLSTM(input_dim=input_dim, hidden_dim=16)
        out_lstm = lstm(synth_x)
        self.assertEqual(out_lstm.shape, (batch_size, 1))
        self.assertTrue(torch.all((out_lstm >= 0.0) & (out_lstm <= 1.0)))

        gru = CoconutProgressionGRU(input_dim=input_dim, hidden_dim=16)
        out_gru = gru(synth_x)
        self.assertEqual(out_gru.shape, (batch_size, 1))
        self.assertTrue(torch.all((out_gru >= 0.0) & (out_gru <= 1.0)))

    # -------------------------------------------------------------
    # 4. Progression Alert Engine Decision Logic
    # -------------------------------------------------------------
    def test_alert_engine_no_alert(self):
        """Test healthy palm triggers NO_ALERT."""
        engine = ProgressionAlertEngine()
        res = engine.evaluate_progression(current_severity=0.0, projected_severity=0.0, primary_disease=None)
        self.assertEqual(res["alert_level"], AlertLevel.NO_ALERT.value)
        self.assertEqual(res["progression_trajectory"], ProgressionTrajectory.STABLE.value)

    def test_alert_engine_monitor(self):
        """Test minimal stable lesion triggers MONITOR."""
        engine = ProgressionAlertEngine()
        res = engine.evaluate_progression(current_severity=0.03, projected_severity=0.032, primary_disease="leaf rot")
        self.assertEqual(res["alert_level"], AlertLevel.MONITOR.value)
        self.assertEqual(res["progression_trajectory"], ProgressionTrajectory.STABLE.value)

    def test_alert_engine_attention(self):
        """Test rapid expansion triggers ATTENTION."""
        engine = ProgressionAlertEngine()
        # Change delta > 0.030
        res = engine.evaluate_progression(current_severity=0.04, projected_severity=0.08, primary_disease="gray leaf spot")
        self.assertEqual(res["alert_level"], AlertLevel.ATTENTION.value)
        self.assertEqual(res["progression_trajectory"], ProgressionTrajectory.RAPID_EXPANSION.value)

    def test_alert_engine_urgent_review_bud_rot(self):
        """Test progressing lethal bud rot triggers URGENT_REVIEW."""
        engine = ProgressionAlertEngine()
        res = engine.evaluate_progression(current_severity=0.03, projected_severity=0.04, primary_disease="bud rot")
        self.assertEqual(res["alert_level"], AlertLevel.URGENT_REVIEW.value)
        self.assertIn("Lethal bud rot detected", res["recommended_action"])

    def test_alert_engine_urgent_review_high_severity(self):
        """Test severe lesion area triggers URGENT_REVIEW."""
        engine = ProgressionAlertEngine()
        res = engine.evaluate_progression(current_severity=0.18, projected_severity=0.20, primary_disease="stembleeding")
        self.assertEqual(res["alert_level"], AlertLevel.URGENT_REVIEW.value)

if __name__ == "__main__":
    unittest.main()
