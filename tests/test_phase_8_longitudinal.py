"""
Unit Tests for Phase 8 Longitudinal Schema and Partitioning Audit.
"""

import unittest
from src.temporal.dataset_schema import (
    LongitudinalObservationRecord,
    audit_temporal_partitioning
)

class TestPhase8LongitudinalSchema(unittest.TestCase):

    def test_record_validation_and_iso_parsing(self):
        """Test longitudinal record instantiates and parses datetime correctly."""
        rec = LongitudinalObservationRecord(
            observation_id="OBS-01",
            tree_id="TREE-101",
            timestamp="2026-10-01T08:00:00Z",
            day_index=0,
            image_path="dummy.jpg",
            severity_percentage=12.5
        )
        self.assertEqual(rec.tree_id, "TREE-101")
        self.assertEqual(rec.day_index, 0)
        dt = rec.parsed_datetime()
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 10)

    def test_leakage_free_split_audit_pass(self):
        """Test clean disjoint specimen partitioning returns zero violations."""
        train_recs = [
            LongitudinalObservationRecord(observation_id="O1", tree_id="T1", timestamp="2026-10-01T08:00:00Z", day_index=0, image_path="t1_d0.jpg"),
            LongitudinalObservationRecord(observation_id="O2", tree_id="T1", timestamp="2026-10-08T08:00:00Z", day_index=7, image_path="t1_d7.jpg")
        ]
        val_recs = [
            LongitudinalObservationRecord(observation_id="O3", tree_id="T2", timestamp="2026-10-01T08:00:00Z", day_index=0, image_path="t2_d0.jpg")
        ]
        test_recs = [
            LongitudinalObservationRecord(observation_id="O4", tree_id="T3", timestamp="2026-10-01T08:00:00Z", day_index=0, image_path="t3_d0.jpg")
        ]
        res = audit_temporal_partitioning(train_recs, val_recs, test_recs)
        self.assertTrue(res.leakage_free_split_possible)
        self.assertEqual(len(res.violations), 0)
        self.assertEqual(res.unique_trees, 3)

    def test_leakage_detection_violation(self):
        """Test shared specimen between train and test is flagged immediately as leakage."""
        train_recs = [
            LongitudinalObservationRecord(observation_id="O1", tree_id="T1", timestamp="2026-10-01T08:00:00Z", day_index=0, image_path="t1_d0.jpg")
        ]
        val_recs = [
            LongitudinalObservationRecord(observation_id="O2", tree_id="T2", timestamp="2026-10-01T08:00:00Z", day_index=0, image_path="t2_d0.jpg")
        ]
        test_recs = [
            # T1 also present in test split -> LEAKAGE!
            LongitudinalObservationRecord(observation_id="O3", tree_id="T1", timestamp="2026-10-08T08:00:00Z", day_index=7, image_path="t1_d7.jpg")
        ]
        res = audit_temporal_partitioning(train_recs, val_recs, test_recs)
        self.assertFalse(res.leakage_free_split_possible)
        self.assertGreater(len(res.violations), 0)
        self.assertIn("Temporal Leakage", res.violations[0])

if __name__ == "__main__":
    unittest.main()
