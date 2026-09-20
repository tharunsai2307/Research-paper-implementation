"""
Phase 8 Longitudinal Dataset Schema & Validation Engine.
Defines machine-readable data structures, specimen tracking contracts,
and strict temporal split validation to prevent cross-partition leakage.
"""

from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime

class CameraMetadata(BaseModel):
    distance_meters: float = Field(default=2.5, ge=0.5, le=20.0)
    elevation_angle_deg: float = Field(default=45.0, ge=0.0, le=90.0)
    focal_length_mm: Optional[float] = None

class EnvironmentalCovariates(BaseModel):
    mean_temp_celsius_7d: Optional[float] = None
    mean_relative_humidity_pct_7d: Optional[float] = None
    cumulative_rainfall_mm_7d: Optional[float] = None
    canopy_wetness_hours_7d: Optional[float] = None

class LongitudinalObservationRecord(BaseModel):
    observation_id: str
    tree_id: str
    site_id: str = "DEFAULT_ORCHARD_01"
    timestamp: str
    day_index: int = Field(ge=0)
    image_path: str
    disease_class: Optional[str] = None
    detection_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    severity_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    camera_metadata: Optional[CameraMetadata] = None
    environmental_metadata: Optional[EnvironmentalCovariates] = None
    annotation_status: str = "UNVERIFIED"

    def parsed_datetime(self) -> datetime:
        return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))

class LongitudinalDatasetAuditResult(BaseModel):
    total_records: int
    unique_trees: int
    total_observation_days: int
    leakage_free_split_possible: bool
    train_trees: List[str]
    val_trees: List[str]
    test_trees: List[str]
    violations: List[str]

def audit_temporal_partitioning(
    train_records: List[LongitudinalObservationRecord],
    val_records: List[LongitudinalObservationRecord],
    test_records: List[LongitudinalObservationRecord]
) -> LongitudinalDatasetAuditResult:
    """
    Verifies that zero specimen IDs cross partition boundaries.
    Guarantees mathematically leakage-free longitudinal dataset partitions.
    """
    train_trees: Set[str] = {r.tree_id for r in train_records}
    val_trees: Set[str] = {r.tree_id for r in val_records}
    test_trees: Set[str] = {r.tree_id for r in test_records}

    violations = []
    # Intersection checks
    train_val_overlap = train_trees.intersection(val_trees)
    if train_val_overlap:
        violations.append(f"Temporal Leakage: {len(train_val_overlap)} tree(s) shared between Train and Val: {list(train_val_overlap)[:3]}")

    train_test_overlap = train_trees.intersection(test_trees)
    if train_test_overlap:
        violations.append(f"Temporal Leakage: {len(train_test_overlap)} tree(s) shared between Train and Test: {list(train_test_overlap)[:3]}")

    val_test_overlap = val_trees.intersection(test_trees)
    if val_test_overlap:
        violations.append(f"Temporal Leakage: {len(val_test_overlap)} tree(s) shared between Val and Test: {list(val_test_overlap)[:3]}")

    all_trees = train_trees.union(val_trees).union(test_trees)
    total_records = len(train_records) + len(val_records) + len(test_records)

    return LongitudinalDatasetAuditResult(
        total_records=total_records,
        unique_trees=len(all_trees),
        total_observation_days=len({r.day_index for r in (train_records + val_records + test_records)}),
        leakage_free_split_possible=len(violations) == 0,
        train_trees=sorted(list(train_trees)),
        val_trees=sorted(list(val_trees)),
        test_trees=sorted(list(test_trees)),
        violations=violations
    )
