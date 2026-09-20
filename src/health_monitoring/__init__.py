"""
Package initialization for the Plantation Health Monitoring Engine.
"""

from .schemas import (
    CLASS_NAMES,
    ObservationStatus,
    DEFAULT_OPERATIONAL_THRESHOLD,
    DEFAULT_CANDIDATE_THRESHOLD
)
from .severity_proxy import (
    box_iou,
    calculate_union_area,
    calculate_relative_affected_area_proxy
)
from .inference import HealthInferenceEngine
from .disease_distribution import calculate_disease_distribution
from .plantation_health import PlantationHealthAggregator

__all__ = [
    "CLASS_NAMES",
    "ObservationStatus",
    "DEFAULT_OPERATIONAL_THRESHOLD",
    "DEFAULT_CANDIDATE_THRESHOLD",
    "box_iou",
    "calculate_union_area",
    "calculate_relative_affected_area_proxy",
    "HealthInferenceEngine",
    "calculate_disease_distribution",
    "PlantationHealthAggregator"
]
