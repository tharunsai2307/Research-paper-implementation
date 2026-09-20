"""
Segmentation package for Phase 7.
"""

from src.segmentation.schemas import PolygonAnnotation, calculate_segmented_severity_estimate
from src.segmentation.interface import AbstractBaseSegmentationModel, FallbackBoundingBoxSegmentationProxy

__all__ = [
    "PolygonAnnotation",
    "calculate_segmented_severity_estimate",
    "AbstractBaseSegmentationModel",
    "FallbackBoundingBoxSegmentationProxy"
]
