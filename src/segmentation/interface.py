"""
Abstract Base Segmentation Model Interface.
Enables plug-and-play integration for future instance/semantic segmentation models
while safely falling back to Phase 5 sweep-line bounding box proxy when masks are pending.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from src.segmentation.schemas import PolygonAnnotation, calculate_segmented_severity_estimate
from src.health_monitoring.severity_proxy import calculate_relative_affected_area_proxy

class AbstractBaseSegmentationModel(ABC):
    """Abstract contract for coconut disease segmentation models."""

    @abstractmethod
    def segment_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Executes segmentation on input image.
        Returns dictionary containing polygons, binary mask, and coverage metrics.
        """
        pass

class FallbackBoundingBoxSegmentationProxy(AbstractBaseSegmentationModel):
    """
    Fallback implementation adhering to Step 3 (Case B).
    Preserves existing Phase 5 YOLO bounding box area proxy while real masks are pending.
    """

    def __init__(self, operational_threshold: float = 0.25):
        self.operational_threshold = operational_threshold
        self.status = "PENDING_REAL_MASK_DATA"

    def segment_image(self, image: np.ndarray, detections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        h, w = image.shape[:2]
        boxes = []
        if detections:
            for d in detections:
                if d.get("confidence", 0.0) >= self.operational_threshold:
                    b = d.get("bounding_box", {})
                    if "x1" in b:
                        boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])

        proxy_ratio = calculate_relative_affected_area_proxy(boxes, w, h)
        
        return {
            "status": self.status,
            "segmentation_status": self.status,
            "polygons": [],
            "mask_available": False,
            "fallback_used": "Phase_5_Sweep_Line_Bounding_Box_Proxy",
            "image_space_severity_ratio": proxy_ratio,
            "image_space_severity_percentage": round(proxy_ratio * 100.0, 4),
            "disclaimer": (
                "Notice: Genuine pixel segmentation masks are currently unavailable in the benchmark dataset. "
                "Severity estimation is provided by the Phase 5 relative 2D image-space bounding box union proxy."
            )
        }
