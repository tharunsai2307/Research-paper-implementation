"""
Phase 7 Segmentation Data Contracts and Geometry Utilities.
Provides structured representation for polygon annotations, binary masks,
and fine-grained lesion area calculation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class PolygonAnnotation:
    """Represents a closed 2D polygon segmentation boundary."""
    def __init__(self, points: List[Tuple[float, float]], class_id: int, class_name: str, confidence: float = 1.0):
        if len(points) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {len(points)}")
        self.points = [(float(x), float(y)) for x, y in points]
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.confidence = float(confidence)

    @property
    def num_vertices(self) -> int:
        return len(self.points)

    def calculate_polygon_area(self) -> float:
        """
        Computes 2D polygon area using the Shoelace formula (Gauss's area formula).
        Guarantees non-negative result.
        """
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns [xmin, ymin, xmax, ymax] enclosing bounding box."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "num_vertices": self.num_vertices,
            "area_px": round(self.calculate_polygon_area(), 2),
            "points": self.points
        }

def calculate_segmented_severity_estimate(
    lesion_mask: np.ndarray,
    canopy_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates image-based severity estimate using fine-grained segmentation masks.
    Supports either full-image denominator or visible plant canopy denominator.
    """
    if lesion_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {lesion_mask.shape}")
    
    h, w = lesion_mask.shape
    total_image_pixels = h * w
    lesion_pixels = int(np.count_nonzero(lesion_mask))

    if canopy_mask is not None:
        if canopy_mask.shape != lesion_mask.shape:
            raise ValueError(f"Canopy mask shape {canopy_mask.shape} does not match lesion mask {lesion_mask.shape}")
        canopy_pixels = int(np.count_nonzero(canopy_mask))
        if canopy_pixels > 0:
            severity_ratio = lesion_pixels / canopy_pixels
            denominator_type = "VISIBLE_CANOPY_PIXELS"
            denom_pixels = canopy_pixels
        else:
            severity_ratio = 0.0
            denominator_type = "CANOPY_MASK_EMPTY_FALLBACK_FULL_FRAME"
            denom_pixels = total_image_pixels
    else:
        severity_ratio = lesion_pixels / total_image_pixels if total_image_pixels > 0 else 0.0
        denominator_type = "FULL_IMAGE_FRAME_PIXELS"
        denom_pixels = total_image_pixels

    severity_ratio = min(1.0, max(0.0, float(severity_ratio)))
    
    return {
        "lesion_pixels": lesion_pixels,
        "denominator_pixels": denom_pixels,
        "denominator_type": denominator_type,
        "image_space_severity_ratio": round(severity_ratio, 6),
        "image_space_severity_percentage": round(severity_ratio * 100.0, 4),
        "disclaimer": (
            "Image-based severity estimate computed via photographic segmentation mask. "
            "Represents 2D visible lesion coverage and does not constitute clinical or certified agronomic staging."
        )
    }
