"""
Analysis script to compare YOLO Detection Bounding Box Area vs True Mask Area.
Formulates the exact ratio metrics and provides statistical comparison
when segmentation masks are available, or documents theoretical bound on existing data.
"""

from typing import List, Dict, Any, Optional
import numpy as np

def compute_box_vs_mask_comparison(
    boxes: List[List[float]],
    polygon_points: Optional[List[List[float]]] = None,
    image_width: int = 768,
    image_height: int = 1024
) -> Dict[str, Any]:
    """
    Compares 2D bounding box footprint against fine-grained polygon lesion boundary.
    box format: [x1, y1, x2, y2]
    """
    img_area = float(image_width * image_height)
    
    # 1. Bounding Box Area (Sum and Union)
    box_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes if b[2] > b[0] and b[3] > b[1]]
    total_box_area = sum(box_areas)
    box_coverage_ratio = total_box_area / img_area if img_area > 0 else 0.0

    # 2. Polygon Mask Area (if available)
    if polygon_points and len(polygon_points) >= 3:
        # Shoelace formula
        n = len(polygon_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon_points[i][0] * polygon_points[j][1]
            area -= polygon_points[j][0] * polygon_points[i][1]
        mask_area = abs(area) / 2.0
        # If coordinates are normalized in [0, 1], scale by image area
        if all(0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0 for pt in polygon_points):
            mask_area = mask_area * img_area

        mask_coverage_ratio = mask_area / img_area if img_area > 0 else 0.0
        box_to_mask_ratio = total_box_area / mask_area if mask_area > 0 else 1.0
        status = "MEASURED"
    else:
        mask_area = None
        mask_coverage_ratio = None
        box_to_mask_ratio = None
        status = "PENDING_REAL_MASK_DATA"

    return {
        "status": status,
        "image_dimensions": [image_width, image_height],
        "image_area_px": img_area,
        "total_bounding_box_area_px": round(total_box_area, 2),
        "bounding_box_coverage_ratio": round(box_coverage_ratio, 6),
        "bounding_box_coverage_pct": round(box_coverage_ratio * 100.0, 4),
        "mask_area_px": round(mask_area, 2) if mask_area is not None else None,
        "mask_coverage_ratio": round(mask_coverage_ratio, 6) if mask_coverage_ratio is not None else None,
        "mask_coverage_pct": round(mask_coverage_ratio * 100.0, 4) if mask_coverage_ratio is not None else None,
        "box_to_mask_inflation_factor": round(box_to_mask_ratio, 2) if box_to_mask_ratio is not None else None,
        "notes": (
            "Bounding boxes systematically overestimate lesion area because rectangles capture "
            "interstitial healthy leaflets and background sky. Real polygon masks delineate exact necrotic boundaries."
        )
    }
