"""
Human Polygon Annotation Manager, Double-Annotator QC, and Validator.
Phase 11 Production Implementation:
- Adheres to Phase 8, 9, 10, and 11 Phytopathology Annotation Protocols:
  0: bud root dropping
  1: bud rot
  2: gray leaf spot
  3: leaf rot
  4: stembleeding
- Enforces strict geometry:
  - n >= 3 vertices
  - normalized coordinates in [0.0, 1.0]
  - positive non-zero area (Shoelace formula)
  - Anti-Bounding-Box gate: Rejects 4-corner axis-aligned box approximations
- Manages dual annotators (Annotator A vs Annotator B)
- Computes raster-level exact Inter-Annotator IoU and Dice coefficient
- Maintains audit trail in annotation_events.csv and updates annotation_manifest.csv
"""

import os
import sys
import json
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_MAP = {
    0: "bud root dropping",
    1: "bud rot",
    2: "gray leaf spot",
    3: "leaf rot",
    4: "stembleeding"
}
CLASS_TO_ID = {v: k for k, v in CLASS_MAP.items()}

def calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
    """Calculate polygon area in normalized coordinate space using Shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def is_axis_aligned_box(points: List[Tuple[float, float]], tolerance: float = 1e-4) -> bool:
    """Detect if polygon is merely an axis-aligned bounding box rectangle (anti-fabrication check)."""
    if len(points) != 4:
        return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    unique_xs = len(set(round(x, 4) for x in xs))
    unique_ys = len(set(round(y, 4) for y in ys))
    return unique_xs == 2 and unique_ys == 2

def validate_polygon(points: List[Tuple[float, float]]) -> Tuple[bool, str]:
    """Validate polygon geometry against Phase 11 specifications."""
    if len(points) < 3:
        return False, f"Polygon has {len(points)} vertices; minimum 3 required."
    for idx, (x, y) in enumerate(points):
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False, f"Vertex {idx} ({x:.4f}, {y:.4f}) outside [0.0, 1.0] image bounds."
    area = calculate_polygon_area(points)
    if area <= 0.0:
        return False, "Polygon has zero or degenerate area."
    if is_axis_aligned_box(points):
        return False, "REJECTED: Polygon matches an axis-aligned bounding box. Manual lesion tracing required."
    return True, "Valid"

def compute_polygon_iou_dice(
    points_a: List[Tuple[float, float]],
    points_b: List[Tuple[float, float]],
    canvas_size: int = 1000
) -> Tuple[float, float]:
    """
    Computes exact raster-level Intersection-over-Union (IoU) and Dice coefficient
    between two normalized polygons on a canvas_size x canvas_size grid.
    """
    mask_a = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    mask_b = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    pts_a_px = np.array([[int(round(x * (canvas_size - 1))), int(round(y * (canvas_size - 1)))] for x, y in points_a], dtype=np.int32)
    pts_b_px = np.array([[int(round(x * (canvas_size - 1))), int(round(y * (canvas_size - 1)))] for x, y in points_b], dtype=np.int32)

    cv2.fillPoly(mask_a, [pts_a_px], 1)
    cv2.fillPoly(mask_b, [pts_b_px], 1)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    area_a = mask_a.sum()
    area_b = mask_b.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / (area_a + area_b)) if (area_a + area_b) > 0 else 0.0

    return iou, dice

def register_annotation(
    image_id: str,
    class_id: int,
    polygon_points: List[Tuple[float, float]],
    annotator_id: str,
    annotator_role: str = "annotator_A",  # "annotator_A", "annotator_B", "reviewer"
    ai_assisted: bool = False,
    ai_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validates and registers a human polygon annotation.
    Saves canonical YOLO polygon file and updates Phase 11 metadata and events audit.
    """
    if class_id not in CLASS_MAP:
        raise ValueError(f"Invalid class_id {class_id}. Valid: {CLASS_MAP}")

    valid, msg = validate_polygon(polygon_points)
    if not valid:
        raise ValueError(f"Polygon validation failed: {msg}")

    p9_img_path = PROJECT_ROOT / "data/external/phase_9_segmentation/images" / image_id
    if not p9_img_path.exists():
        raise FileNotFoundError(f"Source image {image_id} not found at {p9_img_path}")

    img_sha256 = hashlib.sha256(p9_img_path.read_bytes()).hexdigest()

    p11_dir = PROJECT_ROOT / "data/external/phase_11_segmentation"
    stem = Path(image_id).stem
    ann_dir = p11_dir / f"annotations/{annotator_role}"
    meta_dir = p11_dir / "metadata"
    ann_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    ann_path = ann_dir / f"{stem}.txt"
    meta_json_path = meta_dir / f"{stem}_{annotator_role}.json"
    manifest_path = meta_dir / "annotation_manifest.csv"
    events_path = meta_dir / "annotation_events.csv"

    # Write YOLO segmentation polygon line: <class_id> x1 y1 x2 y2 ...
    flat_coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon_points)
    yolo_line = f"{class_id} {flat_coords}\n"
    with open(ann_path, "w", encoding="utf-8") as f:
        f.write(yolo_line)

    area = calculate_polygon_area(polygon_points)
    now_iso = datetime.now(timezone.utc).isoformat()

    metadata_record = {
        "annotation_id": f"ANN-PHASE11-{stem}-{annotator_role}",
        "image_id": image_id,
        "image_sha256": img_sha256,
        "disease_class": CLASS_MAP[class_id],
        "class_id": class_id,
        "annotator_id": annotator_id,
        "annotator_role": annotator_role,
        "annotation_date": now_iso,
        "mask_format": "POLYGON_NORMALIZED",
        "num_vertices": len(polygon_points),
        "polygon_points": polygon_points,
        "normalized_area": area,
        "ai_assisted": ai_assisted,
        "ai_metadata": ai_metadata or {
            "tool": "MANUAL_DELINEATION",
            "human_inspection": True,
            "human_correction": True,
            "human_approval": True,
            "ground_truth_certified": True
        }
    }

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_record, f, indent=2)

    # Update manifest
    if manifest_path.exists():
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["image_id"] == image_id:
                    if annotator_role == "annotator_A":
                        row["annotator_A_status"] = "COMPLETED"
                        row["annotator_A_path"] = str(ann_path.relative_to(PROJECT_ROOT))
                    elif annotator_role == "annotator_B":
                        row["annotator_B_status"] = "COMPLETED"
                        row["annotator_B_path"] = str(ann_path.relative_to(PROJECT_ROOT))
                rows.append(row)

        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Log event
    if events_path.exists():
        with open(events_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_id", "timestamp", "image_id", "event_type", "annotator_id", "details"])
            writer.writerow({
                "event_id": f"EVT-{int(datetime.now().timestamp() * 1000)}",
                "timestamp": now_iso,
                "image_id": image_id,
                "event_type": f"ANNOTATION_SAVED_{annotator_role.upper()}",
                "annotator_id": annotator_id,
                "details": f"Saved {len(polygon_points)} vertices, normalized area={area:.5f}"
            })

    return metadata_record
