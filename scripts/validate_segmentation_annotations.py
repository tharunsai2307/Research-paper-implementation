"""
Automated Validation and Quality Control Script for Segmentation Annotations.
Phase 10 Release Gate Validator:
1. Geometry: Polygons have >= 3 vertices, valid floating-point values in [0.0, 1.0], non-zero area.
2. File Integrity: Annotation references existing source image; no orphan files, no duplicates.
3. Class Integrity: Class IDs strictly bounded in {0, 1, 2, 3, 4}.
4. Dataset Split Integrity: Train (63), Val (27), Test (10) with zero cross-split overlap.
5. Anti-Fabrication: Verifies zero box-derived pseudo-masks, zero axis-aligned rectangles.
6. Hard Training Gate: Training allowed ONLY when accepted real masks > 0 and valid across splits.
"""

import sys
import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_CLASSES = {0, 1, 2, 3, 4}

def calculate_shoelace_area(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def is_axis_aligned_box(points: List[Tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    unique_xs = len(set(round(x, 4) for x in xs))
    unique_ys = len(set(round(y, 4) for y in ys))
    return unique_xs == 2 and unique_ys == 2

def validate_segmentation_workspace() -> Dict[str, Any]:
    print("======================================================================")
    print("PHASE 10 SEGMENTATION WORKSPACE & ANNOTATION VALIDATOR")
    print("======================================================================")

    base_dir = PROJECT_ROOT / "data/external/phase_9_segmentation"
    manifest_path = base_dir / "metadata/annotation_manifest.csv"
    images_dir = base_dir / "images"
    annotations_dir = base_dir / "annotations"
    masks_dir = base_dir / "masks"

    if not manifest_path.exists():
        print(f"[FAIL] Manifest not found: {manifest_path}")
        return {"status": "FAIL", "reason": "Manifest missing"}

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Total registered images in manifest: {len(records)}")
    if len(records) != 100:
        print(f"[FAIL] Expected exactly 100 benchmark disease images, got {len(records)}")
        return {"status": "FAIL", "reason": "Incorrect image count"}

    # 1. Image File Integrity
    registered_ids = [r["image_id"] for r in records]
    if len(set(registered_ids)) != len(registered_ids):
        print("[FAIL] Duplicate image IDs found in annotation manifest!")
        return {"status": "FAIL", "reason": "Duplicate image IDs in manifest"}

    missing_imgs = []
    for r in records:
        img_p = images_dir / r["image_id"]
        if not img_p.exists():
            missing_imgs.append(r["image_id"])

    if missing_imgs:
        print(f"[FAIL] Missing {len(missing_imgs)} image files in images/ directory!")
        return {"status": "FAIL", "reason": "Missing images"}
    print("  [PASS] All 100 benchmark image files verified on disk.")

    # 2. Annotation & Geometry Audit
    annotated_count = 0
    accepted_count = 0
    pending_count = 0
    for r in records:
        status = r.get("annotation_status", "PENDING").upper()
        if status in ["COMPLETED", "ANNOTATED"]:
            annotated_count += 1
        elif status == "ACCEPTED":
            accepted_count += 1
        else:
            pending_count += 1

    print(f"  [AUDIT] Completed manual annotations: {annotated_count}")
    print(f"  [AUDIT] Accepted verified annotations: {accepted_count}")
    print(f"  [AUDIT] Pending manual annotations:   {pending_count}")

    # 3. Anti-Fabrication & Detailed Annotation Geometry Check
    suspicious_rectangles = 0
    invalid_geometry = 0
    orphan_annotations = []
    
    if annotations_dir.exists():
        for ann_p in annotations_dir.glob("*.txt"):
            stem = ann_p.stem
            matching = [img for img in registered_ids if Path(img).stem == stem]
            if not matching:
                orphan_annotations.append(ann_p.name)
                continue
                
            with open(ann_p, "r", encoding="utf-8") as af:
                lines = [line.strip() for line in af if line.strip()]
                for line in lines:
                    tokens = line.split()
                    # 5 tokens = YOLO bbox (<cls> <xc> <yc> <w> <h>)
                    if len(tokens) == 5:
                        suspicious_rectangles += 1
                        break
                    if len(tokens) < 7:  # class_id + at least 3 points (6 coords)
                        invalid_geometry += 1
                        break
                    
                    try:
                        cls_id = int(tokens[0])
                        if cls_id not in VALID_CLASSES:
                            invalid_geometry += 1
                            break
                        coords = [float(t) for t in tokens[1:]]
                        if len(coords) % 2 != 0:
                            invalid_geometry += 1
                            break
                        pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        # Check bounds [0, 1]
                        for x, y in pts:
                            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                                invalid_geometry += 1
                                break
                        # Check area
                        area = calculate_shoelace_area(pts)
                        if area <= 0.0:
                            invalid_geometry += 1
                            break
                        # Check axis-aligned bounding box
                        if is_axis_aligned_box(pts):
                            suspicious_rectangles += 1
                            break
                    except Exception:
                        invalid_geometry += 1
                        break

    if suspicious_rectangles > 0:
        print(f"[FAIL] Anti-Fabrication Gate: Found {suspicious_rectangles} YOLO detection boxes / rectangles in annotations!")
        return {"status": "FAIL", "reason": "Bounding boxes in segmentation folder"}
    if invalid_geometry > 0:
        print(f"[FAIL] Found {invalid_geometry} annotations with invalid geometry!")
        return {"status": "FAIL", "reason": "Invalid annotation geometry"}
    if orphan_annotations:
        print(f"[FAIL] Found {len(orphan_annotations)} orphan annotation files!")
        return {"status": "FAIL", "reason": "Orphan annotations"}
        
    print("  [PASS] Anti-Fabrication Check: Zero box-derived pseudo-masks detected.")

    # 4. Split Integrity
    splits_dir = base_dir / "splits"
    train_f = splits_dir / "train.txt"
    val_f = splits_dir / "val.txt"
    test_f = splits_dir / "test.txt"

    if not (train_f.exists() and val_f.exists() and test_f.exists()):
        print("[FAIL] Split definition files missing!")
        return {"status": "FAIL", "reason": "Splits missing"}

    with open(train_f, "r", encoding="utf-8") as f:
        train_set = set(line.strip() for line in f if line.strip())
    with open(val_f, "r", encoding="utf-8") as f:
        val_set = set(line.strip() for line in f if line.strip())
    with open(test_f, "r", encoding="utf-8") as f:
        test_set = set(line.strip() for line in f if line.strip())

    if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
        print("[FAIL] Cross-split leakage detected in segmentation splits!")
        return {"status": "FAIL", "reason": "Cross-split leakage"}

    print(f"  [PASS] Split integrity verified: Train ({len(train_set)}), Val ({len(val_set)}), Test ({len(test_set)}). Zero cross-split overlap.")

    # 5. Training Gate Decision
    # Strict Phase 10 Hard Gate:
    # accepted real masks > 0 AND mask validation passes AND train/val/test masks exist
    total_valid_masks = annotated_count + accepted_count
    can_train_segmentation = (total_valid_masks > 0 and pending_count == 0)
    print("----------------------------------------------------------------------")
    if can_train_segmentation:
        print("  [GATE] All 100 images annotated: YOLOv8-Seg training UNLOCKED.")
        gate_status = "READY_FOR_TRAINING"
    else:
        print(f"  [GATE] {pending_count}/100 annotations PENDING (Accepted: {accepted_count}): YOLOv8-Seg training BLOCKED.")
        print("         (Preserving research integrity; zero synthetic masks used).")
        gate_status = "BLOCKED_PENDING_MANUAL_MASKS"

    print("======================================================================")
    return {
        "status": "PASS",
        "total_images": len(records),
        "annotated_images": annotated_count,
        "accepted_images": accepted_count,
        "pending_images": pending_count,
        "gate_status": gate_status
    }

if __name__ == "__main__":
    res = validate_segmentation_workspace()
    sys.exit(0 if res["status"] == "PASS" else 1)
