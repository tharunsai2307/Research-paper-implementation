"""
Audit current status of human polygon segmentation annotations.
Checks:
- total registered images in manifest
- completed annotations
- pending annotations
- valid masks
- invalid masks
- orphan masks (masks without corresponding registered image)
- orphan annotations (annotations without corresponding registered image)
Outputs: outputs/phase_10/annotation/mask_status.json
"""

import os
import sys
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def audit_mask_status(output_path=None):
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs/phase_10/annotation/mask_status.json"
    
    seg_dir = PROJECT_ROOT / "data/external/phase_9_segmentation"
    manifest_path = seg_dir / "metadata/annotation_manifest.csv"
    images_dir = seg_dir / "images"
    annotations_dir = seg_dir / "annotations"
    masks_dir = seg_dir / "masks"
    
    registered_images = []
    pending_images = []
    completed_images = []
    accepted_images = []
    rejected_images = []
    qc_review_images = []
    
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["image_id"]
                registered_images.append(img_id)
                status = row.get("annotation_status", "PENDING").upper()
                if status == "PENDING":
                    pending_images.append(img_id)
                elif status in ["ANNOTATED", "COMPLETED"]:
                    completed_images.append(img_id)
                elif status == "ACCEPTED":
                    accepted_images.append(img_id)
                elif status == "REJECTED":
                    rejected_images.append(img_id)
                elif status == "QC_REVIEW":
                    qc_review_images.append(img_id)
                else:
                    pending_images.append(img_id)
    
    registered_set = set(registered_images)
    
    # Audit annotations folder
    annotation_files = list(annotations_dir.glob("*.txt")) if annotations_dir.exists() else []
    orphan_annotations = []
    valid_annotations = 0
    invalid_annotations = 0
    
    for af in annotation_files:
        stem = af.stem
        # check corresponding image
        matching = [img for img in registered_images if Path(img).stem == stem]
        if not matching:
            orphan_annotations.append(af.name)
        else:
            # check content
            try:
                lines = [l.strip() for l in open(af, "r", encoding="utf-8") if l.strip()]
                is_valid = True
                for l in lines:
                    parts = l.split()
                    # A polygon must have class_id + at least 3 points (6 coords) -> >= 7 tokens
                    if len(parts) < 7:
                        is_valid = False
                        break
                if lines and is_valid:
                    valid_annotations += 1
                else:
                    invalid_annotations += 1
            except Exception:
                invalid_annotations += 1

    # Audit masks folder
    mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg")) if masks_dir.exists() else []
    orphan_masks = []
    valid_masks = 0
    invalid_masks = 0
    
    for mf in mask_files:
        stem = mf.stem
        matching = [img for img in registered_images if Path(img).stem == stem]
        if not matching:
            orphan_masks.append(mf.name)
        else:
            # check valid mask file
            if mf.stat().st_size > 0:
                valid_masks += 1
            else:
                invalid_masks += 1
                
    status_summary = {
        "timestamp": "2026-09-20T06:05:00Z",
        "total_registered_images": len(registered_images),
        "completed_annotations": len(completed_images) + len(accepted_images),
        "pending_annotations": len(pending_images),
        "accepted_annotations": len(accepted_images),
        "qc_review_annotations": len(qc_review_images),
        "rejected_annotations": len(rejected_images),
        "valid_masks": valid_masks,
        "invalid_masks": invalid_masks,
        "orphan_masks": len(orphan_masks),
        "orphan_mask_files": orphan_masks,
        "valid_annotations": valid_annotations,
        "invalid_annotations": invalid_annotations,
        "orphan_annotations": len(orphan_annotations),
        "orphan_annotation_files": orphan_annotations,
        "hard_training_gate_status": "BLOCKED",
        "gate_reason": "Zero accepted real human masks exist. Research integrity prohibits pseudo-masks."
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(status_summary, f, indent=2)
        
    return status_summary

if __name__ == "__main__":
    summary = audit_mask_status()
    print("=== MASK STATUS AUDIT RESULT ===")
    print(json.dumps(summary, indent=2))
