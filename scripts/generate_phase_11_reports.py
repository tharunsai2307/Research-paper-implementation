"""
Generate Phase 11 Dataset Completion and QC Reports.
Outputs:
outputs/phase_11/annotation/
├── annotation_status.json
├── annotation_completion.csv
├── class_distribution.csv
├── split_distribution.csv
└── qc_summary.json
"""

import sys
import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["bud root dropping", "bud rot", "gray leaf spot", "leaf rot", "stembleeding"]

def generate_phase_11_reports():
    out_dir = PROJECT_ROOT / "outputs/phase_11/annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = PROJECT_ROOT / "data/external/phase_11_segmentation/metadata/annotation_manifest.csv"
    qc_manifest_path = PROJECT_ROOT / "data/external/phase_11_segmentation/metadata/qc_manifest.csv"
    
    records = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
            
    total_reg = len(records)
    annotated_count = 0
    accepted_count = 0
    pending_count = 0
    rejected_count = 0
    qc_review_count = 0
    
    train_masks = 0
    val_masks = 0
    test_masks = 0
    
    class_stats = {c: {"images": 0, "accepted_masks": 0, "lesions": 0, "total_area": 0.0} for c in CLASSES}
    split_stats = {"train": {"images": 0, "accepted_masks": 0}, "val": {"images": 0, "accepted_masks": 0}, "test": {"images": 0, "accepted_masks": 0}}
    
    completion_rows = []
    
    for r in records:
        img_id = r["image_id"]
        cls_name = r["disease_class"]
        split = r["split"]
        status_a = r.get("annotator_A_status", "PENDING")
        status_b = r.get("annotator_B_status", "PENDING")
        qc_stat = r.get("qc_status", "PENDING")
        acc_stat = r.get("acceptance_status", "PENDING")
        
        if cls_name in class_stats:
            class_stats[cls_name]["images"] += 1
        if split in split_stats:
            split_stats[split]["images"] += 1
            
        if acc_stat == "ACCEPTED":
            accepted_count += 1
            if split == "train": train_masks += 1
            elif split == "val": val_masks += 1
            elif split == "test": test_masks += 1
            if cls_name in class_stats:
                class_stats[cls_name]["accepted_masks"] += 1
        elif acc_stat == "REJECTED":
            rejected_count += 1
        elif qc_stat == "QC_REVIEW_REQUIRED":
            qc_review_count += 1
            
        if status_a == "COMPLETED" or status_b == "COMPLETED":
            annotated_count += 1
        else:
            pending_count += 1
            
        completion_rows.append({
            "image_id": img_id,
            "disease_class": cls_name,
            "split": split,
            "annotator_A_status": status_a,
            "annotator_B_status": status_b,
            "qc_status": qc_stat,
            "acceptance_status": acc_stat
        })
        
    # 1. annotation_status.json
    status_summary = {
        "timestamp": "2026-09-20T06:15:00Z",
        "registered_images": total_reg,
        "annotated_images": annotated_count,
        "pending_images": pending_count,
        "accepted_masks": accepted_count,
        "rejected_masks": rejected_count,
        "qc_review_required": qc_review_count,
        "training_masks": train_masks,
        "validation_masks": val_masks,
        "test_masks": test_masks,
        "double_annotated_images": 0,
        "mean_inter_annotator_iou": "N/A",
        "median_inter_annotator_iou": "N/A",
        "min_iou": "N/A",
        "max_iou": "N/A",
        "hard_training_gate_status": "BLOCKED",
        "gate_reason": "Zero accepted real human masks exist. Research integrity strictly prohibits pseudo-masks."
    }
    with open(out_dir / "annotation_status.json", "w", encoding="utf-8") as f:
        json.dump(status_summary, f, indent=2)
        
    # 2. annotation_completion.csv
    with open(out_dir / "annotation_completion.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "disease_class", "split", "annotator_A_status", "annotator_B_status", "qc_status", "acceptance_status"])
        writer.writeheader()
        writer.writerows(completion_rows)
        
    # 3. class_distribution.csv
    class_rows = []
    for c in CLASSES:
        class_rows.append({
            "disease_class": c,
            "registered_images": class_stats[c]["images"],
            "accepted_masks": class_stats[c]["accepted_masks"],
            "lesion_count": class_stats[c]["lesions"],
            "total_mask_area": f"{class_stats[c]['total_area']:.5f}",
            "mean_mask_area": "N/A" if class_stats[c]["accepted_masks"] == 0 else f"{class_stats[c]['total_area'] / max(1, class_stats[c]['accepted_masks']):.5f}",
            "median_mask_area": "N/A"
        })
    with open(out_dir / "class_distribution.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["disease_class", "registered_images", "accepted_masks", "lesion_count", "total_mask_area", "mean_mask_area", "median_mask_area"])
        writer.writeheader()
        writer.writerows(class_rows)
        
    # 4. split_distribution.csv
    split_rows = []
    for s in ["train", "val", "test"]:
        split_rows.append({
            "split": s,
            "registered_images": split_stats[s]["images"],
            "accepted_masks": split_stats[s]["accepted_masks"],
            "completion_percent": f"{(split_stats[s]['accepted_masks'] / max(1, split_stats[s]['images'])) * 100.0:.1f}%"
        })
    with open(out_dir / "split_distribution.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "registered_images", "accepted_masks", "completion_percent"])
        writer.writeheader()
        writer.writerows(split_rows)
        
    # 5. qc_summary.json
    qc_summary = {
        "timestamp": "2026-09-20T06:15:00Z",
        "qc_protocol": "Double-Annotator Independent Validation (IoU >= 0.75)",
        "double_annotated_subset_target": 20,
        "double_annotated_completed": 0,
        "passed_qc": 0,
        "failed_qc": 0,
        "qc_review_required": 0,
        "inter_annotator_metrics": {
            "mean_iou": "N/A",
            "median_iou": "N/A",
            "min_iou": "N/A",
            "max_iou": "N/A"
        },
        "resolution_workflow": "Annotator A + Annotator B -> Disagreement -> Consensus Adjudication -> Accepted Ground Truth"
    }
    with open(out_dir / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2)
        
    print("Phase 11 completion reports successfully generated in outputs/phase_11/annotation/")

if __name__ == "__main__":
    generate_phase_11_reports()
