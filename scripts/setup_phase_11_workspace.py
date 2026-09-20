"""
Setup and Initialize Phase 11 Segmentation Workspace Metadata.
Computes SHA-256 for all 100 registered images from Phase 9/10, preserves splits,
and generates the initial manifests:
1. data/external/phase_11_segmentation/metadata/annotation_manifest.csv
2. data/external/phase_11_segmentation/metadata/qc_manifest.csv
3. data/external/phase_11_segmentation/metadata/annotation_events.csv
"""

import sys
import os
import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_MAP = {
    0: "bud root dropping",
    1: "bud rot",
    2: "gray leaf spot",
    3: "leaf rot",
    4: "stembleeding"
}
CLASS_TO_ID = {v: k for k, v in CLASS_MAP.items()}

def setup_phase_11():
    print("Initializing Phase 11 Segmentation Workspace...")
    p9_dir = PROJECT_ROOT / "data/external/phase_9_segmentation"
    p11_dir = PROJECT_ROOT / "data/external/phase_11_segmentation"
    p11_meta = p11_dir / "metadata"
    p11_meta.mkdir(parents=True, exist_ok=True)
    
    p9_manifest = p9_dir / "metadata/annotation_manifest.csv"
    if not p9_manifest.exists():
        raise FileNotFoundError(f"Phase 9 manifest missing at {p9_manifest}")
        
    with open(p9_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        p9_rows = list(reader)
        
    p11_manifest_rows = []
    events = []
    now_iso = datetime.now(timezone.utc).isoformat()
    
    for row in p9_rows:
        img_id = row["image_id"]
        img_path = p9_dir / "images" / img_id
        if not img_path.exists():
            raise FileNotFoundError(f"Image {img_id} not found at {img_path}")
            
        sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
        cls_name = row["disease_class"]
        cls_id = CLASS_TO_ID.get(cls_name, -1)
        split = row["split"]
        
        p11_manifest_rows.append({
            "image_id": img_id,
            "source_path": f"data/external/phase_9_segmentation/images/{img_id}",
            "disease_class": cls_name,
            "class_id": cls_id,
            "split": split,
            "image_sha256": sha256,
            "annotator_A_status": "PENDING",
            "annotator_A_path": "NONE",
            "annotator_B_status": "PENDING",
            "annotator_B_path": "NONE",
            "qc_status": "PENDING",
            "final_mask_path": "NONE",
            "acceptance_status": "PENDING",
            "review_notes": "Awaiting certified human manual polygon delineation"
        })
        
    # Write annotation_manifest.csv
    manifest_file = p11_meta / "annotation_manifest.csv"
    fieldnames = [
        "image_id", "source_path", "disease_class", "class_id", "split", "image_sha256",
        "annotator_A_status", "annotator_A_path", "annotator_B_status", "annotator_B_path",
        "qc_status", "final_mask_path", "acceptance_status", "review_notes"
    ]
    with open(manifest_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(p11_manifest_rows)
    print(f"  [CREATED] {manifest_file} with {len(p11_manifest_rows)} benchmark records.")
    
    # Write empty qc_manifest.csv
    qc_file = p11_meta / "qc_manifest.csv"
    qc_fieldnames = [
        "image_id", "class_id", "split", "annotator_A_id", "annotator_B_id",
        "iou", "dice", "qc_result", "resolution_status", "consensus_mask_path",
        "reviewer_id", "timestamp"
    ]
    with open(qc_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=qc_fieldnames)
        writer.writeheader()
    print(f"  [CREATED] {qc_file} (Awaiting double-annotation pairs).")
    
    # Write annotation_events.csv
    events_file = p11_meta / "annotation_events.csv"
    events_fieldnames = ["event_id", "timestamp", "image_id", "event_type", "annotator_id", "details"]
    initial_event = {
        "event_id": "EVT-PHASE11-INIT",
        "timestamp": now_iso,
        "image_id": "ALL_100_IMAGES",
        "event_type": "WORKSPACE_INITIALIZATION",
        "annotator_id": "SYSTEM_RESEARCH_ENGINE",
        "details": "Initialized Phase 11 human annotation workspace with 100 verified benchmark images."
    }
    with open(events_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=events_fieldnames)
        writer.writeheader()
        writer.writerow(initial_event)
    print(f"  [CREATED] {events_file} with initial setup event.")
    
    print("Phase 11 metadata initialization COMPLETE.")

if __name__ == "__main__":
    setup_phase_11()
