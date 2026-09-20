"""
Comprehensive Roboflow YOLOv8 Dataset Inspection & Audit Engine.
Executes:
- Task 4: Structure and annotation verification
- Task 5: Class distribution and bounding box counts
- Task 6: Visual sample overlay generation with bounding boxes
- Task 7: Image quality and blur/exposure statistics
- Task 8: Cross-split leakage and duplicate analysis
"""

import os
import yaml
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from PIL import Image
import cv2
import numpy as np

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(filepath: Path, hash_size: int = 8) -> int:
    try:
        with Image.open(filepath) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            diff = []
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
            decimal_value = 0
            for idx, bit in enumerate(diff):
                if bit:
                    decimal_value += 1 << idx
            return decimal_value
    except Exception:
        return 0

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

class RoboflowDatasetAnalyzer:
    def __init__(self, dataset_dir: Path, output_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_path = self.dataset_dir / "data.yaml"
        self.class_names: List[str] = []
        self.num_classes: int = 0
        self.splits = ["train", "valid", "test"]

    def load_yaml_config(self) -> Dict[str, Any]:
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_dir}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.class_names = cfg.get("names", [])
        self.num_classes = cfg.get("nc", len(self.class_names))
        print(f"[+] Loaded data.yaml: {self.num_classes} classes -> {self.class_names}")
        return cfg

    def audit_structure_and_annotations(self) -> Dict[str, Any]:
        """Task 4: Comprehensive structure and bounding box geometry verification."""
        print("[*] Task 4: Auditing dataset structure and YOLO label integrity...")
        summary = {
            "dataset_path": str(self.dataset_dir),
            "data_yaml_present": self.yaml_path.exists(),
            "declared_classes": self.class_names,
            "declared_num_classes": self.num_classes,
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "images_without_labels": 0,
            "labels_without_images": 0,
            "empty_labels": 0,
            "malformed_annotations": 0,
            "out_of_range_class_ids": 0,
            "unnormalized_coords": 0,
            "total_bounding_boxes": 0,
            "bbox_aspect_ratios": [],
            "bbox_areas": []
        }

        all_seen_images = {}

        for split in self.splits:
            img_dir = self.dataset_dir / split / "images"
            lbl_dir = self.dataset_dir / split / "labels"
            
            img_files = sorted([f for f in img_dir.glob("*") if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS]) if img_dir.exists() else []
            lbl_files = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []

            split_info = {
                "image_count": len(img_files),
                "label_count": len(lbl_files),
                "missing_labels": 0,
                "empty_label_files": 0,
                "bbox_count": 0
            }

            img_stem_map = {f.stem: f for f in img_files}
            lbl_stem_map = {f.stem: f for f in lbl_files}

            summary["total_images"] += len(img_files)
            summary["total_labels"] += len(lbl_files)

            # Check images without labels
            for stem in img_stem_map:
                if stem not in lbl_stem_map:
                    summary["images_without_labels"] += 1
                    split_info["missing_labels"] += 1

            # Check labels without images
            for stem in lbl_stem_map:
                if stem not in img_stem_map:
                    summary["labels_without_images"] += 1

            # Inspect label files
            for lbl_path in lbl_files:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]

                if not lines:
                    summary["empty_labels"] += 1
                    split_info["empty_label_files"] += 1
                    continue

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        summary["malformed_annotations"] += 1
                        continue
                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                    except ValueError:
                        summary["malformed_annotations"] += 1
                        continue

                    if cls_id < 0 or cls_id >= self.num_classes:
                        summary["out_of_range_class_ids"] += 1

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        summary["unnormalized_coords"] += 1
                    else:
                        summary["total_bounding_boxes"] += 1
                        split_info["bbox_count"] += 1
                        summary["bbox_areas"].append(w * h)
                        if h > 0:
                            summary["bbox_aspect_ratios"].append(w / h)

            summary["splits"][split] = split_info

        # Summarize bbox geometry
        areas = summary.pop("bbox_areas")
        ratios = summary.pop("bbox_aspect_ratios")
        summary["bbox_geometry_metrics"] = {
            "mean_area_normalized": float(np.mean(areas)) if areas else 0.0,
            "median_area_normalized": float(np.median(areas)) if areas else 0.0,
            "min_area_normalized": float(np.min(areas)) if areas else 0.0,
            "max_area_normalized": float(np.max(areas)) if areas else 0.0,
            "mean_aspect_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "mean_bboxes_per_image": float(summary["total_bounding_boxes"] / max(1, summary["total_images"]))
        }

        # Save outputs/dataset_analysis/roboflow_annotation_summary.json
        out_json = self.output_dir / "roboflow_annotation_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Task 4 Complete: Saved annotation summary to {out_json}")
        return summary

    def analyze_classes(self) -> List[Dict[str, Any]]:
        """Task 5: Report image count, bbox count, and percentages per class."""
        print("[*] Task 5: Analyzing class distribution and annotation volumes...")
        class_img_counts = defaultdict(set)
        class_bbox_counts = defaultdict(int)

        for split in self.splits:
            lbl_dir = self.dataset_dir / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl_p in lbl_dir.glob("*.txt"):
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cid = int(parts[0])
                                class_bbox_counts[cid] += 1
                                class_img_counts[cid].add(lbl_p.stem)
                            except ValueError:
                                pass

        total_bboxes = max(1, sum(class_bbox_counts.values()))
        records = []
        for cid, cname in enumerate(self.class_names):
            img_cnt = len(class_img_counts[cid])
            box_cnt = class_bbox_counts[cid]
            pct = (box_cnt / total_bboxes) * 100
            records.append({
                "class_id": cid,
                "class_name": cname,
                "image_count": img_cnt,
                "annotation_count": box_cnt,
                "percentage": f"{pct:.2f}%"
            })

        # Export outputs/dataset_analysis/roboflow_class_distribution.csv
        csv_path = self.output_dir / "roboflow_class_distribution.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "image_count", "annotation_count", "percentage"])
            writer.writeheader()
            writer.writerows(records)
        print(f"[+] Task 5 Complete: Saved class distribution to {csv_path}")
        return records

    def validate_visual_annotations(self, samples_per_class: int = 3) -> Dict[str, Any]:
        """Task 6: Generate visual bounding box overlays per class and check quality."""
        print("[*] Task 6: Generating visual annotation verification overlays...")
        samples_dir = self.output_dir / "roboflow_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        class_samples = defaultdict(list)
        for split in self.splits:
            img_dir = self.dataset_dir / split / "images"
            lbl_dir = self.dataset_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img_p in img_dir.glob("*.*"):
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                if lbl_p.exists():
                    with open(lbl_p, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    for l in lines:
                        parts = l.split()
                        if len(parts) == 5:
                            cid = int(parts[0])
                            if len(class_samples[cid]) < samples_per_class:
                                class_samples[cid].append((img_p, lbl_p, split))

        # Color palette for classes (BGR)
        palette = [
            (0, 165, 255),  # 0: orange (bud root dropping)
            (0, 0, 255),    # 1: red (bud rot)
            (128, 128, 128),# 2: gray (gray leaf spot)
            (255, 0, 0),    # 3: blue (leaf rot)
            (0, 0, 139)     # 4: dark red (stembleeding)
        ]

        saved_files = []
        for cid, sample_list in class_samples.items():
            cname = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
            clean_cname = cname.replace(" ", "_")
            for idx, (img_p, lbl_p, split) in enumerate(sample_list):
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                h, w, _ = img.shape
                with open(lbl_p, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]

                for l in lines:
                    parts = l.split()
                    box_cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    color = palette[box_cid % len(palette)]
                    box_cname = self.class_names[box_cid] if box_cid < len(self.class_names) else str(box_cid)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Label banner
                    text = f"{box_cname}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(img, text, (x1 + 5, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                out_name = f"sample_{clean_cname}_{split}_{idx + 1}.png"
                out_path = samples_dir / out_name
                cv2.imwrite(str(out_path), img)
                saved_files.append(out_path)

        print(f"[+] Task 6 Complete: Saved {len(saved_files)} visual overlay samples to {samples_dir}")
        return {"samples_saved": len(saved_files), "directory": str(samples_dir)}

    def analyze_image_quality(self) -> List[Dict[str, Any]]:
        """Task 7: Compute dimensions, blurriness (Laplacian variance), and exposure."""
        print("[*] Task 7: Analyzing image quality metrics (blur, exposure, resolution)...")
        records = []
        for split in self.splits:
            img_dir = self.dataset_dir / split / "images"
            if not img_dir.exists():
                continue
            for img_p in sorted(list(img_dir.glob("*.*"))):
                try:
                    with Image.open(img_p) as im:
                        w, h = im.size
                        fmt = im.format or img_p.suffix.upper().replace(".", "")
                    
                    cv_img = cv2.imread(str(img_p))
                    if cv_img is not None:
                        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        mean_brightness = float(np.mean(gray))
                        is_blurry = laplacian_var < 50.0
                        is_dark = mean_brightness < 40.0
                        is_overexposed = mean_brightness > 220.0
                    else:
                        laplacian_var, mean_brightness = 0.0, 0.0
                        is_blurry, is_dark, is_overexposed = True, False, False

                    records.append({
                        "split": split,
                        "file_name": img_p.name,
                        "width": w,
                        "height": h,
                        "aspect_ratio": f"{(w / h):.3f}" if h > 0 else "0",
                        "format": fmt,
                        "laplacian_variance": f"{laplacian_var:.1f}",
                        "is_blurry": is_blurry,
                        "mean_brightness": f"{mean_brightness:.1f}",
                        "exposure_status": "Dark" if is_dark else ("Overexposed" if is_overexposed else "Normal")
                    })
                except Exception as e:
                    records.append({
                        "split": split,
                        "file_name": img_p.name,
                        "width": 0,
                        "height": 0,
                        "aspect_ratio": "0",
                        "format": "ERROR",
                        "laplacian_variance": "0.0",
                        "is_blurry": True,
                        "mean_brightness": "0.0",
                        "exposure_status": f"Corrupt ({e})"
                    })

        # Save outputs/dataset_analysis/roboflow_image_statistics.csv
        csv_path = self.output_dir / "roboflow_image_statistics.csv"
        if records:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
        print(f"[+] Task 7 Complete: Saved image statistics for {len(records)} images to {csv_path}")
        return records

    def detect_cross_split_leakage(self, hamming_threshold: int = 4) -> Dict[str, Any]:
        """Task 8: Detect exact duplicate and perceptual near-duplicate leakage across splits."""
        print("[*] Task 8: Scanning for cross-split data leakage (train vs val, train vs test, val vs test)...")
        split_images = {}
        split_hashes = defaultdict(list)
        split_dhashes = defaultdict(list)

        for split in self.splits:
            img_dir = self.dataset_dir / split / "images"
            if not img_dir.exists():
                continue
            for img_p in img_dir.glob("*.*"):
                md5_val = compute_md5(img_p)
                dh_val = compute_dhash(img_p)
                split_hashes[split].append((img_p.name, md5_val))
                split_dhashes[split].append((img_p.name, dh_val))

        leakage_report = {
            "splits_evaluated": list(split_hashes.keys()),
            "exact_cross_split_duplicates": [],
            "near_duplicate_cross_split_pairs": [],
            "summary_by_pair": {}
        }

        split_pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]

        for s1, s2 in split_pairs:
            pair_key = f"{s1}_vs_{s2}"
            exact_dups = []
            near_dups = []

            # 1. Exact MD5 matches
            md5_s1 = {md5: name for name, md5 in split_hashes[s1]}
            for name2, md5_2 in split_hashes[s2]:
                if md5_2 in md5_s1:
                    name1 = md5_s1[md5_2]
                    exact_dups.append({
                        f"{s1}_file": name1,
                        f"{s2}_file": name2,
                        "md5": md5_2
                    })

            # 2. Perceptual dHash matches (burst camera leakage)
            for name1, dh1 in split_dhashes[s1]:
                if dh1 == 0:
                    continue
                for name2, dh2 in split_dhashes[s2]:
                    if dh2 == 0:
                        continue
                    dist = hamming_distance(dh1, dh2)
                    if dist <= hamming_threshold:
                        near_dups.append({
                            f"{s1}_file": name1,
                            f"{s2}_file": name2,
                            "hamming_distance": dist
                        })

            leakage_report["summary_by_pair"][pair_key] = {
                "exact_duplicates": len(exact_dups),
                "near_duplicates": len(near_dups)
            }
            leakage_report["exact_cross_split_duplicates"].extend(exact_dups)
            leakage_report["near_duplicate_cross_split_pairs"].extend(near_dups)

        # Save outputs/dataset_analysis/roboflow_leakage_report.json
        out_json = self.output_dir / "roboflow_leakage_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(leakage_report, f, indent=2)
        print(f"[+] Task 8 Complete: Saved leakage report to {out_json}")
        return leakage_report
