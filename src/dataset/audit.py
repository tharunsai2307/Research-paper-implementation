"""
Automated dataset audit engine for Coconut Tree Disease Detection.
Inspects image readability, dimensions, corruptions, duplicate filenames,
annotation integrity, and bounding box geometry.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from PIL import Image
import cv2
import numpy as np

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class DatasetAuditor:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.summary: Dict[str, Any] = {}
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.corrupted_files: List[str] = []
        self.unreadable_files: List[str] = []
        self.duplicate_filenames: Dict[str, List[str]] = defaultdict(list)
        self.dimension_stats: Dict[str, Any] = {}
        self.annotation_stats: Dict[str, Any] = {}
        self.image_records: List[Dict[str, Any]] = []

    def audit(self) -> Dict[str, Any]:
        """Runs full integrity and statistical audit on the dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        print(f"[*] Starting audit on: {self.dataset_path}")
        all_files = list(self.dataset_path.rglob("*"))
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS]
        
        filename_map = defaultdict(list)
        for img_p in image_files:
            filename_map[img_p.name].append(str(img_p))
        self.duplicate_filenames = {k: v for k, v in filename_map.items() if len(v) > 1}

        widths, heights, channels_list = [], [], []
        image_formats = defaultdict(int)
        
        # Check annotations if present (.txt for YOLO, .xml for Pascal VOC)
        total_annotations = 0
        empty_annotations = 0
        malformed_annotations = 0
        bbox_count = 0
        bbox_areas = []
        bbox_aspect_ratios = []
        class_bbox_counts = defaultdict(int)

        for img_path in image_files:
            rel_path = str(img_path.relative_to(self.dataset_path))
            parent_name = img_path.parent.name
            # If arranged in class folders
            cls_name = parent_name if parent_name != self.dataset_path.name else "unspecified"
            self.class_counts[cls_name] += 1

            # Check readability via PIL
            is_valid_pil = False
            img_format = "UNKNOWN"
            w, h, c = 0, 0, 3
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
                    img_format = im.format or img_path.suffix.upper().replace(".", "")
                    mode = im.mode
                    c = len(mode) if mode in ("RGB", "RGBA") else 1
                is_valid_pil = True
            except Exception as e:
                self.corrupted_files.append(f"{rel_path}: PIL verification failed ({e})")

            # Check readability via OpenCV
            is_valid_cv2 = False
            if is_valid_pil:
                try:
                    cv_img = cv2.imread(str(img_path))
                    if cv_img is None:
                        self.unreadable_files.append(f"{rel_path}: cv2.imread returned None")
                    else:
                        is_valid_cv2 = True
                except Exception as e:
                    self.unreadable_files.append(f"{rel_path}: cv2 read error ({e})")

            if is_valid_pil and is_valid_cv2:
                widths.append(w)
                heights.append(h)
                channels_list.append(c)
                image_formats[img_format] += 1

            # Check corresponding YOLO label file
            txt_path = img_path.with_suffix(".txt")
            has_annot = txt_path.exists()
            if has_annot:
                total_annotations += 1
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        empty_annotations += 1
                    else:
                        for line in lines:
                            parts = line.split()
                            if len(parts) != 5:
                                malformed_annotations += 1
                                continue
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                                malformed_annotations += 1
                            else:
                                bbox_count += 1
                                bbox_areas.append(bw * bh)
                                if bh > 0:
                                    bbox_aspect_ratios.append(bw / bh)
                                class_bbox_counts[str(cls_id)] += 1
                except Exception as e:
                    malformed_annotations += 1

            self.image_records.append({
                "file": rel_path,
                "class": cls_name,
                "width": w,
                "height": h,
                "channels": c,
                "format": img_format,
                "corrupt": not (is_valid_pil and is_valid_cv2)
            })

        # Summarize dimensions
        dim_summary = {}
        if widths:
            dim_summary = {
                "min_resolution": f"{min(widths)}x{min(heights)}",
                "max_resolution": f"{max(widths)}x{max(heights)}",
                "mean_width": float(np.mean(widths)),
                "mean_height": float(np.mean(heights)),
                "median_width": float(np.median(widths)),
                "median_height": float(np.median(heights)),
                "unique_resolutions": len(set(zip(widths, heights)))
            }

        # Annotation summary
        annot_summary = {
            "annotations_found": total_annotations > 0,
            "total_annotation_files": total_annotations,
            "empty_annotations": empty_annotations,
            "malformed_annotations": malformed_annotations,
            "total_bounding_boxes": bbox_count,
            "mean_bboxes_per_image": float(bbox_count / max(1, len(image_files))) if total_annotations else 0.0,
            "mean_normalized_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else 0.0,
            "class_bbox_distribution": dict(class_bbox_counts)
        }

        self.summary = {
            "dataset_root": str(self.dataset_path),
            "total_images": len(image_files),
            "valid_images": len(widths),
            "corrupted_images": len(self.corrupted_files),
            "unreadable_images": len(self.unreadable_files),
            "duplicate_filename_groups": len(self.duplicate_filenames),
            "class_distribution": dict(self.class_counts),
            "image_formats": dict(image_formats),
            "dimension_statistics": dim_summary,
            "annotation_statistics": annot_summary,
            "corrupted_details": self.corrupted_files,
            "unreadable_details": self.unreadable_files,
            "duplicate_details": {k: v for k, v in list(self.duplicate_filenames.items())[:10]}
        }

        return self.summary

    def export_reports(self, output_dir: str | Path, markdown_path: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. JSON summary
        json_path = output_dir / "dataset_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
        print(f"[+] Saved summary to {json_path}")

        # 2. Class distribution CSV
        csv_path = output_dir / "class_distribution.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "image_count", "percentage"])
            total = max(1, self.summary.get("total_images", 1))
            for cls_name, count in self.summary.get("class_distribution", {}).items():
                pct = (count / total) * 100
                writer.writerow([cls_name, count, f"{pct:.2f}%"])
        print(f"[+] Saved class distribution to {csv_path}")

        # 3. Markdown Report
        md_content = self._generate_markdown_report()
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[+] Saved audit report to {markdown_path}")

    def _generate_markdown_report(self) -> str:
        s = self.summary
        dim = s.get("dimension_statistics", {})
        annot = s.get("annotation_statistics", {})
        
        md = [
            "# Automated Dataset Audit Report: Coconut Tree Disease",
            "",
            f"**Audit Directory**: `{s.get('dataset_root')}`  ",
            f"**Total Images Audited**: {s.get('total_images')}  ",
            f"**Valid Images**: {s.get('valid_images')}  ",
            f"**Corrupted / Unreadable**: {s.get('corrupted_images')} corrupted, {s.get('unreadable_images')} unreadable  ",
            f"**Duplicate Filename Groups**: {s.get('duplicate_filename_groups')}  ",
            "",
            "## 1. Class Distribution",
            "",
            "| Class Name | Image Count | Proportion |",
            "| :--- | :---: | :---: |"
        ]
        
        total = max(1, s.get("total_images", 1))
        for cls_name, count in s.get("class_distribution", {}).items():
            pct = (count / total) * 100
            md.append(f"| **{cls_name}** | {count} | {pct:.2f}% |")

        md.extend([
            "",
            "## 2. Image Formats and Dimensions",
            "",
            f"- **Supported Formats**: {s.get('image_formats')}",
            f"- **Minimum Resolution**: {dim.get('min_resolution', 'N/A')}",
            f"- **Maximum Resolution**: {dim.get('max_resolution', 'N/A')}",
            f"- **Mean Resolution**: {dim.get('mean_width', 0):.1f} x {dim.get('mean_height', 0):.1f}",
            f"- **Unique Dimensions**: {dim.get('unique_resolutions', 0)} distinct aspect ratios / sizes",
            "",
            "## 3. Annotation & Object Detection Integrity",
            "",
            f"- **Annotations Detected**: {'YES' if annot.get('annotations_found') else 'NO'}",
            f"- **Total Annotation Files**: {annot.get('total_annotation_files', 0)}",
            f"- **Empty Annotation Files**: {annot.get('empty_annotations', 0)}",
            f"- **Malformed Coordinates**: {annot.get('malformed_annotations', 0)}",
            f"- **Total Bounding Boxes**: {annot.get('total_bounding_boxes', 0)}",
            f"- **Mean Bounding Boxes per Image**: {annot.get('mean_bboxes_per_image', 0):.2f}",
            f"- **Mean Normalized Bounding Box Area**: {annot.get('mean_normalized_bbox_area', 0):.4f}",
            "",
            "## 4. Quality Anomalies & Recommendations",
            ""
        ])

        if s.get("corrupted_images") == 0 and s.get("unreadable_images") == 0:
            md.append("- **Data Integrity**: Clean. 100% of examined image files were successfully parsed by PIL and OpenCV.")
        else:
            md.append(f"- **Data Integrity Warning**: Found {s.get('corrupted_images')} corrupted and {s.get('unreadable_images')} unreadable files.")

        if annot.get("annotations_found"):
            md.append("- **Object Detection Suitability**: Verified. Ground truth bounding boxes exist and are formatted for YOLOv8.")
        else:
            md.append("- **Annotation Status**: **Classification-Only Source**. Bounding box ground truth does NOT exist in the raw dataset hierarchy. Manual or AI-assisted bounding box annotation is required before YOLOv8 object detection training can proceed.")

        return "\n".join(md)
