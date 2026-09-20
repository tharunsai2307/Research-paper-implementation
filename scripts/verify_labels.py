"""
Label integrity verification engine for YOLOv8 datasets.
Validates format: class_id x_center y_center width height
Checks coordinates in [0, 1], non-zero positive dimensions, NaN/inf detection,
and bounds preservation. Generates JSON and CSV audit reports.
"""

import math
import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any

def verify_label_integrity(dataset_dir: Path, output_dir: Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_file = dataset_dir / "data.yaml"
    valid_class_ids = set()
    class_names = []
    if yaml_file.exists():
        with open(yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            class_names = cfg.get("names", [])
            valid_class_ids = set(range(len(class_names)))
    
    report = {
        "dataset_path": str(dataset_dir),
        "total_label_files": 0,
        "valid_label_files": 0,
        "invalid_label_files": 0,
        "total_bounding_boxes": 0,
        "valid_bounding_boxes": 0,
        "invalid_bounding_boxes": 0,
        "invalid_coordinate_count": 0,
        "unknown_class_count": 0,
        "nan_inf_count": 0,
        "empty_lines_count": 0,
        "missing_label_count": 0,
        "orphan_label_count": 0,
        "splits": {}
    }

    csv_rows = []
    splits = ["train", "valid", "test", "val"]

    for split in splits:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists() and not lbl_dir.exists():
            continue

        img_files = {f.stem: f for f in img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]} if img_dir.exists() else {}
        lbl_files = {f.stem: f for f in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}

        split_missing = len(set(img_files.keys()) - set(lbl_files.keys()))
        split_orphan = len(set(lbl_files.keys()) - set(img_files.keys()))
        report["missing_label_count"] += split_missing
        report["orphan_label_count"] += split_orphan

        split_valid_files = 0
        split_invalid_files = 0
        split_boxes = 0

        for stem, lbl_p in lbl_files.items():
            report["total_label_files"] += 1
            has_error = False
            file_box_count = 0

            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, raw_line in enumerate(lines, 1):
                line = raw_line.strip()
                if not line:
                    report["empty_lines_count"] += 1
                    continue

                parts = line.split()
                if len(parts) != 5:
                    has_error = True
                    report["invalid_bounding_boxes"] += 1
                    csv_rows.append({
                        "split": split,
                        "file": lbl_p.name,
                        "line_no": line_no,
                        "error_type": "malformed_token_count",
                        "raw_line": line
                    })
                    continue

                # Parse class_id and coords
                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    has_error = True
                    report["invalid_bounding_boxes"] += 1
                    csv_rows.append({
                        "split": split,
                        "file": lbl_p.name,
                        "line_no": line_no,
                        "error_type": "non_numeric_values",
                        "raw_line": line
                    })
                    continue

                # Check for NaN / Inf
                if any(math.isnan(v) or math.isinf(v) for v in [xc, yc, w, h]):
                    has_error = True
                    report["nan_inf_count"] += 1
                    report["invalid_bounding_boxes"] += 1
                    csv_rows.append({
                        "split": split,
                        "file": lbl_p.name,
                        "line_no": line_no,
                        "error_type": "nan_or_inf",
                        "raw_line": line
                    })
                    continue

                # Check class id
                if valid_class_ids and cls_id not in valid_class_ids:
                    has_error = True
                    report["unknown_class_count"] += 1
                    report["invalid_bounding_boxes"] += 1
                    csv_rows.append({
                        "split": split,
                        "file": lbl_p.name,
                        "line_no": line_no,
                        "error_type": f"unknown_class_id_{cls_id}",
                        "raw_line": line
                    })
                    continue

                # Check normalized coordinate ranges
                # xc, yc in [0, 1], w, h in (0, 1]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    has_error = True
                    report["invalid_coordinate_count"] += 1
                    report["invalid_bounding_boxes"] += 1
                    csv_rows.append({
                        "split": split,
                        "file": lbl_p.name,
                        "line_no": line_no,
                        "error_type": "out_of_range_coords",
                        "raw_line": line
                    })
                    continue

                # Check bounds inside [0, 1]
                x_min = xc - w / 2.0
                y_min = yc - h / 2.0
                x_max = xc + w / 2.0
                y_max = yc + h / 2.0

                if x_min < -0.05 or y_min < -0.05 or x_max > 1.05 or y_max > 1.05:
                    has_error = True
                    report["invalid_coordinate_count"] += 1
                    report["invalid_bounding_boxes"] += 1
                    csv_rows.append({
                        "split": split,
                        "file": lbl_p.name,
                        "line_no": line_no,
                        "error_type": "box_exceeds_image_boundaries",
                        "raw_line": line
                    })
                    continue

                # Valid box
                report["valid_bounding_boxes"] += 1
                file_box_count += 1
                report["total_bounding_boxes"] += 1

            if has_error:
                split_invalid_files += 1
                report["invalid_label_files"] += 1
            else:
                split_valid_files += 1
                report["valid_label_files"] += 1

            split_boxes += file_box_count

        report["splits"][split] = {
            "label_files": len(lbl_files),
            "valid_files": split_valid_files,
            "invalid_files": split_invalid_files,
            "valid_boxes": split_boxes,
            "missing_labels": split_missing,
            "orphan_labels": split_orphan
        }

    # Save outputs/dataset_analysis/label_integrity_report.json
    json_path = output_dir / "label_integrity_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save outputs/dataset_analysis/label_integrity_report.csv
    csv_path = output_dir / "label_integrity_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "file", "line_no", "error_type", "raw_line"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if csv_rows:
            writer.writerows(csv_rows)
        else:
            writer.writerow({
                "split": "ALL",
                "file": "NONE",
                "line_no": 0,
                "error_type": "NO_ERRORS_FOUND",
                "raw_line": "All labels verified 100% valid"
            })

    print(f"[+] Verified {report['total_label_files']} label files: {report['valid_label_files']} valid, {report['invalid_label_files']} invalid.")
    print(f"[+] Total BBoxes: {report['total_bounding_boxes']}, Valid: {report['valid_bounding_boxes']}, Invalid: {report['invalid_bounding_boxes']}")
    print(f"[+] Exported JSON to: {json_path}")
    print(f"[+] Exported CSV to: {csv_path}")
    return report

if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    ds_dir = base_dir / "data" / "raw" / "roboflow_coconut_detection"
    out_dir = base_dir / "outputs" / "dataset_analysis"
    verify_label_integrity(ds_dir, out_dir)
