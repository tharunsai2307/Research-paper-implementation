"""
Phase 4 Final Model Evaluation & Research Analysis Script.
Evaluates:
- MODEL A: outputs/training/EXP-001_baseline_yolov8n/weights/best.pt (imgsz=640)
- MODEL B: outputs/training/EXP-002_imgsz512/weights/best.pt (imgsz=512)
against the untouched test set (data/processed/coconut_detection_clean/data.yaml, split='test').

Generates:
- outputs/evaluation/phase_4_test_results.json
- outputs/evaluation/phase_4_test_results.csv
- outputs/evaluation/phase_4_per_class_results.csv
- outputs/evaluation/phase_4_error_registry.csv
- outputs/evaluation/detection_output_schema.json
- Prediction visual artifacts in outputs/evaluation/EXP-001_test_predictions/ and EXP-002_test_predictions/
- Evaluation research figures in outputs/figures/evaluation/
"""

import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "data.yaml"
TEST_IMG_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "test" / "images"
TEST_LBL_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "test" / "labels"

MODEL_A_PATH = BASE_DIR / "outputs" / "training" / "EXP-001_baseline_yolov8n" / "weights" / "best.pt"
MODEL_B_PATH = BASE_DIR / "outputs" / "training" / "EXP-002_imgsz512" / "weights" / "best.pt"

OUTPUT_EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
OUTPUT_FIG_DIR = BASE_DIR / "outputs" / "figures" / "evaluation"
RESEARCH_DIR = BASE_DIR / "research"

CLASS_NAMES = {
    0: "bud root dropping",
    1: "bud rot",
    2: "gray leaf spot",
    3: "leaf rot",
    4: "stembleeding"
}

CLASS_COLORS = {
    0: (0, 165, 255),   # Orange (BGR)
    1: (0, 0, 255),     # Red
    2: (255, 191, 0),   # Deep Sky Blue
    3: (128, 0, 128),   # Purple
    4: (0, 255, 255)    # Yellow
}

def load_ground_truth(image_path: Path):
    lbl_p = TEST_LBL_DIR / f"{image_path.stem}.txt"
    boxes = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    boxes.append({"class_id": cid, "class_name": CLASS_NAMES[cid], "xc": xc, "yc": yc, "w": w, "h": h})
    return boxes

def box_iou(b1, b2):
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
    boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def evaluate_model(model_path: Path, exp_id: str, imgsz: int, conf_thresh: float = 0.25, iou_thresh: float = 0.70):
    print("\n" + "=" * 75)
    print(f"[*] EVALUATING {exp_id} ON UNTOUCHED TEST SET")
    print(f"    Checkpoint: {model_path} | ImgSz: {imgsz} | Conf: {conf_thresh} | IoU: {iou_thresh}")
    print("=" * 75)

    model = YOLO(str(model_path))

    # 1. Run official Ultralytics test validation (default PR curve integration)
    val_res = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        device="cpu",
        verbose=True
    )

    box_metrics = val_res.box
    prec = float(box_metrics.mp)
    rec = float(box_metrics.mr)
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)
    f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Per-class metrics
    per_class_results = {}
    maps_per_class = box_metrics.maps
    for cid, cname in CLASS_NAMES.items():
        if cid < len(maps_per_class):
            ap50_val = round(float(box_metrics.all_ap[cid, 0]), 4) if hasattr(box_metrics, "all_ap") and box_metrics.all_ap.shape[0] > cid else 0.0
            per_class_results[cname] = {
                "ap50_95": round(float(maps_per_class[cid]), 4),
                "ap50": ap50_val,
                "precision": round(float(box_metrics.p[cid]), 4) if hasattr(box_metrics, "p") and len(box_metrics.p) > cid else "NOT_AVAILABLE",
                "recall": round(float(box_metrics.r[cid]), 4) if hasattr(box_metrics, "r") and len(box_metrics.r) > cid else "NOT_AVAILABLE"
            }
        else:
            per_class_results[cname] = {"ap50_95": 0.0, "ap50": 0.0, "precision": 0.0, "recall": 0.0}

    # 2. Individual Image Predictions & Error Analysis
    pred_dir = OUTPUT_EVAL_DIR / f"{exp_id}_test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    test_images = sorted(list(TEST_IMG_DIR.glob("*.*")))
    image_eval_records = []
    latencies = []

    for img_p in test_images:
        img_bgr = cv2.imread(str(img_p))
        h_img, w_img = img_bgr.shape[:2]
        gt_boxes = load_ground_truth(img_p)

        t0 = time.time()
        pred_res = model.predict(source=str(img_p), imgsz=imgsz, conf=conf_thresh, iou=iou_thresh, device="cpu", verbose=False)[0]
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

        boxes = pred_res.boxes
        preds = []
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            preds.append({
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, str(cid)),
                "confidence": round(conf, 4),
                "xyxy": [round(coord, 1) for coord in xyxy]
            })

        # Classification of error
        is_healthy = len(gt_boxes) == 0
        false_positive = False
        false_negative = False
        localization_error = False

        if is_healthy:
            if len(preds) > 0:
                false_positive = True
                error_type = "FALSE_POSITIVE_HEALTHY"
                error_notes = f"Model predicted {len(preds)} disease box(es) on healthy palm image."
            else:
                error_type = "TRUE_NEGATIVE_HEALTHY"
                error_notes = "Correct background rejection (zero disease boxes predicted)."
        else:
            if len(preds) == 0:
                false_negative = True
                error_type = "FALSE_NEGATIVE_MISSED"
                error_notes = f"Missed {len(gt_boxes)} ground truth disease box(es)."
            else:
                # Check matching
                gt_xyxy = []
                for b in gt_boxes:
                    x1 = (b["xc"] - b["w"] / 2.0) * w_img
                    y1 = (b["yc"] - b["h"] / 2.0) * h_img
                    x2 = (b["xc"] + b["w"] / 2.0) * w_img
                    y2 = (b["yc"] + b["h"] / 2.0) * h_img
                    gt_xyxy.append({"class_id": b["class_id"], "xyxy": [x1, y1, x2, y2]})

                matched_gt = set()
                for p_idx, p in enumerate(preds):
                    best_iou = 0.0
                    best_g_idx = -1
                    for g_idx, g in enumerate(gt_xyxy):
                        iou = box_iou(p["xyxy"], g["xyxy"])
                        if iou > best_iou:
                            best_iou = iou
                            best_g_idx = g_idx

                    if best_iou >= 0.5:
                        if p["class_id"] == gt_xyxy[best_g_idx]["class_id"]:
                            matched_gt.add(best_g_idx)
                        else:
                            error_type = "CLASS_CONFUSION"
                    elif best_iou > 0.1:
                        localization_error = True
                        error_type = "POOR_LOCALIZATION"

                if len(matched_gt) == len(gt_xyxy):
                    error_type = "TRUE_POSITIVE_EXACT"
                    error_notes = f"All {len(gt_xyxy)} ground truth boxes localized successfully."
                elif len(matched_gt) > 0:
                    error_type = "PARTIAL_DETECTION"
                    error_notes = f"Matched {len(matched_gt)}/{len(gt_xyxy)} ground truth boxes."
                else:
                    error_type = "MISSED_OR_CONFUSED"
                    error_notes = "Predictions did not match ground truth IoU threshold."

        record = {
            "image": img_p.name,
            "model": exp_id,
            "is_healthy_negative": is_healthy,
            "ground_truth_count": len(gt_boxes),
            "ground_truth_classes": [b["class_name"] for b in gt_boxes],
            "prediction_count": len(preds),
            "predicted_classes": [p["class_name"] for p in preds],
            "predicted_confidences": [p["confidence"] for p in preds],
            "error_type": error_type,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "localization_error": localization_error,
            "notes": error_notes
        }
        image_eval_records.append(record)

        # 3. Draw and save visual comparison artifact
        # Side-by-side: Left = Ground Truth, Right = Prediction
        canvas_gt = img_bgr.copy()
        canvas_pred = img_bgr.copy()

        # Draw GT
        for b in gt_boxes:
            x1 = int((b["xc"] - b["w"] / 2.0) * w_img)
            y1 = int((b["yc"] - b["h"] / 2.0) * h_img)
            x2 = int((b["xc"] + b["w"] / 2.0) * w_img)
            y2 = int((b["yc"] + b["h"] / 2.0) * h_img)
            col = (0, 255, 0) # Green for GT
            cv2.rectangle(canvas_gt, (x1, y1), (x2, y2), col, 2)
            cv2.putText(canvas_gt, f"GT: {b['class_name']}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        if is_healthy:
            cv2.putText(canvas_gt, "GT: Healthy Negative", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw Pred
        for p in preds:
            x1, y1, x2, y2 = [int(v) for v in p["xyxy"]]
            col = CLASS_COLORS.get(p["class_id"], (255, 255, 255))
            cv2.rectangle(canvas_pred, (x1, y1), (x2, y2), col, 2)
            label_text = f"{p['class_name']} {p['confidence']:.2f}"
            cv2.putText(canvas_pred, label_text, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        if len(preds) == 0:
            cv2.putText(canvas_pred, "No Detections", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Header titles
        cv2.putText(canvas_gt, "GROUND TRUTH", (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas_pred, f"PREDICTION ({exp_id})", (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        combined = np.hstack([canvas_gt, canvas_pred])
        save_path = pred_dir / f"eval_{img_p.name}"
        cv2.imwrite(str(save_path), combined)

    mean_latency = float(np.mean(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    # Negative control summary
    healthy_recs = [r for r in image_eval_records if r["is_healthy_negative"]]
    healthy_fps = [r for r in healthy_recs if r["false_positive"]]

    summary = {
        "experiment_id": exp_id,
        "model_checkpoint": str(model_path),
        "imgsz": imgsz,
        "conf_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "test_images_count": len(test_images),
        "overall_metrics": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "f1_score": round(f1, 4)
        },
        "per_class_results": per_class_results,
        "latency_benchmark": {
            "device": "CPU (AMD Ryzen 5 7520U)",
            "mean_total_latency_ms": round(mean_latency, 2),
            "estimated_fps": round(fps, 1),
            "test_images_timed": len(test_images)
        },
        "negative_control_findings": {
            "total_healthy_test_images": len(healthy_recs),
            "healthy_images_with_disease_prediction": len(healthy_fps),
            "healthy_images_without_prediction": len(healthy_recs) - len(healthy_fps),
            "false_positive_rate": round(len(healthy_fps) / len(healthy_recs), 3) if healthy_recs else 0.0,
            "observations": [r["notes"] for r in healthy_fps]
        },
        "image_eval_records": image_eval_records
    }

    print(f"\n[+] TEST EVALUATION COMPLETE FOR {exp_id}:")
    print(f"    mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map50_95:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    print(f"    CPU Latency: {mean_latency:.2f} ms ({fps:.1f} FPS)")
    print(f"    Healthy FP: {len(healthy_fps)} / {len(healthy_recs)} (FP Rate: {summary['negative_control_findings']['false_positive_rate']})")
    return summary

def main():
    print("=" * 75)
    print("[*] STARTING PHASE 4 FINAL MODEL EVALUATION")
    print("=" * 75)

    OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate MODEL A (EXP-001)
    eval_a = evaluate_model(
        model_path=MODEL_A_PATH,
        exp_id="EXP-001_baseline_yolov8n",
        imgsz=640,
        conf_thresh=0.25,
        iou_thresh=0.70
    )

    # Evaluate MODEL B (EXP-002)
    eval_b = evaluate_model(
        model_path=MODEL_B_PATH,
        exp_id="EXP-002_imgsz512",
        imgsz=512,
        conf_thresh=0.25,
        iou_thresh=0.70
    )

    # 1. Generate outputs/evaluation/phase_4_test_results.json
    final_json = {
        "meta": {
            "phase": "Phase 4 — Final Model Evaluation & Research Analysis",
            "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "dataset_version": "v1.0.0-sanitized",
            "test_split_path": str(TEST_IMG_DIR.parent),
            "test_images": 15,
            "test_diseased_images": 10,
            "test_healthy_images": 5,
            "test_annotations": 12,
            "device": "CPU (AMD Ryzen 5 7520U)"
        },
        "model_evaluations": {
            "EXP-001_baseline_yolov8n": eval_a,
            "EXP-002_imgsz512": eval_b
        }
    }
    with open(OUTPUT_EVAL_DIR / "phase_4_test_results.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)
    print(f"[+] Saved phase_4_test_results.json to: {OUTPUT_EVAL_DIR / 'phase_4_test_results.json'}")

    # 2. Generate outputs/evaluation/phase_4_test_results.csv
    test_csv_rows = []
    for r in [eval_a, eval_b]:
        m = r["overall_metrics"]
        lat = r["latency_benchmark"]
        neg = r["negative_control_findings"]
        test_csv_rows.append({
            "model": "YOLOv8n",
            "experiment_id": r["experiment_id"],
            "dataset_version": "v1.0.0-sanitized",
            "split": "test",
            "images": r["test_images_count"],
            "instances": 12,
            "precision": m["precision"],
            "recall": m["recall"],
            "map50": m["mAP50"],
            "map50_95": m["mAP50_95"],
            "f1": m["f1_score"],
            "latency_ms": lat["mean_total_latency_ms"],
            "fps": lat["estimated_fps"],
            "parameters": 3006623,
            "model_size_mb": 5.94 if "640" in r["experiment_id"] else 5.93,
            "confidence_threshold": r["conf_threshold"],
            "iou_threshold": r["iou_threshold"],
            "device": lat["device"],
            "healthy_negatives_evaluated": neg["total_healthy_test_images"],
            "healthy_false_positives": neg["healthy_images_with_disease_prediction"],
            "false_positive_rate": neg["false_positive_rate"]
        })
    pd.DataFrame(test_csv_rows).to_csv(OUTPUT_EVAL_DIR / "phase_4_test_results.csv", index=False)
    print(f"[+] Saved phase_4_test_results.csv to: {OUTPUT_EVAL_DIR / 'phase_4_test_results.csv'}")

    # 3. Generate outputs/evaluation/phase_4_per_class_results.csv
    class_rows = []
    # Test instance counts: Class 0: 4, Class 1: 1, Class 2: 4, Class 3: 2, Class 4: 1
    gt_instances = {"bud root dropping": 4, "bud rot": 1, "gray leaf spot": 4, "leaf rot": 2, "stembleeding": 1}
    gt_images = {"bud root dropping": 4, "bud rot": 1, "gray leaf spot": 2, "leaf rot": 2, "stembleeding": 1}

    for r in [eval_a, eval_b]:
        exp_id = r["experiment_id"]
        for cname, c_res in r["per_class_results"].items():
            class_rows.append({
                "model": exp_id,
                "class": cname,
                "test_images": gt_images[cname],
                "test_instances": gt_instances[cname],
                "precision": c_res["precision"],
                "recall": c_res["recall"],
                "ap50": c_res["ap50"],
                "ap50_95": c_res["ap50_95"]
            })
    pd.DataFrame(class_rows).to_csv(OUTPUT_EVAL_DIR / "phase_4_per_class_results.csv", index=False)
    print(f"[+] Saved phase_4_per_class_results.csv to: {OUTPUT_EVAL_DIR / 'phase_4_per_class_results.csv'}")

    # 4. Generate outputs/evaluation/phase_4_error_registry.csv
    error_rows = []
    for r in [eval_a, eval_b]:
        for rec in r["image_eval_records"]:
            error_rows.append({
                "image": rec["image"],
                "model": rec["model"],
                "is_healthy_negative": rec["is_healthy_negative"],
                "ground_truth_classes": "|".join(rec["ground_truth_classes"]) if rec["ground_truth_classes"] else "HEALTHY",
                "predicted_classes": "|".join(rec["predicted_classes"]) if rec["predicted_classes"] else "NONE",
                "predicted_confidences": "|".join(str(c) for c in rec["predicted_confidences"]) if rec["predicted_confidences"] else "NONE",
                "error_type": rec["error_type"],
                "false_positive": rec["false_positive"],
                "false_negative": rec["false_negative"],
                "localization_error": rec["localization_error"],
                "notes": rec["notes"]
            })
    pd.DataFrame(error_rows).to_csv(OUTPUT_EVAL_DIR / "phase_4_error_registry.csv", index=False)
    print(f"[+] Saved phase_4_error_registry.csv to: {OUTPUT_EVAL_DIR / 'phase_4_error_registry.csv'}")

    # 5. Generate outputs/evaluation/detection_output_schema.json (Task 25)
    output_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CoconutPalmDetectionOutput",
        "description": "Standardized output schema connecting YOLOv8 detection inference to Phase 5 Plantation Health Monitoring Engine",
        "type": "object",
        "required": [
            "image_id",
            "model",
            "model_version",
            "image_width",
            "image_height",
            "timestamp",
            "detections",
            "health_status"
        ],
        "properties": {
            "image_id": {"type": "string", "description": "Unique image filename or mobile capture UUID"},
            "timestamp": {"type": ["string", "null"], "format": "date-time", "description": "ISO-8601 capture timestamp"},
            "gps_location": {
                "type": ["object", "null"],
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                }
            },
            "model": {"type": "string", "example": "YOLOv8n"},
            "model_version": {"type": "string", "example": "v1.0.0-sanitized-exp001"},
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "health_status": {
                "type": "string",
                "enum": ["HEALTHY", "DISEASED", "UNCERTAIN"],
                "description": "High-level plantation health assessment"
            },
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["class_id", "disease_class", "confidence", "bounding_box"],
                    "properties": {
                        "class_id": {"type": "integer", "minimum": 0, "maximum": 4},
                        "disease_class": {
                            "type": "string",
                            "enum": [
                                "bud root dropping",
                                "bud rot",
                                "gray leaf spot",
                                "leaf rot",
                                "stembleeding"
                            ]
                        },
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "bounding_box": {
                            "type": "object",
                            "required": ["x1", "y1", "x2", "y2"],
                            "properties": {
                                "x1": {"type": "number"},
                                "y1": {"type": "number"},
                                "x2": {"type": "number"},
                                "y2": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }
    }
    with open(OUTPUT_EVAL_DIR / "detection_output_schema.json", "w", encoding="utf-8") as f:
        json.dump(output_schema, f, indent=2)
    print(f"[+] Saved detection_output_schema.json to: {OUTPUT_EVAL_DIR / 'detection_output_schema.json'}")

    # 6. Generate Research Figures (Task 17)
    # Figure 1: Validation vs Test mAP Comparison
    val_map50 = [0.6186, 0.5752]
    test_map50 = [eval_a["overall_metrics"]["mAP50"], eval_b["overall_metrics"]["mAP50"]]
    val_map50_95 = [0.2600, 0.2426]
    test_map50_95 = [eval_a["overall_metrics"]["mAP50_95"], eval_b["overall_metrics"]["mAP50_95"]]

    x = np.arange(2)
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    r1 = ax.bar(x - 1.5*width, val_map50, width, label="Val mAP@0.5", color="#1f77b4", edgecolor="black")
    r2 = ax.bar(x - 0.5*width, test_map50, width, label="Test mAP@0.5", color="#aec7e8", edgecolor="black")
    r3 = ax.bar(x + 0.5*width, val_map50_95, width, label="Val mAP@0.5:0.95", color="#2ca02c", edgecolor="black")
    r4 = ax.bar(x + 1.5*width, test_map50_95, width, label="Test mAP@0.5:0.95", color="#98df8a", edgecolor="black")

    ax.set_ylabel("Mean Average Precision", fontsize=11)
    ax.set_title("Validation vs Test Generalization Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["EXP-001 (YOLOv8n @ 640)", "EXP-002 (YOLOv8n @ 512)"], fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in list(r1) + list(r2) + list(r3) + list(r4):
        h = rect.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "val_vs_test_map_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Per-Class Test AP Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = list(CLASS_NAMES.values())
    ap_a = [eval_a["per_class_results"][c]["ap50_95"] for c in classes]
    ap_b = [eval_b["per_class_results"][c]["ap50_95"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width/2, ap_a, width, label="EXP-001 (640x640)", color="#3f51b5", edgecolor="black")
    rects2 = ax.bar(x + width/2, ap_b, width, label="EXP-002 (512x512)", color="#ff9800", edgecolor="black")

    ax.set_ylabel("Test AP@0.5:0.95", fontsize=11)
    ax.set_title("Per-Class AP@0.5:0.95 on Untouched Test Set", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "per_class_test_ap_comparison.png", dpi=300)
    plt.close()

    # Figure 3: CPU Latency Comparison on Test Set
    fig, ax = plt.subplots(figsize=(7, 5))
    lats = [eval_a["latency_benchmark"]["mean_total_latency_ms"], eval_b["latency_benchmark"]["mean_total_latency_ms"]]
    fps_vals = [eval_a["latency_benchmark"]["estimated_fps"], eval_b["latency_benchmark"]["estimated_fps"]]

    bars = ax.bar(["EXP-001 (640x640)", "EXP-002 (512x512)"], lats, color=["#e91e63", "#00bcd4"], edgecolor="black", width=0.4)
    ax.set_ylabel("Inference Latency on CPU (ms / image)", fontsize=11)
    ax.set_title("Inference Latency on Test Set (AMD Ryzen 5 7520U)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 2, f"{h:.1f} ms\n({fps_vals[i]} FPS)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "model_latency_comparison.png", dpi=300)
    plt.close()

    print(f"[+] Saved evaluation research figures to: {OUTPUT_FIG_DIR}")
    print("\n" + "=" * 75)
    print("[+] PHASE 4 FINAL TEST EVALUATION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
