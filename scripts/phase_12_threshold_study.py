"""
Phase 12 — Confidence Threshold Sensitivity Study (Validation-Only).

Sweeps confidence threshold from 0.05 to 0.90 on the VALIDATION set only.
The untouched test set is NOT re-evaluated here.

Outputs:
  outputs/phase_12/threshold_study/threshold_sweep_results.json
  outputs/phase_12/threshold_study/threshold_sweep_results.csv
  outputs/phase_12/threshold_study/threshold_sweep_per_class.csv
  outputs/phase_12/figures/threshold_sensitivity.png
"""

import os
import json
import csv
import time
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "val" / "images"
VAL_LBL_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "val" / "labels"

MODEL_A_PATH = BASE_DIR / "outputs" / "training" / "EXP-001_baseline_yolov8n" / "weights" / "best.pt"
MODEL_B_PATH = BASE_DIR / "outputs" / "training" / "EXP-002_imgsz512" / "weights" / "best.pt"

OUT_DIR = BASE_DIR / "outputs" / "phase_12" / "threshold_study"
FIG_DIR = BASE_DIR / "outputs" / "phase_12" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "bud root dropping", 1: "bud rot", 2: "gray leaf spot", 3: "leaf rot", 4: "stembleeding"}
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.95, 0.05)]
IOU_MATCH_THRESH = 0.5


def load_ground_truth(lbl_dir, image_path):
    lbl = Path(lbl_dir) / f"{Path(image_path).stem}.txt"
    boxes = []
    if lbl.exists():
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    boxes.append((cid, xc, yc, w, h))
    return boxes


def yolo_to_xyxy_norm(xc, yc, w, h):
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def box_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def evaluate_at_threshold(model, img_paths, lbl_dir, conf_thresh, imgsz):
    """Evaluate model on image list at a single confidence threshold."""
    from ultralytics import YOLO
    tp_total, fp_total, fn_total = 0, 0, 0
    per_class_tp = {c: 0 for c in CLASS_NAMES}
    per_class_fp = {c: 0 for c in CLASS_NAMES}
    per_class_fn = {c: 0 for c in CLASS_NAMES}
    confidences_tp = []
    confidences_fp = []

    for img_path in img_paths:
        gts = load_ground_truth(lbl_dir, img_path)
        gt_boxes = [yolo_to_xyxy_norm(*g[1:]) for g in gts]
        gt_classes = [g[0] for g in gts]

        results = model.predict(str(img_path), conf=conf_thresh, iou=0.7, imgsz=imgsz, verbose=False)
        pred_boxes = []
        pred_classes = []
        pred_confs = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # boxes.xyxyn gives normalized coords
                xyxyn = box.xyxyn[0].tolist()
                cid = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                pred_boxes.append(xyxyn)
                pred_classes.append(cid)
                pred_confs.append(conf)

        matched_gt = set()
        matched_pred = set()
        for pi, (pb, pc, pconf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
            best_iou, best_gi = 0.0, -1
            for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
                if gi in matched_gt:
                    continue
                if gc != pc:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= IOU_MATCH_THRESH and best_gi >= 0:
                tp_total += 1
                per_class_tp[pc] += 1
                confidences_tp.append(pconf)
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp_total += 1
                per_class_fp[pc] += 1
                confidences_fp.append(pconf)

        # FN = unmatched GT boxes
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn_total += 1
                per_class_fn[gt_classes[gi]] += 1

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "conf_threshold": conf_thresh,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_class_tp": per_class_tp,
        "per_class_fp": per_class_fp,
        "per_class_fn": per_class_fn,
        "confidences_tp": confidences_tp,
        "confidences_fp": confidences_fp,
    }


def run_threshold_study(model_path, imgsz, model_id, img_paths):
    from ultralytics import YOLO
    print(f"\n[PHASE 12] Threshold sweep: {model_id} | imgsz={imgsz}")
    model = YOLO(str(model_path))
    results = []
    for conf in THRESHOLDS:
        print(f"  conf={conf:.2f} ...", end=" ", flush=True)
        t0 = time.time()
        r = evaluate_at_threshold(model, img_paths, VAL_LBL_DIR, conf, imgsz)
        elapsed = round(time.time() - t0, 2)
        r["elapsed_s"] = elapsed
        r["model_id"] = model_id
        results.append(r)
        print(f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  ({elapsed}s)")
    return results


def find_optimal_threshold(sweep_results):
    """Find threshold maximising F1."""
    best = max(sweep_results, key=lambda x: x["f1"])
    return best


def save_csv(sweep_results, path):
    fields = ["model_id", "conf_threshold", "tp", "fp", "fn", "precision", "recall", "f1", "elapsed_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sweep_results:
            row = {k: r[k] for k in fields}
            w.writerow(row)


def save_per_class_csv(all_results, path):
    """Per-class precision/recall at each threshold."""
    rows = []
    for r in all_results:
        for cid, cname in CLASS_NAMES.items():
            tp = r["per_class_tp"][cid]
            fp = r["per_class_fp"][cid]
            fn = r["per_class_fn"][cid]
            p = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * p * rec / (p + rec + 1e-9)
            rows.append({
                "model_id": r["model_id"],
                "conf_threshold": r["conf_threshold"],
                "class_id": cid,
                "class_name": cname,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })
    fields = ["model_id", "conf_threshold", "class_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_threshold_sensitivity(results_a, results_b, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 12: Confidence Threshold Sensitivity (Validation Set)", fontsize=13, fontweight="bold")
    colors = {"Precision": "#2196F3", "Recall": "#FF5722", "F1": "#4CAF50"}

    for ax, results, title in zip(axes, [results_a, results_b], ["EXP-001 (640×640)", "EXP-002 (512×512)"]):
        threshs = [r["conf_threshold"] for r in results]
        prec = [r["precision"] for r in results]
        rec = [r["recall"] for r in results]
        f1 = [r["f1"] for r in results]

        ax.plot(threshs, prec, "o-", color=colors["Precision"], label="Precision", linewidth=2)
        ax.plot(threshs, rec, "s-", color=colors["Recall"], label="Recall", linewidth=2)
        ax.plot(threshs, f1, "^-", color=colors["F1"], label="F1", linewidth=2.5)

        best = max(results, key=lambda x: x["f1"])
        ax.axvline(x=best["conf_threshold"], color="#4CAF50", linestyle="--", alpha=0.7,
                   label=f"Optimal conf={best['conf_threshold']:.2f} (F1={best['f1']:.3f})")

        ax.set_xlabel("Confidence Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PHASE 12] Threshold sensitivity figure saved: {out_path}")


def main():
    img_paths = sorted(VAL_IMG_DIR.glob("*.jpg")) + sorted(VAL_IMG_DIR.glob("*.png"))
    print(f"[PHASE 12] Validation images found: {len(img_paths)}")

    if not img_paths:
        print("[ERROR] No validation images found.")
        return

    results_a = run_threshold_study(MODEL_A_PATH, 640, "EXP-001_baseline_yolov8n", img_paths)
    results_b = run_threshold_study(MODEL_B_PATH, 512, "EXP-002_imgsz512", img_paths)

    all_results = results_a + results_b

    # Strip large internal lists before saving JSON
    def strip_for_json(r):
        d = dict(r)
        d.pop("confidences_tp", None)
        d.pop("confidences_fp", None)
        return d

    out_json = OUT_DIR / "threshold_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "meta": {
                    "phase": "Phase 12 — Threshold Sensitivity Study",
                    "dataset": "Validation set (val/images)",
                    "note": "Test set NOT re-evaluated. Val-only sweep.",
                    "iou_match_threshold": IOU_MATCH_THRESH,
                    "thresholds_swept": THRESHOLDS,
                    "val_images": len(img_paths),
                },
                "EXP-001_baseline_yolov8n": [strip_for_json(r) for r in results_a],
                "EXP-002_imgsz512": [strip_for_json(r) for r in results_b],
                "optimal_thresholds": {
                    "EXP-001_baseline_yolov8n": strip_for_json(find_optimal_threshold(results_a)),
                    "EXP-002_imgsz512": strip_for_json(find_optimal_threshold(results_b)),
                },
            },
            f, indent=2,
        )
    print(f"[PHASE 12] JSON results saved: {out_json}")

    save_csv(all_results, OUT_DIR / "threshold_sweep_results.csv")
    save_per_class_csv(all_results, OUT_DIR / "threshold_sweep_per_class.csv")
    print(f"[PHASE 12] CSV results saved.")

    plot_threshold_sensitivity(results_a, results_b, FIG_DIR / "threshold_sensitivity.png")

    # Print optimal summary
    for results, name in [(results_a, "EXP-001"), (results_b, "EXP-002")]:
        best = find_optimal_threshold(results)
        print(f"\n[PHASE 12] {name} optimal conf_threshold={best['conf_threshold']:.2f}  "
              f"P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}")

    print("\n[PHASE 12] Threshold sensitivity study COMPLETE.")


if __name__ == "__main__":
    main()
