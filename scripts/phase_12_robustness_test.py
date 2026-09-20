"""
Phase 12 — Perturbation-Based Robustness Testing.

Tests model robustness on the VALIDATION set under controlled image perturbations:
  1. Gaussian Blur (sigma = 1, 2, 3)
  2. Gaussian Noise (std = 10, 25, 50)
  3. Brightness Shift (factor = 0.5, 0.75, 1.25, 1.5)
  4. JPEG Compression (quality = 80, 60, 40, 20)

Metric: detection F1 at conf=0.25, IoU-match=0.50

Outputs:
  outputs/phase_12/robustness/robustness_results.json
  outputs/phase_12/robustness/robustness_results.csv
  outputs/phase_12/figures/robustness_perturbation.png
"""

import os
import io
import json
import csv
import time
import tempfile
import shutil
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lazy import to avoid crash if not installed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "val" / "images"
VAL_LBL_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean" / "val" / "labels"

MODEL_A_PATH = BASE_DIR / "outputs" / "training" / "EXP-001_baseline_yolov8n" / "weights" / "best.pt"
MODEL_B_PATH = BASE_DIR / "outputs" / "training" / "EXP-002_imgsz512" / "weights" / "best.pt"

OUT_DIR = BASE_DIR / "outputs" / "phase_12" / "robustness"
FIG_DIR = BASE_DIR / "outputs" / "phase_12" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "bud root dropping", 1: "bud rot", 2: "gray leaf spot", 3: "leaf rot", 4: "stembleeding"}
CONF_THRESH = 0.25
IOU_MATCH = 0.50

PERTURBATIONS = {
    "baseline": [("none", 0)],
    "gaussian_blur": [("sigma_1", 1), ("sigma_2", 2), ("sigma_3", 3)],
    "gaussian_noise": [("std_10", 10), ("std_25", 25), ("std_50", 50)],
    "brightness": [("factor_0.5", 0.5), ("factor_0.75", 0.75), ("factor_1.25", 1.25), ("factor_1.5", 1.5)],
    "jpeg_compression": [("quality_80", 80), ("quality_60", 60), ("quality_40", 40), ("quality_20", 20)],
}


def apply_perturbation(img_array, perturb_type, param):
    """Apply perturbation to a uint8 numpy RGB array. Returns perturbed array."""
    arr = img_array.copy()
    if perturb_type == "none":
        return arr
    elif perturb_type.startswith("sigma"):
        if HAS_CV2:
            arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=param)
        else:
            # Fallback: box blur approximation using numpy
            from scipy.ndimage import gaussian_filter
            arr = np.stack([gaussian_filter(arr[:, :, c], sigma=param) for c in range(3)], axis=2).astype(np.uint8)
    elif perturb_type.startswith("std"):
        noise = np.random.normal(0, param, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif perturb_type.startswith("factor"):
        arr = np.clip(arr.astype(np.float32) * param, 0, 255).astype(np.uint8)
    elif perturb_type.startswith("quality"):
        if HAS_PIL:
            buf = io.BytesIO()
            PILImage.fromarray(arr).save(buf, format="JPEG", quality=param)
            buf.seek(0)
            arr = np.array(PILImage.open(buf))
    return arr


def load_ground_truth(image_path):
    lbl = VAL_LBL_DIR / f"{Path(image_path).stem}.txt"
    boxes = []
    if lbl.exists():
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    boxes.append((cid, xc - w/2, yc - h/2, xc + w/2, yc + h/2))
    return boxes


def box_iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def evaluate_perturbed(model, img_paths, perturb_type, param, imgsz):
    from ultralytics import YOLO
    tp, fp, fn = 0, 0, 0
    tmpdir = Path(tempfile.mkdtemp())
    try:
        np.random.seed(42)  # For noise reproducibility
        for img_path in img_paths:
            # Load image
            if HAS_PIL:
                arr = np.array(PILImage.open(img_path).convert("RGB"))
            elif HAS_CV2:
                arr = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            else:
                # Cannot load image
                continue

            arr_perturbed = apply_perturbation(arr, perturb_type, param)
            tmp_img = tmpdir / img_path.name
            if HAS_PIL:
                PILImage.fromarray(arr_perturbed).save(str(tmp_img))
            elif HAS_CV2:
                cv2.imwrite(str(tmp_img), cv2.cvtColor(arr_perturbed, cv2.COLOR_RGB2BGR))

            gts = load_ground_truth(img_path)

            results = model.predict(str(tmp_img), conf=CONF_THRESH, iou=0.7, imgsz=imgsz, verbose=False)
            pred_boxes = []
            pred_classes = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxyn = box.xyxyn[0].tolist()
                    pred_boxes.append((int(box.cls[0].item()),) + tuple(xyxyn))
                    pred_classes.append(int(box.cls[0].item()))

            matched_gt = set()
            matched_pred = set()
            for pi, pb in enumerate(pred_boxes):
                pc = pb[0]; pb_box = pb[1:]
                best_iou, best_gi = 0.0, -1
                for gi, gt in enumerate(gts):
                    if gi in matched_gt or gt[0] != pc: continue
                    iou = box_iou(pb_box, gt[1:])
                    if iou > best_iou: best_iou, best_gi = iou, gi
                if best_iou >= IOU_MATCH and best_gi >= 0:
                    tp += 1; matched_gt.add(best_gi); matched_pred.add(pi)
                else:
                    fp += 1
            fn += sum(1 for gi in range(len(gts)) if gi not in matched_gt)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)}


def run_robustness_study(model_path, imgsz, model_id, img_paths):
    from ultralytics import YOLO
    print(f"\n[PHASE 12] Robustness study: {model_id} | imgsz={imgsz}")
    model = YOLO(str(model_path))
    results = {}
    for perturb_group, variants in PERTURBATIONS.items():
        results[perturb_group] = []
        for (pname, pval) in variants:
            print(f"  {perturb_group}:{pname} ...", end=" ", flush=True)
            t0 = time.time()
            r = evaluate_perturbed(model, img_paths, pname, pval, imgsz)
            elapsed = round(time.time() - t0, 2)
            r.update({"perturbation_group": perturb_group, "perturbation_name": pname,
                      "param": pval, "model_id": model_id, "elapsed_s": elapsed})
            results[perturb_group].append(r)
            print(f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  ({elapsed}s)")
    return results


def flatten_results(nested):
    flat = []
    for group, variants in nested.items():
        for v in variants:
            flat.append(v)
    return flat


def plot_robustness(results_a_flat, results_b_flat, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    fig.suptitle("Phase 12: Perturbation Robustness (F1 Score, Validation Set)", fontsize=13, fontweight="bold")

    perturb_groups = ["baseline", "gaussian_blur", "gaussian_noise", "brightness", "jpeg_compression"]
    group_labels = {
        "baseline": "Baseline",
        "gaussian_blur": "Gaussian Blur",
        "gaussian_noise": "Gaussian Noise",
        "brightness": "Brightness",
        "jpeg_compression": "JPEG Quality",
    }

    for row_idx, (results_flat, model_name) in enumerate([(results_a_flat, "EXP-001 (640×640)"),
                                                           (results_b_flat, "EXP-002 (512×512)")]):
        by_group = {}
        for r in results_flat:
            g = r["perturbation_group"]
            if g not in by_group:
                by_group[g] = []
            by_group[g].append(r)

        baseline_f1 = by_group.get("baseline", [{}])[0].get("f1", 0)

        for col_idx, group in enumerate(["gaussian_blur", "gaussian_noise", "brightness", "jpeg_compression"]):
            ax = axes[row_idx][col_idx]
            variants = by_group.get(group, [])
            labels = [v["perturbation_name"] for v in variants]
            f1s = [v["f1"] for v in variants]

            colors = ["#2196F3" if f >= baseline_f1 * 0.9 else "#FF5722" for f in f1s]
            bars = ax.bar(range(len(labels)), f1s, color=colors, edgecolor="white", linewidth=0.5)
            ax.axhline(baseline_f1, color="gray", linestyle="--", alpha=0.6, linewidth=1.5, label="Baseline")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("F1 Score" if col_idx == 0 else "", fontsize=9)
            if row_idx == 0:
                ax.set_title(group_labels[group], fontsize=10)
            if col_idx == 0:
                ax.text(-0.3, 0.5, model_name, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center", rotation=90)
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PHASE 12] Robustness figure saved: {out_path}")


def main():
    img_paths = sorted(VAL_IMG_DIR.glob("*.jpg")) + sorted(VAL_IMG_DIR.glob("*.png"))
    print(f"[PHASE 12] Validation images for robustness: {len(img_paths)}")

    if not img_paths:
        print("[ERROR] No validation images found.")
        return

    results_a = run_robustness_study(MODEL_A_PATH, 640, "EXP-001_baseline_yolov8n", img_paths)
    results_b = run_robustness_study(MODEL_B_PATH, 512, "EXP-002_imgsz512", img_paths)

    out_json = OUT_DIR / "robustness_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "meta": {
                "phase": "Phase 12 — Perturbation Robustness Testing",
                "dataset": "Validation set (val/images)",
                "conf_threshold": CONF_THRESH,
                "iou_match_threshold": IOU_MATCH,
                "perturbation_groups": list(PERTURBATIONS.keys()),
                "val_images": len(img_paths),
                "note": "numpy seed=42 for noise reproducibility",
            },
            "EXP-001_baseline_yolov8n": results_a,
            "EXP-002_imgsz512": results_b,
        }, f, indent=2)
    print(f"[PHASE 12] Robustness JSON saved: {out_json}")

    flat_a = flatten_results(results_a)
    flat_b = flatten_results(results_b)

    fields = ["model_id", "perturbation_group", "perturbation_name", "param",
              "tp", "fp", "fn", "precision", "recall", "f1", "elapsed_s"]
    with open(OUT_DIR / "robustness_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_a + flat_b:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[PHASE 12] Robustness CSV saved.")

    plot_robustness(flat_a, flat_b, FIG_DIR / "robustness_perturbation.png")
    print("\n[PHASE 12] Robustness study COMPLETE.")


if __name__ == "__main__":
    main()
