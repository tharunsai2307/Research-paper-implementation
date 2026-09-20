"""
Phase 12 — Per-Class Error Taxonomy & Calibration Analysis.

Derives from existing Phase 4 test evaluation records (phase_4_test_results.json).
Does NOT re-run the test set.

Analyses:
  1. Per-class error taxonomy (FP, FN, TRUE_NEGATIVE) from Phase 4 image-level records
  2. Confidence calibration: ECE (Expected Calibration Error) & reliability diagram
  3. Per-class confusion summary table
  4. Ablation summary table (imgsz effect, model parameter count, mAP50, latency)

Outputs:
  outputs/phase_12/error_analysis/per_class_error_taxonomy.json
  outputs/phase_12/error_analysis/per_class_error_taxonomy.csv
  outputs/phase_12/error_analysis/calibration_analysis.json
  outputs/phase_12/error_analysis/ablation_summary.json
  outputs/phase_12/error_analysis/ablation_summary.csv
  outputs/phase_12/figures/error_taxonomy.png
  outputs/phase_12/figures/calibration_reliability.png
  outputs/phase_12/figures/ablation_comparison.png
"""

import json
import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE4_JSON = BASE_DIR / "outputs" / "evaluation" / "phase_4_test_results.json"
PHASE3_JSON = BASE_DIR / "outputs" / "training" / "phase_3_results.json"

OUT_DIR = BASE_DIR / "outputs" / "phase_12" / "error_analysis"
FIG_DIR = BASE_DIR / "outputs" / "phase_12" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bud root dropping", "bud rot", "gray leaf spot", "leaf rot", "stembleeding"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────
# 1. Per-class Error Taxonomy from Phase 4 records
# ─────────────────────────────────────────────
def build_error_taxonomy(phase4_data):
    taxonomy = {}
    for model_id, model_data in phase4_data["model_evaluations"].items():
        records = model_data.get("image_eval_records", [])
        class_counts = {cn: {"fp": 0, "fn": 0, "tn": 0, "tp": 0, "total_gt": 0} for cn in CLASS_NAMES}

        for rec in records:
            gt_classes = rec.get("ground_truth_classes", [])
            pred_classes = rec.get("predicted_classes", [])
            error_type = rec.get("error_type", "")
            is_healthy = rec.get("is_healthy_negative", False)

            if is_healthy:
                # Healthy image: any prediction = FP
                for pc in pred_classes:
                    if pc in class_counts:
                        class_counts[pc]["fp"] += 1
                # No GT disease classes: all classes get a TN for this image
                for cn in CLASS_NAMES:
                    class_counts[cn]["tn"] += 1
                continue

            # Disease image
            for gc in gt_classes:
                if gc in class_counts:
                    class_counts[gc]["total_gt"] += 1

            if error_type == "FALSE_NEGATIVE_MISSED":
                for gc in gt_classes:
                    if gc in class_counts:
                        class_counts[gc]["fn"] += 1
            elif error_type == "TRUE_POSITIVE":
                for gc in gt_classes:
                    if gc in class_counts:
                        class_counts[gc]["tp"] += 1
            elif error_type == "FALSE_POSITIVE":
                for pc in pred_classes:
                    if pc in class_counts:
                        class_counts[pc]["fp"] += 1

        # Derive precision/recall per class
        for cn in CLASS_NAMES:
            cc = class_counts[cn]
            tp = cc["tp"]; fp = cc["fp"]; fn = cc["fn"]
            cc["precision"] = round(tp / (tp + fp + 1e-9), 4)
            cc["recall"] = round(tp / (tp + fn + 1e-9), 4)
            cc["f1"] = round(2 * cc["precision"] * cc["recall"] /
                              (cc["precision"] + cc["recall"] + 1e-9), 4)
            cc["class_name"] = cn

        # Also pull per-class AP from Phase 4 official metrics
        per_class = model_data.get("per_class_results", {})
        for cn in CLASS_NAMES:
            if cn in per_class:
                class_counts[cn]["ap50"] = per_class[cn].get("ap50", 0)
                class_counts[cn]["ap50_95"] = per_class[cn].get("ap50_95", 0)
                class_counts[cn]["official_precision"] = per_class[cn].get("precision", 0)
                class_counts[cn]["official_recall"] = per_class[cn].get("recall", 0)

        taxonomy[model_id] = {"per_class": class_counts, "total_records": len(records)}
    return taxonomy


def save_taxonomy_csv(taxonomy, path):
    fields = ["model_id", "class_name", "tp", "fp", "fn", "tn", "total_gt",
              "precision", "recall", "f1", "ap50", "ap50_95", "official_precision", "official_recall"]
    rows = []
    for model_id, data in taxonomy.items():
        for cn, cc in data["per_class"].items():
            row = {"model_id": model_id}
            row.update(cc)
            rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_error_taxonomy(taxonomy, out_path):
    models = list(taxonomy.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=False)
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Phase 12: Per-Class Error Taxonomy (Test Set, Phase 4 Records)", fontsize=12, fontweight="bold")

    for ax, model_id in zip(axes, models):
        data = taxonomy[model_id]["per_class"]
        classes = CLASS_NAMES
        tps = [data[c]["tp"] for c in classes]
        fps = [data[c]["fp"] for c in classes]
        fns = [data[c]["fn"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25
        ax.bar(x - width, tps, width, label="TP", color="#4CAF50", edgecolor="white")
        ax.bar(x, fps, width, label="FP", color="#FF5722", edgecolor="white")
        ax.bar(x + width, fns, width, label="FN", color="#2196F3", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" ", "\n") for c in classes], fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(model_id.replace("_", " "), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PHASE 12] Error taxonomy figure saved: {out_path}")


# ─────────────────────────────────────────────
# 2. Calibration Analysis
# ─────────────────────────────────────────────
def compute_calibration(phase4_data):
    """
    Expected Calibration Error (ECE) derived from Phase 4 confidence records.
    We bucket predictions by confidence and compare to observed accuracy.
    NOTE: accuracy here is binary: did the prediction match a GT box?
    """
    results = {}
    for model_id, model_data in phase4_data["model_evaluations"].items():
        records = model_data.get("image_eval_records", [])
        confs = []
        correct = []

        for rec in records:
            pred_confs = rec.get("predicted_confidences", [])
            error_type = rec.get("error_type", "")
            is_healthy = rec.get("is_healthy_negative", False)

            for conf in pred_confs:
                confs.append(conf)
                if is_healthy:
                    # Prediction on healthy = incorrect
                    correct.append(0)
                elif error_type == "TRUE_POSITIVE":
                    correct.append(1)
                elif error_type in ("FALSE_POSITIVE", "FALSE_NEGATIVE_MISSED"):
                    # Overly conservative — mark FP confs as incorrect
                    correct.append(0)
                else:
                    correct.append(0)

        if not confs:
            results[model_id] = {"ece": None, "num_predictions": 0, "note": "No predictions with confidence recorded"}
            continue

        confs = np.array(confs)
        correct = np.array(correct)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_data = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                bin_data.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0,
                                 "mean_conf": None, "accuracy": None})
                continue
            mean_conf = float(confs[mask].mean())
            acc = float(correct[mask].mean())
            bin_data.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(mask.sum()),
                             "mean_conf": round(mean_conf, 4), "accuracy": round(acc, 4)})
            ece += (mask.sum() / len(confs)) * abs(acc - mean_conf)

        results[model_id] = {
            "ece": round(float(ece), 4),
            "num_predictions": len(confs),
            "num_correct": int(correct.sum()),
            "bins": bin_data,
        }

    return results


def plot_calibration(calib_results, out_path):
    models = list(calib_results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(12, 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Phase 12: Confidence Calibration Reliability Diagram (Test Set)", fontsize=12, fontweight="bold")

    for ax, model_id in zip(axes, models):
        data = calib_results[model_id]
        bins = [b for b in data.get("bins", []) if b["count"] > 0 and b["accuracy"] is not None]
        if not bins:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_id)
            continue

        mean_confs = [b["mean_conf"] for b in bins]
        accs = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Perfect calibration")
        sc = ax.scatter(mean_confs, accs, c=counts, cmap="Blues", s=100, zorder=5,
                        edgecolors="steelblue", linewidths=1.5, label="Bins (size=count)")

        ece = data.get("ece", None)
        ece_str = f"ECE={ece:.4f}" if ece is not None else "ECE=N/A"
        ax.set_xlabel("Mean Confidence", fontsize=10)
        ax.set_ylabel("Fraction Correct", fontsize=10)
        ax.set_title(f"{model_id.replace('_', ' ')}\n{ece_str}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="# predictions")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PHASE 12] Calibration figure saved: {out_path}")


# ─────────────────────────────────────────────
# 3. Ablation Summary Table
# ─────────────────────────────────────────────
def build_ablation_summary(phase3_data, phase4_data):
    rows = []
    for exp in phase3_data.get("completed_experiments", []):
        eid = exp["experiment_id"]
        # Get test results from phase 4
        p4_model = phase4_data["model_evaluations"].get(eid, {})
        test_metrics = p4_model.get("overall_metrics", {})
        latency = p4_model.get("latency_benchmark", {})
        val_metrics = exp.get("overall_metrics", {})
        row = {
            "experiment_id": eid,
            "architecture": exp.get("model_architecture", "yolov8n"),
            "imgsz": exp.get("imgsz"),
            "epochs": exp.get("epochs"),
            "parameters": exp.get("parameters"),
            "model_size_mb": exp.get("model_size_mb"),
            "training_duration_s": exp.get("training_duration_seconds"),
            "val_mAP50": val_metrics.get("mAP50"),
            "val_mAP50_95": val_metrics.get("mAP50_95"),
            "val_precision": val_metrics.get("precision"),
            "val_recall": val_metrics.get("recall"),
            "val_f1": val_metrics.get("f1_score"),
            "test_precision": test_metrics.get("precision"),
            "test_recall": test_metrics.get("recall"),
            "test_mAP50": test_metrics.get("mAP50"),
            "test_mAP50_95": test_metrics.get("mAP50_95"),
            "test_f1": test_metrics.get("f1_score"),
            "cpu_mean_latency_ms": latency.get("mean_total_latency_ms") if latency else None,
            "estimated_fps": latency.get("estimated_fps") if latency else None,
        }
        rows.append(row)
    return rows


def plot_ablation(ablation_rows, out_path):
    if not ablation_rows:
        return

    exps = [r["experiment_id"].replace("_", "\n") for r in ablation_rows]
    metrics = {
        "mAP50 (Val)": [r["val_mAP50"] for r in ablation_rows],
        "mAP50 (Test)": [r["test_mAP50"] for r in ablation_rows],
        "mAP50-95 (Val)": [r["val_mAP50_95"] for r in ablation_rows],
        "mAP50-95 (Test)": [r["test_mAP50_95"] for r in ablation_rows],
    }
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    x = np.arange(len(exps))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 12: Ablation Summary — Model Comparison", fontsize=12, fontweight="bold")

    # Accuracy metrics
    ax = axes[0]
    for i, (metric, vals) in enumerate(metrics.items()):
        vals_clean = [v if v is not None else 0 for v in vals]
        ax.bar(x + (i - 1.5) * width, vals_clean, width, label=metric, color=colors[i], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy Metrics (Val & Test)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    # Latency
    ax = axes[1]
    latencies = [r["cpu_mean_latency_ms"] for r in ablation_rows]
    fps_vals = [r["estimated_fps"] for r in ablation_rows]
    ax2 = ax.twinx()
    bars = ax.bar(x - 0.15, [l if l else 0 for l in latencies], 0.3,
                  color="#FF5722", edgecolor="white", label="Latency (ms)")
    ax2.bar(x + 0.15, [f if f else 0 for f in fps_vals], 0.3,
            color="#2196F3", edgecolor="white", alpha=0.7, label="FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=9)
    ax.set_ylabel("Latency (ms)", color="#FF5722")
    ax2.set_ylabel("FPS", color="#2196F3")
    ax.set_title("CPU Inference Latency & Throughput")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PHASE 12] Ablation comparison figure saved: {out_path}")


def main():
    print("[PHASE 12] Error Analysis, Calibration & Ablation")
    print("=" * 60)

    with open(PHASE4_JSON) as f:
        phase4_data = json.load(f)
    with open(PHASE3_JSON) as f:
        phase3_data = json.load(f)

    # 1. Error taxonomy
    print("\n[1/3] Building per-class error taxonomy...")
    taxonomy = build_error_taxonomy(phase4_data)
    out_json = OUT_DIR / "per_class_error_taxonomy.json"
    with open(out_json, "w") as f:
        json.dump({"meta": {"phase": "Phase 12", "source": "Phase 4 test records",
                            "note": "Derived from image_eval_records — no test re-run"},
                   "taxonomy": taxonomy}, f, indent=2)
    save_taxonomy_csv(taxonomy, OUT_DIR / "per_class_error_taxonomy.csv")
    plot_error_taxonomy(taxonomy, FIG_DIR / "error_taxonomy.png")
    print(f"  Error taxonomy saved.")

    # 2. Calibration
    print("\n[2/3] Computing confidence calibration (ECE)...")
    calib = compute_calibration(phase4_data)
    with open(OUT_DIR / "calibration_analysis.json", "w") as f:
        json.dump({"meta": {"phase": "Phase 12", "source": "Phase 4 predicted_confidences",
                            "note": "ECE computed from test-set predictions"},
                   "calibration": calib}, f, indent=2)
    plot_calibration(calib, FIG_DIR / "calibration_reliability.png")
    for mid, cd in calib.items():
        ece_str = f"{cd['ece']:.4f}" if cd.get("ece") is not None else "N/A"
        print(f"  {mid}: ECE={ece_str}  n_predictions={cd.get('num_predictions', 0)}")

    # 3. Ablation summary
    print("\n[3/3] Building ablation summary table...")
    ablation_rows = build_ablation_summary(phase3_data, phase4_data)
    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump({"meta": {"phase": "Phase 12",
                            "source": "Phase 3 training results + Phase 4 test results",
                            "note": "Ablation: image size effect on accuracy and latency"},
                   "experiments": ablation_rows}, f, indent=2)

    fields = list(ablation_rows[0].keys()) if ablation_rows else []
    with open(OUT_DIR / "ablation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ablation_rows)

    plot_ablation(ablation_rows, FIG_DIR / "ablation_comparison.png")
    print(f"  Ablation summary saved.")

    print("\n[PHASE 12] Error analysis, calibration & ablation COMPLETE.")


if __name__ == "__main__":
    main()
