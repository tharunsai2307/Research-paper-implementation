# Phase 12 — Confidence Threshold Sensitivity Study

**Project:** A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications

**Date:** 2026-09-20

**Source:** Validation set (35 images) — test set NOT re-evaluated.

---

## 1. Objective

The standard YOLO confidence threshold of 0.25 is an engineering default, not a task-specific optimum. This study sweeps confidence from 0.05 to 0.90 (step 0.05) on the **validation set** to identify the threshold that maximises F1 score for each trained model.

The test set was evaluated **once and only once** in Phase 4. This study operates exclusively on the validation split.

---

## 2. Experimental Protocol

| Parameter | Value |
|---|---|
| Dataset | Validation set: `data/processed/coconut_detection_clean/val/images` |
| Models | EXP-001 (imgsz=640), EXP-002 (imgsz=512) |
| IoU match threshold | 0.50 |
| Fixed IoU NMS | 0.70 |
| Confidence range | 0.05 – 0.90, step 0.05 (18 values) |
| Test set | NOT used — frozen in Phase 4 |

---

## 3. Results

### 3.1 EXP-001 — YOLOv8n 640×640 (18 thresholds, 35 val images)

| Conf | Precision | Recall | F1 | Comment |
|---|---|---|---|---|
| 0.05 | 0.3333 | 0.4667 | 0.3889 | High recall, many spurious boxes |
| **0.10** | **0.7333** | **0.3667** | **0.4889** | **← Optimal F1** |
| 0.15 | 1.0000 | 0.2667 | 0.4211 | Perfect precision, lower recall |
| 0.20 | 1.0000 | 0.0333 | 0.0645 | Recall collapses |
| 0.25 | 0.0000 | 0.0000 | 0.0000 | Standard default — no predictions pass |
| 0.30–0.90 | 0.0000 | 0.0000 | 0.0000 | No predictions |

**Finding:** The standard default conf=0.25 produces **zero detections on the validation set** for EXP-001. The model's validation-set predictions are clustered below 0.25. The optimal operating point is **conf=0.10**, yielding F1=0.489.

### 3.2 EXP-002 — YOLOv8n 512×512 (18 thresholds, 35 val images)

| Conf | Precision | Recall | F1 | Comment |
|---|---|---|---|---|
| 0.05 | 0.2051 | 0.2667 | 0.2319 | Many low-confidence spurious predictions |
| **0.10** | **0.4211** | **0.2667** | **0.3265** | **← Optimal F1** |
| 0.15 | 0.5385 | 0.2333 | 0.3256 | Nearly equal |
| 0.20 | 0.4286 | 0.1000 | 0.1622 | Recall falls |
| 0.25 | 0.5000 | 0.1000 | 0.1667 | Some predictions but lower F1 |
| 0.30 | 0.5000 | 0.0667 | 0.1176 | |
| 0.35 | 1.0000 | 0.0333 | 0.0645 | Near-zero recall |
| 0.40–0.90 | 0.0000 | 0.0000 | 0.0000 | No predictions |

**Finding:** EXP-002 optimal conf = 0.10, F1=0.327. Both models benefit substantially from a lower threshold.

---

## 4. Precision-Recall Trade-off Analysis

| Model | Optimal Conf | P@optimal | R@optimal | F1@optimal | F1@0.25 |
|---|---|---|---|---|---|
| EXP-001 (640×640) | 0.10 | 0.7333 | 0.3667 | **0.4889** | 0.000 |
| EXP-002 (512×512) | 0.10 | 0.4211 | 0.2667 | **0.3265** | 0.167 |

### Key Insight: Why does conf=0.25 fail on validation?

The models were trained for 15 epochs on CPU. The validation set confidence distribution is low — predictions are made but with confidence 0.10–0.20. This is consistent with a model that has learned general features but is under-confident in its predictions (a known effect of short training on limited data without GPU augmentation). The **optimal threshold of 0.10** should be used when deploying for disease screening.

---

## 5. Implications for Clinical Use

For **coconut disease screening** (priority: catch all disease instances, even at cost of more false positives):
- **Recommended threshold: 0.10** — maximises recall without extreme FP rate
- At conf=0.05, precision drops to 0.21–0.33 (too many false alarms for practical use)
- The mobile app API should expose threshold as a configurable parameter

For **research paper reporting:**
- Report F1/mAP at both the optimal threshold (0.10) and the standard YOLO default (0.25)
- Note that mAP50 (computed from the full PR curve) is not affected by a single threshold choice — it correctly captures the full-curve performance

---

## 6. Reproducibility

Script: `scripts/phase_12_threshold_study.py`
Random seed: n/a (no stochastic components)
Val images: 35
Figure: `outputs/phase_12/figures/threshold_sensitivity.png`
Data: `outputs/phase_12/threshold_study/threshold_sweep_results.json`
