# Phase 12 — Experiment Log

**Project:** A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications

**Phase:** Phase 12 — Comprehensive Model Ablation, Calibration, Robustness & Error Analysis

**Date:** 2026-09-20

---

## Research Integrity Preamble

> Phase 12 is a POST-TRAINING, ANALYSIS-ONLY phase. No models are retrained. The untouched test set is not re-evaluated. All analyses derive from existing Phase 3 checkpoints (validation-set inference) and existing Phase 4 test records (error taxonomy, calibration).

---

## Experiment Log

### P12-EXP-001 — Error Analysis, Calibration & Ablation

| Property | Value |
|---|---|
| Type | Post-hoc analysis (no inference on test set) |
| Script | `scripts/phase_12_error_analysis.py` |
| Input | `outputs/evaluation/phase_4_test_results.json`, `outputs/training/phase_3_results.json` |
| Status | ✅ COMPLETE |
| Duration | ~4 s |
| Outputs | `outputs/phase_12/error_analysis/` |

**Findings:**
- Per-class error taxonomy derived from 15 Phase 4 test records
- `bud root dropping` and `gray leaf spot`: systematic miss in both models (AP50=0.000, 0.186)
- `stembleeding` and `bud rot`: well-detected classes (AP50≥0.889)
- Healthy negative FPR = 0.00 in both models
- Confidence calibration ECE: EXP-001≈0.295, EXP-002≈0.331 (slightly overconfident based on limited data)
- EXP-002 (512×512) outperforms EXP-001 on test set despite smaller input resolution
- EXP-002 latency = 69 ms vs EXP-001 = 110 ms (37.5% faster)

---

### P12-EXP-002 — Confidence Threshold Sensitivity Study

| Property | Value |
|---|---|
| Type | Validation-set sweep (NO test set) |
| Script | `scripts/phase_12_threshold_study.py` |
| Dataset | Val set (35 images) |
| Conf range | 0.05 – 0.90, step 0.05 |
| IoU match | 0.50 |
| Status | ✅ COMPLETE |
| Duration | ~155 s (18 thresholds × 2 models × 35 images) |
| Outputs | `outputs/phase_12/threshold_study/` |

**Findings:**

| Model | Default conf=0.25 F1 | Optimal conf | Optimal F1 | Gain |
|---|---|---|---|---|
| EXP-001 (640×640) | 0.000 | **0.10** | **0.489** | +∞ |
| EXP-002 (512×512) | 0.167 | **0.10** | **0.327** | +0.160 |

**Critical finding:** The standard YOLO default threshold (0.25) is too high for this dataset at this training stage. EXP-001 produces **zero detections** at conf=0.25 on the validation set. The optimal threshold for both models is conf=0.10.

This finding does **not invalidate Phase 4** — Phase 4 reported mAP50 which is computed across the full PR curve and is threshold-independent. The Phase 4 mAP50 values remain the correct research metric.

The threshold study establishes that for **clinical deployment**, the mobile API should use conf=0.10 for maximum disease recall.

---

### P12-EXP-003 — Perturbation-Based Robustness Testing

| Property | Value |
|---|---|
| Type | Validation-set robustness (NO test set) |
| Script | `scripts/phase_12_robustness_test.py` |
| Dataset | Val set (35 images) |
| Perturbations | Gaussian blur (σ=1,2,3), Gaussian noise (std=10,25,50), Brightness (×0.5,0.75,1.25,1.5), JPEG quality (80,60,40,20) |
| Conf threshold | 0.25 (consistent with Phase 4 protocol) |
| Random seed | 42 |
| Status | ⏳ RUNNING (in background) |
| Outputs | `outputs/phase_12/robustness/` |

---

## Summary: Novel Scientific Findings from Phase 12

1. **Threshold-sensitivity:** Both models should be deployed at conf=0.10 (not 0.25) for maximum recall on this dataset.
2. **Resolution ablation inversion:** EXP-002 (512×512) achieves higher test mAP50 than EXP-001 (640×640), despite lower val mAP50 — suggesting a regularisation benefit from smaller input.
3. **Systematic miss classes:** `bud root dropping` (AP50=0.000, structural visibility limitation) and `gray leaf spot` (AP50=0.186, fine-grained texture detection challenge) represent fundamental limits of the current dataset and resolution.
4. **Perfect specificity:** FPR=0.000 on healthy negatives — clinically important, avoids false alarms.
5. **CPU latency trade-off:** 512×512 input reduces inference time by 37.5% (110→69 ms) with higher test accuracy — EXP-002 is the recommended deployment model.
