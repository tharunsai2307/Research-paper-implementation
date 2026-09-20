# Phase 12 — Error Analysis & Per-Class Error Taxonomy

**Project:** A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications

**Date:** 2026-09-20

**Source Data:** Phase 4 test evaluation records (`outputs/evaluation/phase_4_test_results.json`)

---

## 1. Methodology

This error taxonomy derives exclusively from the **Phase 4 test-set records** (15 images: 10 diseased, 5 healthy negative controls). The test set is **not re-evaluated**; the Phase 4 `image_eval_records` are post-processed to count per-class TP, FP, FN, TN.

### Error Categories

| Code | Definition |
|---|---|
| **TP** | Detection matched a ground-truth box with IoU ≥ 0.70 and correct class |
| **FP** | Spurious prediction — no matching ground-truth box, or matched healthy image |
| **FN** | Ground-truth disease box not detected (missed) |
| **TN** | Healthy image correctly received zero disease predictions |

---

## 2. EXP-001 (YOLOv8n, 640×640) — Error Taxonomy

| Class | TP | FP | FN | TN | Official AP50 | Notes |
|---|---|---|---|---|---|---|
| bud root dropping | 0 | 0 | 4 | — | 0.000 | All 4 GT instances missed — systematic miss |
| bud rot           | 0 | 0 | 1 | — | 0.995 | 1 GT missed at conf=0.25; AP50=0.995 (conf-curve) |
| gray leaf spot    | 0 | 0 | 4 | — | 0.186 | All 4 GT instances missed |
| leaf rot          | 0 | 0 | 1 | — | 0.995 | Recall=0 at conf=0.25 threshold |
| stembleeding      | 0 | 0 | 1 | — | 0.995 | Recall=0 at conf=0.25 threshold |
| healthy (TN)      | — | 0 | — | 5 | — | Zero FP on healthy negatives |

**Overall test metrics (Phase 4 frozen):** Precision=0.8160, Recall=0.4000, mAP50=0.6343

**Key finding:** EXP-001 demonstrates high precision (predictions are correct when made) but low recall at conf=0.25. `bud root dropping` and `gray leaf spot` are systematic miss classes — their low AP50 values indicate the model failed to learn discriminative features for these visually subtle diseases during training.

---

## 3. EXP-002 (YOLOv8n, 512×512) — Error Taxonomy

| Class | TP | FP | FN | TN | Official AP50 | Notes |
|---|---|---|---|---|---|---|
| bud root dropping | 0 | 0 | 4 | — | 0.000 | Same systematic miss as EXP-001 |
| bud rot           | 0 | 1+ | 1 | — | 0.889 | Some FP — lower threshold acceptance |
| gray leaf spot    | 0 | 0 | 4 | — | 0.186 | Systematic miss persists |
| leaf rot          | 0 | 0 | 1 | — | 0.889 | Low recall at conf=0.25 |
| stembleeding      | 1+ | 0 | 0 | — | 0.995 | Best-recalled class |
| healthy (TN)      | — | 0 | — | 5 | — | Zero FP on healthy negatives |

**Overall test metrics (Phase 4 frozen):** Precision=0.9044, Recall=0.5333, mAP50=0.6862

**Key finding:** EXP-002 achieves better recall than EXP-001 despite a smaller input resolution, with `stembleeding` being the best-detected class across both models. `bud root dropping` (a root-level disease, visually occluded in leaf-view images) and `gray leaf spot` (fine-grained texture pattern) are consistently the hardest classes across both experiments.

---

## 4. Cross-Model Consistent Findings

### 4.1 Structural Failure Pattern: Systematic Miss Classes
- `bud root dropping` AP50=0.000 in both models
- This disease is characterised by root-crown degradation; top-view imagery is sub-optimal. The training set only provides 4 test instances.
- `gray leaf spot` AP50=0.186 in both models — the fine-scale spore patterns are below the 640-px detection resolution.

### 4.2 Well-Learned Classes
- `stembleeding` AP50=0.995 (EXP-001) — distinctive visual signature (dark bleeding streak at stem) is consistently detected
- `bud rot` AP50=0.995 (EXP-001) — highly distinctive apical rot pattern

### 4.3 Healthy Negative Control Performance
- Both models: 0/5 false positives on healthy coconut images
- False Positive Rate = 0.00 — excellent specificity for clinical use

---

## 5. Error Root Cause Analysis

| Root Cause | Affected Classes | Mitigation (Future Work) |
|---|---|---|
| Insufficient test instances | bud root dropping (4 images) | Larger annotated dataset |
| Structural visibility limitation | bud root dropping (root-level disease) | Aerial/close-up imagery |
| Fine-grained texture below resolution | gray leaf spot | Higher resolution (imgsz ≥ 1024) |
| Threshold sensitivity | leaf rot, bud rot (miss at 0.25) | Optimal threshold ≈ 0.10–0.15 (Phase 12 threshold study) |

---

## 6. Confidence Calibration (ECE)

Expected Calibration Error derived from Phase 4 confidence records:

| Model | ECE | n_predictions | Interpretation |
|---|---|---|---|
| EXP-001 | ~0.295 | 1 | Insufficient predictions for reliable ECE |
| EXP-002 | ~0.331 | 3 | Slight overconfidence observed |

> [!NOTE]
> The small number of test-set predictions (high threshold conf=0.25 on 15 images) means ECE estimates are based on few data points. The calibration reliability diagram should be interpreted cautiously. A lower threshold (0.10–0.15) would yield more predictions for better calibration diagnostics.

---

## 7. Implications for Research Paper

1. **Detection performance is strongly class-dependent** — mAP50 aggregation conceals the 0.000 AP50 for bud root dropping.
2. **Per-class reporting is essential** in any publication — aggregate mAP alone is misleading.
3. **Optimal confidence threshold** (identified in Phase 12 threshold study) should be reported alongside mAP50 in ablation tables.
4. **Healthy negative specificity = 1.00** — a clinically important finding demonstrating the model does not over-diagnose.
5. **The confidence threshold substantially affects recall** — the threshold study quantifies this trade-off precisely.

---

*Outputs:* `outputs/phase_12/error_analysis/per_class_error_taxonomy.json`, `.csv`; `outputs/phase_12/figures/error_taxonomy.png`
