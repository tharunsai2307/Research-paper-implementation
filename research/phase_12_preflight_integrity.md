# Phase 12 — Preflight Integrity Report

**Project:** A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications

**Date:** 2026-09-20

---

## Purpose

This document records the pre-Phase-12 integrity verification: frozen model checksums, metric consistency across prior phases, and enforcement of all research-integrity constraints.

---

## 1. Phase 3 Training — Protected Artifacts

| Experiment | Architecture | imgsz | Val mAP50 | Val mAP50-95 | Status |
|---|---|---|---|---|---|
| EXP-001_baseline_yolov8n | YOLOv8n | 640 | 0.6186 | 0.2600 | ✅ VERIFIED |
| EXP-002_imgsz512         | YOLOv8n | 512 | 0.5752 | 0.2426 | ✅ VERIFIED |

Checkpoints verified present:
- `outputs/training/EXP-001_baseline_yolov8n/weights/best.pt`
- `outputs/training/EXP-002_imgsz512/weights/best.pt`

> [!IMPORTANT]
> **DO NOT RETRAIN.** Phase 3 models are frozen. Phase 12 only reads existing checkpoints for inference on the validation set.

---

## 2. Phase 4 Test Evaluation — Frozen Metrics

The untouched test set was evaluated ONCE in Phase 4. Phase 12 does not re-evaluate the test set.

| Experiment | Precision | Recall | mAP50  | mAP50-95 | F1     |
|---|---|---|---|---|---|
| EXP-001_baseline_yolov8n | 0.8160 | 0.4000 | 0.6343 | 0.3663 | 0.5368 |
| EXP-002_imgsz512         | 0.9044 | 0.5333 | 0.6862 | 0.4055 | 0.6667 |

Source: `outputs/evaluation/phase_4_test_results.json` — timestamp preserved.

> [!CAUTION]
> These metrics are the **final test-set ground truth**. Phase 12 analyses that appear to contradict these values are errors in Phase 12, not corrections to Phase 4.

---

## 3. Phase 5 Health Monitoring Engine

Validator: `scripts/validate_health_monitoring.py`
Status: **11/11 PASS** (verified)
Health engine logic: unchanged.

---

## 4. Phase 6 Mobile Deployment

FastAPI + PWA deployment verified.
CPU inference latency: EXP-001 = 110.07 ms, EXP-002 = 68.74 ms.
Phase 6 tests: **8/8 PASS**.

---

## 5. Phase 7–11 Audit

| Phase | Outcome |
|---|---|
| Phase 7  | Segmentation & temporal data capability gap documented |
| Phase 8  | Real data acquisition; 100 images registered |
| Phase 9  | Benchmark dataset: 100 images, 0 masks, training gate = BLOCKED |
| Phase 10 | Annotation workspace established; 0 accepted masks |
| Phase 11 | Double-annotator QC protocol; training gate = BLOCKED |

> [!NOTE]
> The YOLOv8-Seg training gate remains BLOCKED. Phase 12 does not create, modify, or unlock segmentation training.

---

## 6. Phase 12 Scope

Phase 12 is a **validation-only, analysis-only** phase:
- Uses validation set (never test set) for threshold sweeps and robustness tests
- Uses Phase 4 records (never re-runs test set) for error taxonomy and calibration
- Does not retrain any model
- Does not create fake or synthetic results
- Does not unlock segmentation training

---

## Integrity Attestation

All Phase 12 results derive exclusively from:
1. Existing validated checkpoints (Phase 3)
2. Existing validated test records (Phase 4)
3. Validation set images (for val-only analyses)

No fabricated metrics. No unsupported capability claims.
