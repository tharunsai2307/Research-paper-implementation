# Bug Fix Report: End-to-End Detection Failure

**Date:** 2026-09-20
**Project:** A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications

---

## Problem

When a genuine coconut disease image (including images used during model training) is uploaded to the application, the UI displays:
```
Observation Status: NO DISEASE DETECTED
Confirmed Lesions: 0
```
even though the model genuinely detects the disease.

---

## Root Cause

**Two-tier confidence architecture with a misconfigured operational threshold.**

The application uses:
```
DEFAULT_OPERATIONAL_THRESHOLD = 0.25   ← accepts into accepted_detections
DEFAULT_CANDIDATE_THRESHOLD   = 0.10   ← model inference floor
```

Phase 12 (Threshold Sensitivity Study, 2026-09-20) established that:
```
EXP-002 best confidence on training image BudRot004.jpg = 0.1592
EXP-001 best confidence on training image BudRot004.jpg = 0.0689
```

Since 0.1592 < 0.25, the detection is placed in `candidate_detections` (not `accepted_detections`), and `observation_status = LOW_CONFIDENCE_CANDIDATE`, not `DISEASE_DETECTED`.

---

## Evidence

Raw inference on `BudRot004.jpg` (genuine training image, ground truth class=1 bud rot):

```
Model: EXP-002 (deployed model)
  conf=0.01 → 11 detections  [bud rot conf=0.1592] ...
  conf=0.05 → 3  detections  [bud rot conf=0.1592, 0.1181, 0.0902]
  conf=0.10 → 2  detections  [bud rot conf=0.1592, 0.1181]
  conf=0.15 → 1  detection   [bud rot conf=0.1592]
  conf=0.25 → 0  detections  ← DEFAULT_OPERATIONAL_THRESHOLD

Model: EXP-001
  conf=0.05 → 1  detection   [bud rot conf=0.0689]
  conf=0.10 → 0  detections
  conf=0.25 → 0  detections
```

**Conclusion: CASE A — Model detects disease. Application loses the detection downstream.**

---

## Affected Component

`src/health_monitoring/schemas.py` — Line 23:
```python
DEFAULT_OPERATIONAL_THRESHOLD = 0.25   # ← TOO HIGH for this model's confidence distribution
```

The inference engine uses this value to split detections:
- `conf >= operational_threshold` → `accepted_detections` → `DISEASE_DETECTED`
- `candidate_threshold <= conf < operational_threshold` → `candidate_detections` → `LOW_CONFIDENCE_CANDIDATE`

At 0.25, the deployed model's actual predictions never reach `accepted_detections`.

---

## Pipeline Confidence Filter Table

| Location | Threshold Value | Purpose |
|---|---|---|
| `schemas.py` DEFAULT_CANDIDATE_THRESHOLD | 0.10 | Model inference floor (correct) |
| `schemas.py` DEFAULT_OPERATIONAL_THRESHOLD | **0.25** ← **BUG** | accepted_detections gate |
| `inference.py` model.predict(conf=...) | 0.10 (candidate_threshold) | YOLO inference |
| `inference.py` if conf >= operational_threshold | **0.25** | accept/reject split |
| `main.py` HealthInferenceEngine(operational_threshold=...) | inherits 0.25 | passes to engine |
| Frontend `app.js` | No additional filtering | Correctly renders API response |
| Frontend HTML labels | "(conf ≥ 0.25)" | Display label only |

The bug is **entirely in the backend** at `DEFAULT_OPERATIONAL_THRESHOLD`.

---

## Fix

Change `DEFAULT_OPERATIONAL_THRESHOLD` from `0.25` to `0.10` in `src/health_monitoring/schemas.py`.

**Justification:**
- Phase 12 threshold sweep (validation set, 35 images) found optimal F1 at conf=0.10 for EXP-002
- The deployed model's confidence distribution peaks below 0.25 for this dataset
- 0.25 is the YOLO default for general-purpose datasets, not calibrated for this specific fine-tuned model
- Phase 4 mAP50 metric is NOT affected — it uses the full PR curve, not a single threshold

**What is NOT changed:**
- Historical Phase 4 test results (frozen)
- Model weights (frozen)
- Dataset (untouched)
- Phase 5 health monitoring logic
- API response schema fields
- Frontend rendering logic

The `DEFAULT_CANDIDATE_THRESHOLD` (0.10) becomes the new `DEFAULT_OPERATIONAL_THRESHOLD`.
A new `DEFAULT_CANDIDATE_THRESHOLD` of `0.05` is established for the lower screening tier.

---

## Why the Fix Works

After the fix:
```
model inference at conf=0.05 (new candidate floor)
  bud rot conf=0.1592  →  conf >= 0.10 (new operational)  →  accepted_detections
  bud rot conf=0.1181  →  conf >= 0.10                    →  accepted_detections
  bud rot conf=0.0902  →  0.05 <= conf < 0.10             →  candidate_detections

observation_status = DISEASE_DETECTED
```

---

## Also: Candidate Tier Labels (HTML/Frontend)

The HTML labels `"(conf ≥ 0.25)"` and `"(0.10 ≤ conf < 0.25)"` are display strings. These are updated to match the new thresholds to avoid confusing users. The frontend `app.js` fetches the actual thresholds from the `/api/v1/health` endpoint and displays them dynamically — no JS change required.

---

## Regression Tests

1. All 72 existing unit tests must continue to pass
2. Phase 5/10/11/12 validators must continue to pass
3. Phase 4 frozen test metrics must be bitwise unchanged
4. Segmentation training gate remains BLOCKED (accepted=0 masks)
5. New pipeline test: BudRot004.jpg → DISEASE_DETECTED via the engine at threshold 0.10
