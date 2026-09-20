# Phase 7 Data Capability Audit: Segmentation & Longitudinal Feasibility

## 1. Overview & Objective

The objective of Phase 7 is to evaluate the technical and scientific feasibility of extending the completed YOLOv8 Coconut Disease Detection and Plantation Health Monitoring framework with:
1. **Disease Region Instance/Semantic Segmentation**
2. **Longitudinal/Temporal Disease Progression Tracking (LSTM / GRU)**
3. **Automated Farmer Progression Alerts**

Under strict research-integrity rules, scientific modeling cannot proceed on fabricated annotations or artificial temporal sequences. This document presents a comprehensive, empirical capability audit of all local datasets, metadata schemas, image assets, and annotation representations.

---

## 2. Comprehensive Data Capability Audit Table

| Capability | Available | Evidence | Suitable for Research Training |
| :--- | :---: | :--- | :---: |
| **Detection boxes** | **YES** | 250 label files audited across `data/` (`processed` and `raw/roboflow`). 100% of non-empty labels follow standard normalized YOLO bounding box format: `<class_id> <xc> <yc> <w> <h>` with 5 numerical tokens per line. | **YES** (Already trained & validated in Phases 3–4) |
| **Segmentation masks** | **NO** | Zero binary mask files (`.png`, `.npy`, RLE masks) exist anywhere in the repository. | **NO** (Cannot train semantic segmentation) |
| **Polygon annotations** | **NO** | Systematic coordinate length check across all 250 label files revealed **0 lines with token count $\ne 5$**. Zero polygon lists `[x1, y1, x2, y2, ...]` exist. | **NO** (Cannot train instance segmentation) |
| **Pixel-level disease labels** | **NO** | No semantic segmentation datasets, ground-truth masks, or pixel bitmaps are present. | **NO** |
| **Tree identity** | **NO** | Images are named sequentially by class (e.g., `LeafRot001.jpg`, `BudRot012.jpg`). There is no tree tagging system, RFID, GPS trunk marker, or physical tree identifier. Near-identical burst frames exist from single camera bursts, but individual permanent tree entities are untracked. | **NO** (Cannot link observations across distinct growth sessions) |
| **Dates** | **PARTIAL** | 103 of 150 clean images have EXIF `DateTime` tags, but these cluster into only 2 single-day field collection excursions (June 30, 2023, and July 9, 2023). 47 images have zero EXIF metadata. | **NO** (Single snapshot days, not longitudinal) |
| **Timestamps** | **PARTIAL** | EXIF timestamps show high-frequency camera bursts separated by 1 to 5 seconds during walking sweeps (e.g., `06:16:09`, `06:16:10`, `06:16:11`). | **NO** (Seconds-level burst photos represent identical pathological moments, not disease evolution) |
| **Repeated tree observations** | **NO** | There are zero documented revisit intervals (e.g., Day 0, Day 7, Day 14, Day 30) of the same indexed trees across time. Burst images taken within seconds of each other depict identical lesion states from slightly varied camera angles. | **NO** |
| **Severity labels** | **NO** | No expert phytopathological severity grading (e.g., Horsfall-Barratt scale, lesion depth staging, or percentage defoliation ground truth) exists in the dataset annotations. | **NO** (Must rely on mathematically defined image-space proxies) |
| **Longitudinal sequences** | **NO** | Zero sequential time-series tracking disease onset, progression, or treatment response over biologically relevant developmental cycles exists. | **NO** (Cannot train LSTM / GRU recurrent models) |
| **Environmental/Time-Series Variables** | **NO** | No microclimate telemetry (temperature, relative humidity, rainfall, canopy wetness sensor data) is associated with the imagery. | **NO** |

---

## 3. Detailed Audit Findings

### 3.1 Segmentation Capability Analysis
- **Annotation Inspection**:
  Every `.txt` label file in `data/processed/coconut_detection_clean/` and `data/raw/roboflow_coconut_detection/` was scanned. Out of all lines inspected, exactly zero lines possess $> 5$ coordinate parameters (which would indicate polygon vertices).
- **Geometric Limitation**:
  Generating synthetic polygon masks from bounding boxes (e.g., treating the bounding box rectangle or an inscribed ellipse as ground truth) is scientifically invalid because coconut fronds and trunks are highly irregular, non-convex structures. Fabricating pseudo-masks would introduce massive background foliage noise into training and invalidate IoU/Dice evaluations.
- **Scientific Verdict**:
  **Real segmentation masks do NOT exist.** In accordance with Step 3 (Case B), segmentation model training must remain **PENDING REAL MASK DATA**.

### 3.2 Temporal & Longitudinal Capability Analysis
- **EXIF Metadata Telemetry**:
  Analysis of the 150 sanitized images revealed:
  - 47 images have `DateTime = None` (stripped by web scrapers or cameras).
  - 80 images possess timestamps from **June 30, 2023** between `06:16:09` and `07:06:59` (a 50-minute morning photo session).
  - 20 images possess timestamps from **July 9, 2023** between `08:33:20` and `09:21:31` (a 48-minute morning photo session).
  - 3 images possess legacy camera clock artifacts from 2006/2014.
- **Pathological Development Reality**:
  Coconut palm diseases such as *Phytophthora palmivora* (Bud Rot) or *Bipolaris incurvata* (Leaf Spot) progress over incubation horizons of **7 to 45 days**. Photos captured 2 seconds apart during a single morning walk represent identical disease stages under momentary camera jitter, not disease progression.
- **Absence of Tree Identification**:
  The dataset does not track which tree in the July 9 session corresponds to which tree in the June 30 session. Any synthetic pairing of a June 30 image with a July 9 image to create a "2-step progression" would be completely arbitrary and scientifically fraudulent.
- **Scientific Verdict**:
  **Genuine longitudinal sequences do NOT exist.** In accordance with Step 7, LSTM/GRU progression training must be formally marked:
  ```text
  NOT SCIENTIFICALLY SUPPORTED BY CURRENT DATA
  ```

---

## 4. Architectural Roadmap Based on Audit Evidence

Under strict adherence to the Master Prompt's research integrity guidelines:

1. **Phase 3 & Phase 4 (YOLOv8 Detection)**:
   - Remains completely frozen and protected.
   - Deployed model: `EXP-002_imgsz512` (mAP@0.5 = 0.6862, Precision = 0.9044).
2. **Phase 5 Severity Estimation**:
   - Preserved unchanged: **Relative 2D image-space affected-area proxy** computed via exact sweep-line union area of detected bounding boxes.
   - Extended with a formal mathematically defined Plant Canopy Coverage denominator proxy for future segmentations.
3. **Phase 7 Segmentation**:
   - Architectural and interface contracts (`src/segmentation/`) are established for future polygon/mask annotations.
   - Zero fabricated masks or artificial IoU/Dice metrics reported.
4. **Phase 7 Temporal Progression & Alert Engine**:
   - Document the critical data gap in `research/phase_7_temporal_data_gap.md`.
   - Build a mathematically rigorous, leakage-controlled longitudinal data specification and modular simulation/interface suite (`src/temporal/`) for prospective field trials.
   - Baseline heuristic progression and decision-support alert logic (`src/alerts/`) implemented as pure engineering monitoring tiers.
   - LSTM / GRU deep temporal training: **NOT TRAINED** (honestly reported).
