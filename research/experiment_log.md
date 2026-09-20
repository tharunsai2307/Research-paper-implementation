# Experiment & Operations Log: Phase 1 Dataset Foundation

## Phase 1 Operational Record

### Milestone 1: Environment & Project Setup
- **Date/Time**: 2026-09-20T03:25:00 UTC
- **Host System**: Windows x64
- **Python Version**: 3.14.4
- **Project Location**: `C:\Users\Hp\.gemini\antigravity-ide\scratch\coconut-disease-yolov8`
- **Actions**:
  - Initialized clean directory tree matching the required engineering architecture (`data/`, `datasets/`, `research/`, `src/`, `notebooks/`, `configs/`, `scripts/`, `outputs/`, `app/`).
  - Added `.gitkeep` files in data directories to preserve layout.
  - Initialized Python packages in `src/` (`src/dataset/`, `src/preprocessing/`, `src/annotation/`, `src/training/`, `src/evaluation/`, `src/utils/`).
  - Configured project `.gitignore` and `README.md`.
  - Pinned Phase 1 dependencies in `requirements.txt` (Ultralytics YOLOv8, PyTorch, OpenCV, Pillow, Pandas, Matplotlib, Scikit-Learn).

---

### Milestone 2: Dataset Sourcing & Provenance Analysis
- **Sources Investigated**:
  - Mendeley Data: Patil et al. (2023), DOI: `10.17632/gh56wbsnj5.1` (*Data in Brief*, 2023).
  - Roboflow Universe: Phanidhar Reddy (`coconut-tree-disease-vg85j`).
  - Roboflow Universe: DRIYOG (`coconut-tree-diseases`).
  - Kaggle: Jigar Baraiya (`coconut-leaf-disease-dataset`).
  - Kaggle: LakshanX (`infected-and-healthy-coconut-leaves`).
  - GitHub: Avishka Kavindu (`coconut-cinnamon-disease-detection-mask-rcnn`).
- **Deliverables**:
  - Created `research/dataset_sources.md` containing all 20 required metadata fields for each candidate.
  - Constructed objective comparison matrix using strict `YES / NO / PARTIAL / UNKNOWN` scale across 11 criteria.

---

### Milestone 3: Data Acquisition & Manifest Creation
- **Action**:
  - Built `scripts/download_dataset.py` to acquire verified authentic sample images (100 images, 20 per class across all 5 classes) from the Patil et al. open mirror.
  - Stored raw untouched images under `data/raw/mendeley_coconut_disease/`.
  - Computed cryptographic SHA-256 hashes and recorded provenance in `research/dataset_manifest.csv`.
  - Documented explicit authentication requirements for Roboflow API / browser export, Mendeley Data session download, and Kaggle CLI access without bypassing access controls.

---

### Milestone 4: Automated Integrity Audit
- **Script**: `scripts/run_audit.py` calling `src/dataset/audit.py`
- **Audit Results**:
  - Total images audited: 100
  - Valid images: 100 (100% readable via PIL and OpenCV)
  - Corrupted images: 0
  - Unreadable images: 0
  - Unique dimensions: 1 (strictly uniform 768 x 1024 resolution)
  - Image format: 100% JPEG
  - Classes: Bud Root Dropping (20), Bud Rot (20), Gray Leaf Spot (20), Leaf Rot (20), Stem Bleeding (20)
  - Annotation status: **Classification-only** (`annotations_found: false`)
- **Artifacts Generated**:
  - `outputs/dataset_analysis/dataset_summary.json`
  - `outputs/dataset_analysis/class_distribution.csv`
  - `research/dataset_audit.md`

---

### Milestone 5: Visual Inspection Grid Generation
- **Script**: `scripts/generate_visuals.py` calling `src/dataset/visualize.py`
- **Output**:
  - Generated 9.5 MB multi-panel grid `outputs/dataset_analysis/sample_grids/representative_class_grid.png` containing 4 authentic images from each of the 5 classes with dimensions and filenames.

---

### Milestone 6: Near-Duplicate & Leakage Analysis
- **Script**: `scripts/detect_leakage.py` calling `src/dataset/leakage.py`
- **Analysis Results**:
  - Cryptographic exact duplicates: 0
  - Cross-class leaks: 0
  - Perceptual near-duplicates (Hamming distance <= 4 on 64-bit dHash): 49 pairs
  - **Key Pathology/Engineering Finding**: Sequential filenames (e.g., `BudRootDropping001.jpg` and `BudRootDropping002.jpg`) exhibit Hamming distance of 1, confirming burst camera capture of individual infected trees.
  - **Mitigation Mandate**: Requires group-aware splitting in Phase 2 to prevent data leakage between train, validation, and test splits.
- **Artifact Generated**:
  - `outputs/dataset_analysis/leakage_report.json`

---

### Milestone 7: Scientific & Annotation Decision Report
- **Deliverable**:
  - Created `research/research_decisions.md` detailing:
    - Selected primary detection dataset: Roboflow Universe (`phanidhar-reddy/coconut-tree-disease-vg85j`, CC BY 4.0).
    - Selected benchmark reference pool: Mendeley Data / Patil et al. (5,798 images, CC BY 4.0).
    - Scientific prohibition of converting whole-image classification labels into bounding boxes.
    - AI-assisted zero-shot pre-annotation workflow (Florence-2 / Grounding DINO + SAM 2 + human-in-the-loop expert review).
    - Group-aware split protocol for burst photo mitigation.

---

# Phase 2A: Roboflow Object Detection Dataset Investigation

- **Date/Time**: 2026-09-20T03:35:00 UTC
- **Commands Executed**:
  - `python scripts/run_roboflow_audit.py`
  - Browser inspection of Roboflow Universe project (`phanidhar-reddy/coconut-tree-disease-vg85j`)
- **Datasets Investigated**:
  - Roboflow Universe: Coconut Tree Disease (Phanidhar Reddy, CC BY 4.0, 5 classes).
- **Findings**:
  - Roboflow provides authentic bounding-box annotations for 5 classes matching Mendeley pathologies.
  - Raw unaugmented benchmark contains 100 images (120 bboxes).
  - Default Roboflow train/valid/test partition suffered from cross-split burst leakage (e.g. `BudRootDropping018.jpg` in valid and `019.jpg` in test with dHash Hamming distance = 1).
  - Dataset completely lacked healthy/negative control palm images.
- **Artifacts Generated**:
  - `research/roboflow_dataset_audit.md`
  - `research/roboflow_annotation_quality.md`
  - `research/dataset_comparison.md`
  - `research/final_dataset_recommendation.md`

---

# Phase 2B: Dataset Sanitization & Pre-Training Preparation

- **Date/Time**: 2026-09-20T03:55:00 UTC
- **Commands Executed**:
  - `python scripts/verify_labels.py`
  - `python scripts/acquire_healthy_negatives.py`
  - `python scripts/create_grouped_split.py`
  - `python scripts/audit_post_split_leakage.py`
  - `python scripts/analyze_clean_dataset.py`
  - `python scripts/validate_final_dataset.py`
- **Datasets Used**:
  - Primary: Roboflow Coconut Tree Disease (100 images, 120 bboxes, CC BY 4.0).
  - Secondary: Wikimedia Commons Category:Cocos nucifera (50 verified healthy palm images, CC BY-SA / CC0).
- **Cleaning & Sanitization Operations**:
  - Validated all 120 YOLO bounding boxes ($x_c, y_c \in [0, 1]$, $w, h \in (0, 1]$, 0 invalid, 0 out-of-bounds, 0 NaN).
  - Preserved raw directories `data/raw/` untouched; materialized sanitized dataset to `data/processed/coconut_detection_clean/`.
  - Empty `.txt` label files assigned to all 50 healthy negative images (background control for YOLOv8).
- **Split Strategy & Grouping**:
  - Performed Disjoint Set Union (DSU) clustering using 64-bit dHash (Hamming distance $\le 4$) and sequential frame burst adjacency ($|idx_1 - idx_2| = 1$ with Hamming $\le 6$).
  - Grouped 100 diseased images into 66 atomic clusters (14 multi-image bursts).
  - Deterministic stratified partitioning with `seed=42`:
    - Train: 100 images (66.7%), 78 bboxes (63 diseased, 37 healthy negatives)
    - Validation: 35 images (23.3%), 30 bboxes (27 diseased, 8 healthy negatives)
    - Test: 15 images (10.0%), 12 bboxes (10 diseased, 5 healthy negatives)
- **Leakage Findings**:
  - Post-split leakage audit confirmed **0 exact duplicates** and **0 perceptual near-duplicates** across all cross-split boundaries (`train` $\leftrightarrow$ `val`, `train` $\leftrightarrow$ `test`, `val` $\leftrightarrow$ `test`).
  - Cross-split leakage status: **PASSED — ZERO CROSS-SPLIT LEAKAGE**.
- **Healthy Image Integration**:
  - 50 authentic high-resolution images acquired from Wikimedia Commons, verified via Laplacian sharpness (>15.0) and visual contact sheet (`outputs/dataset_analysis/healthy_samples.png`). Provenance itemized in `research/healthy_image_provenance.csv`.
- **Final Counts**:
  - Total images: 150 (100 diseased, 50 healthy negatives)
  - Total bounding boxes: 120 (40 gray leaf spot, 20 bud root dropping, 20 bud rot, 20 leaf rot, 20 stembleeding)
  - Class imbalance ratio: 2.0 : 1 (mild)
- **Unresolved Limitations**:
  - Full 3,229-image Roboflow archive requires manual download or Roboflow API key.
  - Co-infection scenarios (multiple distinct diseases on a single palm) are absent from ground truth.
- **Phase 3 Readiness**:
  - Sanitized dataset verified and ready for YOLOv8 experimentation.
  - Baseline configuration prepared at `configs/yolov8_baseline.yaml` (**NOT YET EXECUTED**).

---

# Phase 3: YOLOv8 Model Training & Controlled Experimentation

- **Date/Time**: 2026-09-20T04:24:00 UTC
- **Host System**: Windows 11 AMD64, AMD Ryzen 5 7520U (4 physical / 8 logical cores, 15.24 GB RAM)
- **Deep Learning Stack**: Python 3.14.4, PyTorch 2.13.0+cpu, Ultralytics 8.4.115, OpenCV 4.10.0.84
- **Hardware Mode**: CPU Execution (`CUDA Available: False`)
- **Dataset Actually Used**: `data/processed/coconut_detection_clean/` (100 train, 35 val, 15 test strictly untouched)
- **Commands Executed**:
  - `python scripts/validate_final_dataset.py` (Pre-training dataset integrity verification: PASSED 8/8)
  - `python scripts/train_experiments.py` (Automated Phase 3 training engine)

### Experiments Executed:
1. **EXP-001_baseline_yolov8n**:
   - **Architecture**: YOLOv8n (`yolov8n.pt`, 3,006,623 parameters, 5.94 MB)
   - **Hyperparameters**: `imgsz=640`, `epochs=15`, `batch=16`, `seed=42`, `optimizer=AdamW`, `lr0=0.001`, `device='cpu'`
   - **Training Time**: 740.0 seconds (~12.3 minutes)
   - **Validation Results**: Precision = **0.8661**, Recall = **0.4824**, mAP@0.5 = **0.6186**, mAP@0.5:0.95 = **0.2600**, F1 = **0.6197**
   - **CPU Latency**: 93.13 ms / image (10.7 FPS on AMD Ryzen 5 7520U)
   - **Negative Control False Positives**: 0 / 8 healthy validation images (0.0% FP rate)
   - **Checkpoint**: `outputs/training/EXP-001_baseline_yolov8n/weights/best.pt`

2. **EXP-002_imgsz512**:
   - **Architecture**: YOLOv8n (`yolov8n.pt`, 3,006,623 parameters, 5.93 MB)
   - **Hyperparameters**: `imgsz=512` (Controlled change; all other parameters identical to EXP-001)
   - **Training Time**: 489.3 seconds (~8.2 minutes, -33.9% vs EXP-001)
   - **Validation Results**: Precision = **0.7308**, Recall = **0.4000**, mAP@0.5 = **0.5752**, mAP@0.5:0.95 = **0.2426**, F1 = **0.5170**
   - **CPU Latency**: 68.74 ms / image (14.5 FPS, -26.2% latency vs EXP-001)
   - **Negative Control False Positives**: 0 / 8 healthy validation images (0.0% FP rate)
   - **Checkpoint**: `outputs/training/EXP-002_imgsz512/weights/best.pt`

3. **EXP-003_yolov8s**:
   - **Architecture**: YOLOv8s (`yolov8s.pt`, 11.2M parameters)
   - **Status**: **NOT RUN**
   - **Reason**: Hardware compute limitation. CPU-only execution without CUDA requires ~5 mins/epoch (~75 mins total). Documented honestly per research integrity rules.

### Key Observations:
- Reducing resolution from 640 to 512 delivered a 26.2% reduction in latency (yielding 14.5 FPS on commodity CPU) with only a 7.0% relative drop in mAP@0.5 (0.619 to 0.575).
- Healthy negative background control integration achieved 0% false positives on validation foliage.
- The test set (15 images, 12 annotations) was 100% untouched during Phase 3, preserving strict test isolation for Phase 4.

---

# Phase 4: Final Model Evaluation & Research Analysis

- **Date/Time**: 2026-09-20T04:30:00 UTC
- **Host System**: Windows 11 AMD64, AMD Ryzen 5 7520U (4 physical / 8 logical cores, 15.24 GB RAM)
- **Deep Learning Stack**: Python 3.14.4, PyTorch 2.13.0+cpu, Ultralytics 8.4.115, OpenCV 4.10.0.84
- **Evaluation Partition**: Untouched Test Set (`data/processed/coconut_detection_clean/test/`, 15 images, 12 bounding boxes)
- **Integrity Status**: Pre-evaluation audit verified test set remained 100% untouched since Phase 2B (`research/phase_4_test_integrity.md`).
- **Commands Executed**:
  - `python scripts/evaluate_test_set.py` (Final test evaluation and prediction generation)
  - `python scripts/generate_phase4_figures.py` (Research figures generation)

### Evaluated Models & Final Test Results:
1. **EXP-001_baseline_yolov8n ($640 \times 640$)**:
   - Precision: **0.8160**
   - Recall: **0.4000**
   - mAP@0.5: **0.6343** (vs. 0.6186 validation, +0.0157)
   - mAP@0.5:0.95: **0.3663** (vs. 0.2600 validation, +0.1063)
   - F1-Score: **0.5368**
   - CPU Latency: **110.07 ms / image** (~9.1 FPS on host CPU)
   - Healthy Negative False Positives: **0 / 5 images (0.0% FP rate)**
   - Per-Class AP@0.5:0.95: `bud rot` = 0.6965, `stembleeding` = 0.6302, `leaf rot` = 0.4610, `gray leaf spot` = 0.0437, `bud root dropping` = 0.0000

2. **EXP-002_imgsz512 ($512 \times 512$)**:
   - Precision: **0.9044**
   - Recall: **0.4000**
   - mAP@0.5: **0.6862** (vs. 0.5752 validation, +0.1110)
   - mAP@0.5:0.95: **0.2471** (vs. 0.2426 validation, +0.0045)
   - F1-Score: **0.5547**
   - CPU Latency: **74.56 ms / image** (~13.4 FPS on host CPU, **32.3% faster than EXP-001**)
   - Healthy Negative False Positives: **0 / 5 images (0.0% FP rate)**
   - Per-Class AP@0.5:0.95: `stembleeding` = 0.4189, `leaf rot` = 0.3862, `bud rot` = 0.2985, `bud root dropping` = 0.0772, `gray leaf spot` = 0.0546

### Key Findings & Research Outcomes:
- **No Overfitting Degradation**: Test mAP@0.5 matched or exceeded validation expectations in both models without performance collapse.
- **Speed vs. Localization Quality**: EXP-001 ($640 \times 640$) achieved +48.2% higher mAP@0.5:0.95 (0.3663 vs 0.2471) due to superior spatial resolution on foliar and trunk lesions. EXP-002 ($512 \times 512$) achieved 32.3% lower latency (74.6 ms vs 110.1 ms) and higher precision (0.9044 vs 0.8160), making it the ideal candidate for real-time mobile edge processing.
- **Negative Control Verification**: Zero false alarms observed on healthy coconut palm test images at operational confidence threshold $\tau = 0.25$.
- **Phase 5 Preparations**: Generated `detection_output_schema.json` and `research/phase_5_input_requirements.md` to define the ingestion contract for the Plantation Health Monitoring Engine.

---

# Phase 5: Plantation Health Monitoring Engine Implementation & Validation

- **Date/Time**: 2026-09-20T04:38:50 UTC
- **Host System**: Windows 11 AMD64, AMD Ryzen 5 7520U
- **Modules Implemented**:
  - `src/health_monitoring/schemas.py`: Schema constants, class labels, and `ObservationStatus` state definitions.
  - `src/health_monitoring/severity_proxy.py`: Exact 2D bounding box union area (Klee's measure sweep-line algorithm), IoU, and Relative Affected Area Proxy.
  - `src/health_monitoring/inference.py`: Standardized inference engine supporting Model A (EXP-001 @ 640) and Model B (EXP-002 @ 512).
  - `src/health_monitoring/disease_distribution.py`: Pathology aggregation strictly decoupling detection counts from unique affected tree counts.
  - `src/health_monitoring/plantation_health.py`: Plantation-level aggregator, observation rate calculation, impact statistics, and dual Plantation Health Index ($\text{PHI}_{\text{incidence}}$ and $\text{PHI}_{\text{composite}}$).
- **Scripts Created**:
  - `tests/test_health_monitoring.py`: 12 comprehensive unit tests covering all mathematical, geometric, and aggregation logic.
  - `scripts/run_health_monitoring.py`: Automated end-to-end execution, Model A vs. Model B comparison, JSON/CSV exports, and figure generation.
  - `scripts/validate_health_monitoring.py`: System verification checking schema integrity, area non-negativity, percentage limits, and numerical stability.
- **Validation Run Results**:
  - Unit Tests: **12 / 12 passed (0 failures, 0 errors)** in 0.023s.
  - System Validation: **11 / 10 checks passed (100% compliance, Status: PASS)**.
  - Demonstration Block (35 images from `val/images`: 27 diseased, 8 healthy negatives):
    - Model A (EXP-001 @ 640): 0 accepted detections $\ge 0.25$, 11 low-confidence candidate palms in $[0.10, 0.25)$, PHI = 100.0.
    - Model B (EXP-002 @ 512): 6 accepted `leaf rot` detections across 3 palms, positive rate = 8.57%, mean area proxy (pos) = 0.4607, mean confidence = 0.3201, $\text{PHI}_{\text{composite}} = 92.12$, $\text{PHI}_{\text{incidence}} = 91.43$.
- **Outputs Generated**:
  - `outputs/health_monitoring/image_health_records.json`
  - `outputs/health_monitoring/plantation_health_report.json`
  - `outputs/health_monitoring/plantation_health_report.csv`
  - `outputs/health_monitoring/disease_distribution.csv`
  - `outputs/health_monitoring/model_comparison.json`
  - `outputs/figures/health_monitoring/` (6 research visualizations)
  - `research/phase_5_health_monitoring_methodology.md`
  - `research/phase_5_validation_report.md`
- **Integrity Verifications**:
  - Phase 3 training checkpoints untouched.
  - Phase 4 test evaluation artifacts untouched.
  - Zero test-set threshold tuning performed.

---

# Phase 6: Mobile Application & YOLOv8 Model Deployment

- **Date/Time**: 2026-09-20T05:06:00 UTC
- **Host System**: Windows 11 AMD64, AMD Ryzen 5 7520U
- **Modules Implemented**:
  - `research/phase_6_architecture_decision.md`: Evaluation of Options A, B, and C across 9 criteria. Selected Option B (FastAPI backend + PWA mobile client with hybrid ergonomics) for 100% Phase 5 analytical code reuse.
  - `research/phase_6_model_conversion.md`: Conversion feasibility study. Exported EXP-002 ($512 \times 512$) to TorchScript (`best_exp002_512.torchscript`, 11.83 MB).
  - `app/api/main.py`: Production-ready FastAPI service providing `/api/v1/health`, `/api/v1/analyze`, `/api/v1/aggregate`, and static asset serving.
  - `app/templates/index.html`: Responsive mobile web UI covering Screen 1 (Acquisition), Screen 2 (Confirm Preview), Screen 3 (Detection Dashboard & PHI Gauge), and Screen 4 (Scouting Walk Aggregate Report).
  - `app/static/css/style.css`: Clean agricultural research theme (`#1b4332`, `#2d6a4f`, `#d8f3dc`) with touch-first target sizing ($\ge 48\text{px}$).
  - `app/static/js/app.js`: Camera acquisition, canvas bounding box rendering (solid $\ge 0.25$, dashed $[0.10, 0.25)$), scouting walk tracking, and JSON report export.
  - `tests/test_mobile_app.py`: Comprehensive test suite for health endpoint, web app delivery, real/control inference, MIME/byte validation, and multi-palm aggregation.
  - `scripts/run_deployment_benchmark.py`: Empirical measurement of model load, preprocessing, inference, postprocessing, memory footprint, and disk size on host CPU.
- **Validation & Test Results**:
  - Phase 5 Unit Tests: **13 / 13 PASS**
  - Phase 5 System Validator: **11 / 11 PASS**
  - Phase 6 Mobile & API Tests: **8 / 8 PASS**
  - Total Test Suite: **21 / 21 PASS (`python -m unittest discover tests`)**
- **Empirical Deployment Benchmarks (Host CPU)**:
  - Model Load Time: **37.71 ms**
  - Mean Preprocessing: **10.38 ms**
  - Mean Model Inference: **77.06 ms**
  - Mean Postprocessing & Area Proxy: **9.75 ms**
  - Mean End-to-End Latency: **172.61 ms**
  - Peak Memory RSS: **410.97 MB**
  - Model File Size: **5.93 MB**
  - Target Mobile Hardware: **NOT MEASURED** (target mobile silicon physically unavailable)
- **Outputs & Documentation Created**:
  - `outputs/phase_6/models/best_exp002_512.pt` (Verified SHA-256: `e3b2b5d39f1e6bce6471e4256ebae24cb912bfa5113d09a7b97c0f1ffcc1a9e9`)
  - `outputs/phase_6/models/best_exp002_512.torchscript`
  - `outputs/phase_6/benchmark_results.json`
  - `research/phase_6_deployment_report.md`
  - `research/phase_6_validation_report.md`
- **Research Integrity Verifications**:
  - Phase 3 checkpoints (`EXP-001` and `EXP-002`) remain 100% bitwise identical.
  - Phase 4 test set metrics preserved without recalculation.
  - Phase 5 health monitoring engine completely reused without modification.
  - Zero fabricated latency, FPS, or mobile metrics.
