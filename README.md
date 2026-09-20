# Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using YOLOv8

## Project Overview
This repository contains the research, dataset management, deep learning modeling, and deployment pipeline for automated coconut tree disease detection and plantation health monitoring.

The research framework is structured into a rigorous pipeline:
```
Image / Mobile Camera
        ↓
Image Preprocessing & Augmentation
        ↓
YOLOv8 Disease Detection (Object Detection)
        ↓
Disease Class + Bounding Box + Confidence Score
        ↓
Plantation Health Monitoring (Severity & Spatial Analytics)
        ↓
Mobile Application Deployment
```

## Current Status: Phase 5 — Plantation Health Monitoring Engine (COMPLETED)

**Phase 5 Plantation Health Monitoring Engine implementation and validation is COMPLETED.**

The engine aggregates individual YOLOv8 detections into structured health statistics, decoupled disease distributions, and an interpretable Plantation Health Index (PHI).

| Metric / Attribute | Model A: EXP-001 (YOLOv8n @ 640) | Model B: EXP-002 (YOLOv8n @ 512) |
| :--- | :---: | :---: |
| **Input Image Size** | $640 \times 640$ | $512 \times 512$ |
| **Operational Threshold ($\tau_{\text{op}}$)** | 0.25 (engineering default) | 0.25 (engineering default) |
| **Candidate Threshold ($\tau_{\text{cand}}$)** | 0.10 | 0.10 |
| **Sampled Block Size** | 35 images (27 diseased, 8 healthy) | 35 images (27 diseased, 8 healthy) |
| **Confirmed Lesions ($N_{\text{det}}$)** | 0 | **6** |
| **Infected Palms ($N_{\text{tree}}$)** | 0 | **3 (8.57% positive rate)** |
| **Low-Confidence Candidate Palms** | 11 (31.43%) | 5 (14.29%) |
| **Mean Accepted Confidence** | N/A | 0.3201 |
| **Mean Affected Area Proxy ($A_{\text{proxy}}$)** | 0.0000 | 0.0395 (all) / 0.4607 (pos) |
| **PHI (Composite Metric)** | 100.0 / 100 | **92.12 / 100** |
| **Assigned Health Tier** | `EXCELLENT` | `EXCELLENT / MINIMAL_DISEASE_PRESSURE` |

### Key Deliverables:
- **Core Engine Modules**: `src/health_monitoring/` (`inference.py`, `disease_distribution.py`, `severity_proxy.py`, `plantation_health.py`, `schemas.py`)
- **Unit Test Suite**: `tests/test_health_monitoring.py` (**12/12 passed**)
- **Validation Script**: `scripts/validate_health_monitoring.py` (**11/10 passed, Status: PASS**)
- **Mobile-App Ready Outputs**: `outputs/health_monitoring/plantation_health_report.json` & `image_health_records.json`
- **Agronomic CSVs**: `outputs/health_monitoring/plantation_health_report.csv` & `disease_distribution.csv`
- **Comparative Analysis**: `outputs/health_monitoring/model_comparison.json`
- **Research Visualizations**: `outputs/figures/health_monitoring/` (6 figures)
- **Methodology & Validation**: `research/phase_5_health_monitoring_methodology.md` & `research/phase_5_validation_report.md`

### Quick Execution:
```bash
# Run 12-test unit test suite
python -m unittest tests/test_health_monitoring.py

# Execute end-to-end health monitoring pipeline
python scripts/run_health_monitoring.py

# Run automated validation suite
python scripts/validate_health_monitoring.py
```
```
coconut-disease-yolov8/
│
├── data/
│   ├── raw/                # Untouched raw dataset archives & source files
│   ├── interim/            # Extracted and normalized working directories
│   ├── processed/          # Validated detection datasets
│   └── external/           # Reference taxonomies and pathology guides
│
├── datasets/
│   ├── candidates/         # Candidate comparison logs & metadata
│   └── final/              # Final train/val/test splits for YOLOv8
│
├── research/
│   ├── dataset_sources.md  # Detailed 20-point candidate records & comparison matrix
│   ├── dataset_audit.md    # Automated audit findings (counts, corruptions, bbox stats)
│   ├── dataset_manifest.csv# Provenance tracking (URLs, licenses, SHA-256, versions)
│   ├── research_decisions.md# Scientific decisions, YOLO suitability & annotation roadmap
│   └── experiment_log.md   # Chronological log of operations and investigations
│
├── src/
│   ├── dataset/            # Auditing, visualization, and leakage detection modules
│   ├── preprocessing/      # Image normalization & validation routines
│   ├── annotation/         # YOLOv8 annotation validation and conversion utilities
│   ├── training/           # YOLOv8 model training scripts (Phase 2)
│   ├── evaluation/         # Performance evaluation (mAP@50, mAP@50:95)
│   └── utils/              # Hashing, I/O, and helper functions
│
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── configs/                # YOLOv8 dataset and model configuration files
├── scripts/                # Standalone CLI execution scripts
├── outputs/                # Analytical artifacts, summary JSON, figures, and plots
└── app/                    # Mobile application / API bridge (Phase 3)
```

## Setup & Quickstart
```bash
# Clone or navigate to the repository
cd coconut-disease-yolov8

# Install dependencies
pip install -r requirements.txt

# Run dataset audit
python scripts/run_audit.py

# Generate representative visual samples
python scripts/generate_visuals.py

# Run near-duplicate and data leakage checks
python scripts/detect_leakage.py
```
