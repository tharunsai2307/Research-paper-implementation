# Phase 6 Deployment Report: Mobile Application & Serving Infrastructure

## 1. Executive Summary

Phase 6 transitions the trained YOLOv8 coconut tree disease detection model and Phase 5 Plantation Health Monitoring engine into an operational, responsive mobile application. In accordance with strict scientific guidelines, this phase introduces **zero synthetic predictions, zero fabricated latency metrics, and zero modifications to frozen Phase 3, 4, or 5 research artifacts**.

The system pairs a high-performance **FastAPI backend inference service** with a **mobile-optimized progressive web application (PWA)** client. This architecture guarantees:
1. **Mathematical Reproducibility**: Exact reuse of Phase 5 sweep-line union area proxy algorithms and composite Plantation Health Index (PHI) scoring without FP16/INT8 numerical drift.
2. **Deterministic Confidence Dual-Thresholding**: Distinction between confirmed disease detections ($\tau_{\text{op}} = 0.25$) and exploratory candidates ($0.10 \le \text{conf} < 0.25$).
3. **True Empirical Benchmarking**: Latency, memory footprint, and load times are measured directly on the host deployment CPU. Mobile edge hardware was physically unavailable, and this report explicitly records those metrics as `NOT MEASURED` rather than fabricating simulated edge values.

---

## 2. Deployed Model Architecture & Provenance

### 2.1 Model Selection Rationale
Between the two Phase 3 experimental checkpoints, **EXP-002 ($512 \times 512$)** was selected over EXP-001 ($640 \times 640$) based on empirical Phase 4 evaluation:
- **Test Precision**: $0.9044$ (EXP-002) vs $0.8160$ (EXP-001) — a $+8.84\%$ absolute improvement in false positive suppression.
- **mAP@0.5**: $0.6862$ (EXP-002) vs $0.6343$ (EXP-001) — a $+5.19\%$ improvement across disease classes.
- **CPU Inference Latency**: $74.56\text{ ms}$ (EXP-002) vs $110.07\text{ ms}$ (EXP-001) — a $32.3\%$ latency reduction critical for mobile responsiveness.
- **Parameter Count & Size**: Both utilize the lightweight `YOLOv8n` backbone ($3.01\text{M}$ parameters, $5.93\text{ MB}$ uncompressed).

### 2.2 Model Provenance & Checkpoint Verification
- **Source Checkpoint**: `outputs/training/EXP-002_imgsz512/weights/best.pt`
- **Deployment Checkpoint**: `outputs/phase_6/models/best_exp002_512.pt`
- **SHA-256 Hash**: `e3b2b5d39f1e6bce6471e4256ebae24cb912bfa5113d09a7b97c0f1ffcc1a9e9` *(Exact match; zero weight modification)*
- **TorchScript Export Checkpoint**: `outputs/phase_6/models/best_exp002_512.torchscript` ($11.83\text{ MB}$)
- **Input Dimensions**: $512 \times 512 \times 3$ (RGB)
- **Class Ontology (5 Classes)**:
  0: `bud root dropping`
  1: `bud rot`
  2: `gray leaf spot`
  3: `leaf rot`
  4: `stembleeding`

---

## 3. Mobile Client & Backend System Architecture

### 3.1 Service Endpoints (`app/api/main.py`)
- `GET /`: Serves the responsive mobile client application (`app/templates/index.html`).
- `GET /api/v1/health`: Returns API readiness status, active model ID, input dimensions, configured thresholds, and class definitions.
- `POST /api/v1/analyze`: Accepts multipart image upload (`image/jpeg`, `image/png`, max 15MB), performs inference via `HealthInferenceEngine`, executes Phase 5 2D image-space sweep-line union area proxy calculations, attaches single-palm health assessment, and returns bounding boxes, confidence scores, and runtime latency.
- `POST /api/v1/aggregate`: Accepts a batch of inspection records from a scouting walk, invokes Phase 5 `PlantationHealthAggregator`, and generates a multi-palm plantation health summary report.

### 3.2 Mobile User Interface Workflow
The mobile application is built using modern responsive web standards (`viewport-fit=cover`, CSS grid/flexbox, touch-first ergonomics $\ge 48\text{px}$ targets) organized across 4 views:
1. **Screen 1 — Palm Image Acquisition**:
   - Camera direct capture (`capture="environment"`) or gallery photo upload.
   - Live API connectivity status indicator.
   - Visual guidance card detailing optimal palm canopy framing.
2. **Screen 2 — Confirm & Pre-Analysis Preview**:
   - High-fidelity preview of the selected image.
   - Tree/Block metadata entry (Plantation Block ID, optional geolocation coordinates).
   - "Analyze Coconut Palm" execution trigger with non-fake progress animation.
3. **Screen 3 — Detection Canvas & Tree Health Dashboard**:
   - High-DPI dynamic HTML5 canvas overlaying bounding boxes and disease tags onto the original photo.
   - Visual distinction: solid boxes for confirmed lesions ($\text{conf} \ge 0.25$), dashed boxes for low-confidence candidates ($0.10 \le \text{conf} < 0.25$).
   - Tree Health Index (PHI) radial gauge and assigned engineering tier.
   - Relative 2D Image-Space Affected Area Proxy percentage and raw pixel metrics.
   - Prominent, non-dismissible Engineering Health Tier Disclaimer.
4. **Screen 4 — Plantation Scouting Walk Aggregate Report**:
   - Multi-palm session accumulator tracking diseased vs healthy palms.
   - Disease prevalence percentage across scouted trees.
   - Class-by-class lesion breakdown and unique tree infection counts.
   - Comprehensive plantation health tier and action recommendations.
   - Client-side JSON export of the complete scouting session audit trail.

---

## 4. Empirical Deployment Benchmarks

Benchmark measurements were conducted on the host CPU deployment environment using 20 timed evaluation trials on validation imagery (`LeafRot007.jpg`, $768 \times 1024$ native resolution).

### 4.1 Measured Runtime Performance Table
| Component | Metric | Measured Value | Unit | Status |
|---|---|---|---|---|
| **Model Size** | Disk Footprint | 5.93 | MB | MEASURED |
| **Model Load Time** | CPU Initialization | 37.71 | ms | MEASURED |
| **Process Memory (Initial)** | RSS | 250.85 | MB | MEASURED |
| **Process Memory (Post-Load)**| RSS | 269.14 | MB | MEASURED |
| **Process Memory (Peak)** | RSS | 410.97 | MB | MEASURED |
| **Preprocessing Latency** | Mean (std) | 10.38 (±0.61) | ms | MEASURED |
| **Model Inference Latency** | Mean (std) | 77.06 (±9.47) | ms | MEASURED |
| **Postprocessing & Area Proxy**| Mean (std) | 9.75 (±4.94) | ms | MEASURED |
| **End-to-End Latency** | Mean (std) | 172.61 (±12.1) | ms | MEASURED |
| **Host CPU Throughput** | Equivalent FPS | ~5.8 | FPS | MEASURED (CPU Batch-1) |

### 4.2 Target Mobile Edge Hardware Disclaimers
In accordance with Section 30 of the research specifications:
- **On-Device Mobile Edge Latency**: `NOT MEASURED` — Target smartphone hardware (e.g., Apple A-series Bionic Neural Engine, Qualcomm Snapdragon NPU) was physically unavailable in the execution environment.
- **On-Device Mobile Edge FPS**: `NOT MEASURED` — Cannot be claimed without physical profiling on target mobile silicon.
- **On-Device Power / Thermal Dissipation**: `NOT MEASURED` — Hardware power analyzers were not connected.

---

## 5. Methodological & Research Disclaimers

1. **Relative 2D Image-Space Affected Area Proxy**:
   All area metrics generated by the mobile application represent the 2D bounding box union footprint projected onto the camera imaging plane. They do not constitute 3D volumetric foliage loss or certified agronomic severity staging.
2. **Engineering Health Tiers**:
   Health tiers (EXCELLENT, GOOD, MODERATE, POOR) are engineering monitoring thresholds derived from the heuristic weighting formula:
   $$\text{PHI}_{\text{composite}} = 100 \times \left(1.0 - \left(0.85 \times \frac{N_{\text{pos}}}{N_{\text{total}}} + 0.15 \times \text{Mean Area Proxy}\right)\right)$$
   They do not represent certified clinical pathology diagnoses or legally binding agronomic treatment mandates.
3. **Prevalence Interpretation**:
   Prevalence statistics displayed in the scouting summary reflect only the photos captured by the scout during that specific walk, not a scientifically stratified regional prevalence study.
