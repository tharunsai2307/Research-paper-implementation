# Phase 6 — Deployment Architecture Decision & Trade-Off Analysis

**Project Title:** “A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications”  
**Project Root:** `C:\Users\Hp\.gemini\antigravity-ide\scratch\coconut-disease-yolov8`  
**Phase:** Phase 6 — Mobile Application & Model Deployment  
**Date:** 2026-09-20  

---

## 1. Objective

To determine, justify, and formally document the deployment architecture for serving the trained YOLOv8 coconut tree disease detection model and the Phase 5 Plantation Health Monitoring Engine to mobile field practitioners (agronomists, plantation managers, and farmers).

---

## 2. Evaluation of Architectural Alternatives

We evaluate three potential architectural designs:
* **Option A: On-Device Edge Inference** (Mobile Client $\rightarrow$ Local Edge Model $\rightarrow$ Client Health Engine)
* **Option B: Backend REST API Inference** (Mobile Client $\rightarrow$ HTTP/REST API $\rightarrow$ Python YOLOv8 Model $\rightarrow$ Python Health Engine $\rightarrow$ Mobile UI)
* **Option C: Hybrid Architecture** (Mobile Client Preprocessing/Cache $\rightarrow$ Local/Cloud Edge API $\rightarrow$ Central Synchronization)

### Multi-Criteria Decision Matrix

| Evaluation Criterion | Option A: On-Device | Option B: Backend REST API | Option C: Hybrid Architecture |
| :--- | :---: | :---: | :---: |
| **1. Existing Model Compatibility** | Moderate (requires TFLite/ONNX conversion) | **High** (native PyTorch/Ultralytics engine) | **High** (native PyTorch/Ultralytics engine) |
| **2. Research Reproducibility** | Low (quantization drift, NMS differences) | **Very High** (exact Phase 5 Python engine) | **Very High** (identical analytical engine) |
| **3. Implementation Complexity** | Very High (C++/Java/Swift/Dart bindings) | **Moderate** (clean FastAPI + Mobile Client) | High (dual client/server state sync) |
| **4. Inference Latency (Local)** | 70–250 ms (device dependent) | 75 ms inference + network RTT | 75 ms inference + network RTT |
| **5. Offline Capability** | **Full** (no network needed) | Requires local Wi-Fi or cellular connection | Local queuing with batch sync |
| **6. Mobile Hardware Constraints** | Severe (low-end Android 2GB RAM risk) | **Minimal** (runs smoothly on any browser/device)| Minimal on client |
| **7. Health Engine Portability** | Low (2D sweep-line union must be ported) | **High** (direct reuse of `src/health_monitoring`)| High (direct reuse) |
| **8. Maintainability & Updates** | Low (requires app store re-certification) | **Very High** (instant backend model updates) | High |
| **9. Privacy & Data Governance** | High (data remains on device) | Moderate (secure TLS transport required) | Moderate |

---

## 3. Deep-Dive Rationale

### 3.1 Research Reproducibility & Algorithmic Integrity
The Phase 5 Plantation Health Monitoring Engine relies on:
1. Exact analytical 2D bounding-box union area via Klee's measure sweep-line algorithm to eliminate double counting.
2. Distinct decoupling between total lesion count ($N_{\text{det}}$) and unique infected palms ($N_{\text{tree}}$).
3. The dual Plantation Health Index ($\text{PHI}_{\text{incidence}}$ and $\text{PHI}_{\text{composite}}$ with weights $w_1=0.85, w_2=0.15$).

Porting this analytical framework to client-side JavaScript or native mobile languages (Dart/Kotlin/Swift) creates a significant risk of **algorithmic drift**, floating-point rounding mismatches, and maintenance divergence. In contrast, **Option B / Hybrid** directly executes the frozen, validated Phase 5 Python modules (`src/health_monitoring/`), guaranteeing 100% research fidelity.

### 3.2 Mobile Hardware Realities in Coconut Farming
Coconut palms are cultivated primarily in tropical rural regions (e.g., South Asia, Southeast Asia, Pacific Islands). Agricultural extension workers and smallholder farmers frequently utilize budget or mid-range Android smartphones with restricted memory (2GB–3GB RAM) and entry-level SoCs. Running multi-megabyte PyTorch or unquantized edge models directly on-device can induce:
* Out-Of-Memory (OOM) operating system kills.
* Severe thermal throttling during repeated scans.
* Heavy battery drain during field surveys.

A lightweight, responsive mobile frontend communicating with a local edge server (e.g. farm laptop, Raspberry Pi local Wi-Fi hotspot) or agricultural cloud server ensures uniform 75 ms inference regardless of whether the mobile handset costs $100 or $1,000.

---

## 4. Final Architecture Selection: Option B with Hybrid Mobile Capabilities

We select **Option B (Backend API with Mobile Client)** enhanced with client-side mobile ergonomics:

```text
[ Mobile Handset / Browser Client ]
  │
  ├── Camera Capture (capture="environment") / Gallery Picker
  ├── Client-side EXIF Orientation & Aspect-Ratio Preservation
  ├── Real-time UI State Management ("Analyzing coconut tree...")
  ├── Interactive Canvas Bounding Box & Label Rendering
  │
  ▼ HTTP Multipart / REST POST (/api/v1/analyze)
[ FastAPI Backend Application (app/api/) ]
  │
  ├── Request Validation & Image Decoding (OpenCV)
  ├── Model Forward Pass (EXP-002 YOLOv8n @ 512x512)
  ├── Confidence Filter (Operational cutoff = 0.25, Candidate cutoff = 0.10)
  ├── Phase 5 Health Engine (src/health_monitoring/)
  │     ├── Bounding Box Sweep-Line Union Area
  │     ├── Relative Affected Area Proxy
  │     ├── Disease Distribution
  │     └── Plantation Health Index (PHI)
  │
  ▼ JSON Response (Standardized Phase 5 Schema)
[ Mobile Dashboard View ]
  ├── Screen 1: Home & Image Selector
  ├── Screen 2: Real-time Inspection & Status
  ├── Screen 3: Annotated Detections & Health Analytics Card
  └── Plantation Aggregation Panel (Multiple Palm Scouting Session)
```

---

## 5. Security, Privacy & Data Governance

1. **Transient Processing**: Uploaded images are held in memory buffers for OpenCV decoding and inference; images are **not** permanently written to public cloud storage or logged with personal identifying information.
2. **Zero Sensor Fabrication**: GPS metadata is captured only if voluntarily provided by the handset geolocation API; otherwise it remains strictly `null`.
3. **Transparent Disclaimer**: The UI explicitly displays the mandatory engineering disclaimer that visual affected area proxies and PHI scores represent image-space visual indicators, not certified agronomic or clinical diagnosis.
