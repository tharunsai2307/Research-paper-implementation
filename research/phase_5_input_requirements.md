# Phase 5 Input Requirements: Bridging Detection to Plantation Health Monitoring

**Project Title**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Component**: Specification for Phase 5 (Plantation Health Monitoring Engine)  
**Status**: **PREPARATORY ARCHITECTURAL SPECIFICATION — ZERO PHASE 5 CODE EXECUTED**

---

## 1. Overview & System Boundary

The research framework bridges computer vision detection into agronomic plantation health monitoring:
```text
Phase 3 & 4 (Vision Layer):
Image Capture → Preprocessing → YOLOv8 Inference → Raw Bounding Boxes & Confidence Scores
                                                        ↓
Phase 5 (Agronomic Analytics Layer):
Raw Detection JSON Schema → Spatial & Severity Analytics → Plantation Health Index (PHI)
                                                        ↓
Phase 6 (Deployment Layer):
Mobile Edge Application (Flutter / TFLite) & Plantation Heatmap
```

Phase 4 defines the exact machine-readable contract that Phase 5 will ingest. Phase 5 will implement the plantation health index, disease severity grading, and spatial aggregation logic without altering the vision backbone.

---

## 2. Ingestion Schema (`detection_output_schema.json`)

Phase 5 will consume output records adhering to `outputs/evaluation/detection_output_schema.json`. Each inference record provides:
1. `image_id`: Unique identifier (string).
2. `timestamp`: ISO-8601 capture timestamp (`YYYY-MM-DDTHH:MM:SSZ`) or null if running batch offline.
3. `gps_location`: Latitude and longitude coordinates (object or null).
4. `model` & `model_version`: Ingesting model identifier (e.g., `YOLOv8n`, `v1.0.0-sanitized-exp001`).
5. `image_width` & `image_height`: Native image resolution for box area normalization.
6. `health_status`: High-level image assessment (`HEALTHY`, `DISEASED`, `UNCERTAIN`).
7. `detections`: Array of localized lesions, each with:
   - `class_id`: Integer $\in [0, 4]$.
   - `disease_class`: String name (`bud root dropping`, `bud rot`, `gray leaf spot`, `leaf rot`, `stembleeding`).
   - `confidence`: Calibrated prediction probability $\in [0.0, 1.0]$.
   - `bounding_box`: Pixel coordinates `{x1, y1, x2, y2}`.

---

## 3. Required Processing Capabilities for Phase 5

Phase 5 will implement the following modules based on the detector outputs:

### A. Lesion Severity Indexing (LSI)
- **Area-Based Severity**: Ratio of disease bounding box area to visible palm structure area:
  $$\text{Relative Area} = \frac{\sum_{i} (x_{2,i} - x_{1,i}) \times (y_{2,i} - y_{1,i})}{W \times H}$$
- **Pathological Severity Weighting**: Different diseases have different mortality risks:
  - `bud rot`: Weight 1.0 (Lethal crown rot; leads to palm death if untreated)
  - `stembleeding`: Weight 0.85 (Vascular pathogen; weakens trunk load capacity)
  - `bud root dropping`: Weight 0.70 (Direct economic crop loss)
  - `leaf rot`: Weight 0.50 (Reduces photosynthetic capacity)
  - `gray leaf spot`: Weight 0.30 (Folio-fungal spotting; manageable with copper fungicide)

### B. Plantation Health Index (PHI)
- Phase 5 will aggregate multiple tree inspections across a plantation block into a composite Plantation Health Index ($PHI \in [0, 100]$):
  - $PHI \ge 85$: Healthy / Excellent
  - $70 \le PHI < 85$: Mild Infestation (Targeted Spot Treatment)
  - $50 \le PHI < 70$: Moderate Outbreak (Quarantine / Canopy Spraying)
  - $PHI < 50$: Severe Epidemic (Agronomic Emergency Intervention)

### C. Background Rejection Handling
- Incorporate verified healthy/negative frames (zero bounding boxes) as positive health evidence to increase the plantation health score, balancing disease detections with verified healthy palm counts.

---

## 4. Phase 5 Implementation Boundaries

- **Strict Separation**: Phase 5 will NOT train or fine-tune neural network weights.
- **Data Invariance**: Phase 5 operates strictly as a downstream analytical consumer of Phase 4 outputs.
