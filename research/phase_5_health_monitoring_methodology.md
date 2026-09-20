# Phase 5 — Plantation Health Monitoring Methodology & Mathematical Framework

**Project Title:** A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Project Directory:** `C:\Users\Hp\.gemini\antigravity-ide\scratch\coconut-disease-yolov8`  
**Phase:** Phase 5 — Plantation Health Monitoring Engine  
**Status:** COMPLETE  
**Date:** 2026-09-20  

---

## 1. Objective

The primary objective of Phase 5 is to design, implement, and validate an automated **Plantation Health Monitoring Engine** that bridges low-level YOLOv8 object detections (bounding boxes, class labels, confidences) and high-level agronomic indicators for coconut plantation health assessment.

The engine executes this standardized pipeline:

$$\text{Field Image(s)} \longrightarrow \text{YOLOv8 Inference} \longrightarrow \text{Standardized Detection Records} \longrightarrow \text{Disease Distribution} \longrightarrow \text{Geometric Severity Proxy} \longrightarrow \text{Plantation Health Index (PHI)} \longrightarrow \text{Mobile JSON / Agronomic Reports}$$

### Core Design Principles
1. **Model & Weight Independence**: Operates strictly downstream of inference; zero YOLO retraining and zero modification of Phase 3 training checkpoints or Phase 4 evaluation metrics.
2. **Scientific Integrity**: Zero fabrication of disease incidence, severity grades, or plantation health scores.
3. **Decoupled Counting**: Explicit distinction between total lesion count ($N_{\text{det}}$) and unique affected tree/image count ($N_{\text{tree}}$).
4. **Strict Overlap Handling**: Exact analytical 2D bounding box union area calculation via Klee's measure sweep-line algorithm, strictly preventing double-counting of overlapping lesion boxes.
5. **Separation of Evidence vs. Clinical Claims**: Image-space geometric proxies are clearly designated as visual impact indicators, strictly distinguished from agronomic or biological disease staging.

---

## 2. Standardized Input Contract

The input contract builds upon `outputs/evaluation/detection_output_schema.json` and defines the required schema for image-level and detection-level records.

### Image-Level Record Fields
* `image_id` (str): Filename or unique capture hash.
* `image_path` (str): Absolute file URI on host.
* `plantation_id` (str, optional): Target plantation or block identifier (e.g., `BLOCK_01`).
* `timestamp` (str, ISO 8601): UTC timestamp of capture/processing (`YYYY-MM-DDTHH:MM:SSZ`).
* `gps_location` (dict, nullable): `{"latitude": float, "longitude": float}` if provided by mobile sensor; strictly `null` if unavailable (zero GPS fabrication).
* `model_id` (str): Unique identifier of inference model (`EXP-001_baseline_yolov8n` or `EXP-002_imgsz512`).
* `model_input_size` (int): Native input image dimension (640 or 512).
* `image_width` (int), `image_height` (int): Pixel dimensions.
* `observation_status` (str): State classification (see Section 5).
* `accepted_detection_count` (int): Total detections satisfying confidence cutoff $\ge \tau_{\text{op}}$.
* `candidate_detection_count` (int): Detections in sub-threshold band $[\tau_{\text{cand}}, \tau_{\text{op}})$.
* `accepted_detections` (list[dict]): Standardized detection records.
* `candidate_detections` (list[dict]): Low-confidence candidate records.
* `detected_classes` (list[str]): Unique sorted disease classes identified in frame.
* `relative_affected_area_proxy` (float): Geometrical union area ratio $\in [0.0, 1.0]$.
* `confidence_summary` (dict): `mean`, `max`, `min` of accepted confidences.

### Detection Record Fields
* `class_id` (int): Integer identifier $\in [0, 4]$.
* `disease_class` (str): Pathology label (`bud root dropping`, `bud rot`, `gray leaf spot`, `leaf rot`, `stembleeding`).
* `confidence` (float): Softmax score $\in [0.0, 1.0]$.
* `bounding_box` (dict): Pixel coordinates `{"x1": float, "y1": float, "x2": float, "y2": float}`.
* `normalized_box` (dict): Bounded coordinates `{"xc": float, "yc": float, "w": float, "h": float}` $\in [0.0, 1.0]$.

---

## 3. Operational Confidence Threshold Configuration

In accordance with Phase 4 research recommendations and research integrity rules:
* The operational threshold was **NOT** post-hoc tuned on the Phase 4 test set.
* Default operational threshold: $\tau_{\text{op}} = 0.25$ (standard YOLO engineering configuration).
* Default low-confidence candidate threshold: $\tau_{\text{cand}} = 0.10$.
* Default IoU threshold for NMS: $\tau_{\text{iou}} = 0.70$.

All thresholds are modular, externally configurable via CLI arguments or configuration files, and explicitly documented as operational engineering assumptions.

---

## 4. Observation State Classification

Every processed image is assigned one of five mutually exclusive operational states:

1. **`DISEASE_DETECTED`**:
   $$\exists \ d \in \text{Detections} \quad \text{such that} \quad \text{conf}(d) \ge \tau_{\text{op}}$$
   Confirmed target pathology localized above the operational confidence threshold.

2. **`NO_DISEASE_DETECTED`**:
   $$\forall \ d \in \text{Detections}, \quad \text{conf}(d) < \tau_{\text{cand}} \quad \text{or} \quad |\text{Detections}| = 0$$
   Visual frame contains no detected disease features (healthy foliage / background).

3. **`LOW_CONFIDENCE_CANDIDATE`**:
   $$|\{d \mid \text{conf}(d) \ge \tau_{\text{op}}\}| = 0 \quad \land \quad \exists \ d \mid \tau_{\text{cand}} \le \text{conf}(d) < \tau_{\text{op}}$$
   Potential pathological features observed, but confidence is insufficient for confirmed diagnosis. Cataloged for human review without inflating confirmed disease statistics.

4. **`INVALID_INPUT`**:
   Image file missing, corrupted, unreadable, or invalid pixel buffer.

5. **`INFERENCE_ERROR`**:
   Model forward-pass failure or hardware execution exception.

---

## 5. Disease Aggregation & Distribution Methodology

### Decoupling Detection Count and Affected-Image Count
A critical failure mode in crop health analytics is equating lesion counts with infected palm counts. For diseases like *gray leaf spot* or *leaf rot*, a single canopy photograph may contain 10+ distinct necrotic foci. Treating each detection as an infected tree severely inflates disease incidence.

The engine strictly maintains:
* **Total Accepted Detections ($N_{\text{det}}$)**: Total count of accepted bounding boxes across all observations.
* **Unique Affected Images / Trees ($N_{\text{tree}}$)**: Total count of distinct photos exhibiting at least one accepted detection of disease class $i$.

### Mathematical Formulas

#### Class Proportion of Total Detections ($P_{\text{det}, i}$):
$$P_{\text{det}, i} = \frac{\sum_{j=1}^{M} D_{i, j}}{N_{\text{det}}} \times 100$$
Where $D_{i, j}$ is the count of class $i$ detections in image $j$, and $N_{\text{det}} = \sum_i \sum_j D_{i, j}$.

#### Tree/Image Infection Rate ($P_{\text{tree}, i}$):
$$P_{\text{tree}, i} = \frac{|\{j \in [1, M] \mid D_{i, j} > 0\}|}{M_{\text{valid}}} \times 100$$
Where $M_{\text{valid}}$ is the total number of successfully processed observation photos.

#### Overall Disease-Positive Observation Rate:
$$\text{Rate}_{\text{positive}} = \frac{M_{\text{pos}}}{M_{\text{valid}}} \times 100$$

#### Overall Disease-Negative (Healthy) Observation Rate:
$$\text{Rate}_{\text{negative}} = \frac{M_{\text{neg}}}{M_{\text{valid}}} \times 100$$

---

## 6. Geometric Severity Proxy & Overlap Handling

### Bounding Box Intersection over Union (IoU)
For two boxes $B_1 = [x_1^{(1)}, y_1^{(1)}, x_2^{(1)}, y_2^{(1)}]$ and $B_2 = [x_1^{(2)}, y_1^{(2)}, x_2^{(2)}, y_2^{(2)}]$:

$$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} = \frac{\max(0, \min(x_2^{(1)}, x_2^{(2)}) - \max(x_1^{(1)}, x_1^{(2)})) \times \max(0, \min(y_2^{(1)}, y_2^{(2)}) - \max(y_1^{(1)}, y_1^{(2)}))}{\text{Area}(B_1) + \text{Area}(B_2) - \text{Area}(B_1 \cap B_2)}$$

### Exact 2D Union Area (Klee's Measure Sweep-Line Algorithm)
Summing raw bounding box areas $\sum \text{Area}(B_k)$ causes gross overestimation when necrotic spots overlap, frequently exceeding 100% of the image frame.

The engine implements an exact analytical sweep-line algorithm (Klee's measure in 2D):
1. Extract all unique X-coordinates $\{x_1^{(k)}, x_2^{(k)}\}_{k=1}^K$ and sort them: $x_{(1)} < x_{(2)} < \dots < x_{(P)}$.
2. For each vertical strip $[x_{(p)}, x_{(p+1)}]$ with width $W_p = x_{(p+1)} - x_{(p)}$:
   - Identify all active boxes covering this X-range ($x_1 \le x_{(p)}$ and $x_2 \ge x_{(p+1)}$).
   - Project active boxes onto the Y-axis as intervals $[y_1, y_2]$.
   - Sort Y-intervals by start coordinate and merge overlapping 1D intervals into disjoint segments $[y_{\text{start}, m}, y_{\text{end}, m}]$.
   - Compute covered height $H_p = \sum_m (y_{\text{end}, m} - y_{\text{start}, m})$.
   - Add strip union area $\Delta A_p = W_p \times H_p$.
3. Total Union Area:
   $$A_{\text{union}} = \bigcup_{k=1}^K B_k = \sum_{p=1}^{P-1} W_p H_p$$

### Relative Affected Area Proxy ($A_{\text{proxy}}$)
$$A_{\text{proxy}} = \frac{A_{\text{union}}}{W_{\text{img}} \times H_{\text{img}}} \in [0.0, 1.0]$$

> [!IMPORTANT]
> **Scientific Limitation**: $A_{\text{proxy}}$ measures the visible 2D bounding box footprint in camera space. It is a visual impact proxy and does NOT represent in-situ biological tissue necrotic percentage or agronomic disease severity grades.

---

## 7. Plantation Health Index (PHI) Formulation

The Plantation Health Index translates aggregated tree observations into an interpretable metric on a $[0, 100]$ scale, where $100$ represents zero observed disease pressure and $0$ represents severe systemic infestation.

### 1. Transparent Incidence-Only Formulation ($\text{PHI}_{\text{incidence}}$)
When spatial lesion proxies are unavailable or uncalibrated, the unweighted incidence-only index provides a pure, assumption-free benchmark:

$$\text{PHI}_{\text{incidence}} = 100 \times \left(1.0 - \frac{M_{\text{pos}}}{M_{\text{valid}}}\right)$$

### 2. Composite Formulation ($\text{PHI}_{\text{composite}}$)
Incorporates both tree infection incidence and mean visible lesion footprint:

$$\text{PHI}_{\text{composite}} = 100 \times \max\left(0.0, \ 1.0 - \left[w_1 \cdot \frac{M_{\text{pos}}}{M_{\text{valid}}} + w_2 \cdot \overline{A_{\text{proxy}}}\right]\right)$$

Where:
* $w_1 = 0.85$: Incidence weight (proportion of infected palms).
* $w_2 = 0.15$: Area proxy weight (mean visible lesion area across all observations).
* $w_1 + w_2 = 1.0$.
* $\overline{A_{\text{proxy}}} = \frac{1}{M_{\text{valid}}} \sum_{j=1}^{M_{\text{valid}}} A_{\text{proxy}, j}$.

### Categorical Management Tiers
| PHI Score Range | Assigned Health Tier | Agronomic Management Directive |
| :--- | :--- | :--- |
| **85.0 – 100.0** | `EXCELLENT / MINIMAL_DISEASE_PRESSURE` | Maintain routine prophylactic monitoring; no immediate chemical intervention required. |
| **70.0 – 84.9** | `GOOD / LOCALIZED_TREATMENT_RECOMMENDED` | Deploy targeted field inspection to positive palms; apply spot fungicide/biocontrol treatment. |
| **50.0 – 69.9** | `MODERATE / ACTIVE_OUTBREAK_MONITORING` | Initiate block quarantine and scheduled canopy/trunk therapeutic intervention. |
| **0.0 – 49.9** | `POOR / SYSTEMIC_INTERVENTION_REQUIRED` | Severe epidemic pressure observed. Immediate agricultural extension intervention mandated. |

---

## 8. Multi-Visit & Time-Series Support

The engine data structures natively support temporal aggregation across sequential plantation scouting sessions.

### Data Model Hierarchy:
$$\text{Plantation Block} \longrightarrow \text{Observation Sessions } (t_1, t_2, \dots) \longrightarrow \text{Tree Observations} \longrightarrow \text{Lesion Records}$$

* `plantation_id`: Persistent identifier for agricultural block/parcel.
* `observation_session_id`: Unique visit identifier (e.g., `VISIT_20260920`).
* `timestamp`: ISO 8601 UTC timestamp.
* Longitudinal analysis is enabled by comparing successive $\text{PHI}(t)$ values over time. A single observation visit does not constitute a temporal trend.

---

## 9. Assumptions & Research Limitations

1. **Sampling vs. True Prevalence**: Observation rates reflect submitted photographs. Unless random, stratified spatial sampling across the entire block is conducted, rates represent **sample observation rates**, not true epidemiological prevalence.
2. **Camera Perspective**: Bounding box union area varies with camera distance, zoom, and angle. $A_{\text{proxy}}$ is an image-space proxy, not a 3D leaf area index.
3. **Weighting Heuristic**: Weights $w_1=0.85$ and $w_2=0.15$ are engineered methodological parameters prioritizing tree infection incidence over photographic frame coverage.
4. **No Automated Eradication Guarantee**: A 100% PHI score indicates zero disease detected in the sampled images, not guaranteed disease-free status across unphotographed palms.
