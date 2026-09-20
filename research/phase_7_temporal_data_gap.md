# Phase 7 Temporal Data Gap & Longitudinal Collection Protocol

## 1. Overview & Problem Definition

The objective of an AI-driven disease progression model is to predict the future severity state $S_{t+h}$ of a specific plant at prediction horizon $h$ given a historical sequence of health observations:
$$\{ (x_{t_0}, S_{t_0}), (x_{t_1}, S_{t_1}), \dots, (x_{t_k}, S_{t_k}) \}$$

Following the comprehensive Phase 7 Data Capability Audit (`research/phase_7_data_capability_audit.md`), it was conclusively established that:
```text
NOT SCIENTIFICALLY SUPPORTED BY CURRENT DATA
```
Training deep recurrent neural networks (LSTM / GRU) on the existing dataset is scientifically untenable. This document details:
1. The exact mathematical and biological reasons why existing data cannot support temporal modeling.
2. The risks of pseudo-temporal data synthesis.
3. The definitive field collection protocol and metadata schema required for prospective longitudinal research.

---

## 2. Why Existing Data Cannot Support Temporal Progression

### 2.1 Static Snapshot vs. Longitudinal Process
Coconut palm foliar necrosis, bud decay, and stem bleeding represent dynamic phytopathological interactions governed by pathogen proliferation, host defense mechanisms, and environmental microclimate. In commercial *Cocos nucifera* plantations:
- **Leaf Rot (*Bipolaris incurvata* / *Colletotrichum gloeosporioides*)**: Expands over **14 to 30 days** under high relative humidity ($> 85\%$).
- **Bud Rot (*Phytophthora palmivora*)**: Causes progressive heart rot leading to spear leaf collapse over **7 to 21 days**.
- **Stem Bleeding (*Thielaviopsis paradoxa*)**: Exudes dark viscous sap that tracks downward over **30 to 90 days**.

In contrast, the available images in `data/processed/coconut_detection_clean/` originate from two morning field visits (June 30, 2023, and July 9, 2023). Within each session, multiple photos of the same scene were captured within **1 to 5 seconds** as camera burst sequences. The time elapsed $\Delta t \approx 2\text{ seconds}$ represents zero pathological progression.

### 2.2 Lack of Permanent Specimen Tracking
The dataset provides no tree indexing (no physical ear tags, trunk paint numbers, QR codes, or differential GPS coordinates). It is impossible to ascertain whether `LeafRot007.jpg` captured on June 30 is the same physical palm tree as `LeafRot015.jpg` captured on July 9, or a completely different tree in another row or orchard. Pairing them into a sequential time-series would construct synthetic noise that teaches a recurrent model spurious transitions.

### 2.3 Danger of Pseudo-Temporal Augmentation
A common but scientifically invalid practice in literature is "pseudo-temporal synthesis" — taking static photos, sorting them arbitrarily by bounding box size or confidence, and presenting them as an LSTM training sequence. 
- **Flaw 1**: Confuses spatial camera distance (a closer photograph has larger bounding boxes) with disease expansion.
- **Flaw 2**: Destroys temporal causality and invents non-existent epidemiological dynamics.
- **Flaw 3**: Leads to severely inflated, non-reproducible evaluation metrics that immediately fail in real-world agronomic deployment.

Under strict research integrity, **we refuse to fabricate artificial sequences**.

---

## 3. Recommended Longitudinal Field Collection Protocol

To enable legitimate temporal progression modeling in future field campaigns, agronomists and field researchers must follow the structured protocol outlined below.

### 3.1 Specimen Identification & Tagging
1. **Permanent Physical Tagging**: Each observed coconut tree must be assigned a unique, immutable specimen identifier (e.g., `PALM-BLK01-R04-T12`).
2. **Geospatial Anchoring**: High-precision GNSS/GPS coordinates (latitude, longitude, altitude) must be recorded for each specimen to allow GIS mapping.
3. **Fixed Camera Standpoint**: For each tree, mark a fixed ground observation station (distance $2.5\text{ m}$, orientation e.g., North-facing canopy view) to eliminate distance-induced scale discrepancies.

### 3.2 Longitudinal Sampling Frequency
Observations must be collected at biologically meaningful intervals:
- **High-Risk Wet Season**: Revisit every **7 days** ($\pm 1$ day).
- **Dry / Maintenance Season**: Revisit every **14 to 28 days**.
- **Minimum Sequence Length**: A minimum of **4 consecutive visits** per tree (e.g., Day 0, Day 7, Day 14, Day 21) is required to capture an initial state, a growth trajectory, and an inflection point.

### 3.3 Required Longitudinal Schema Contract

```json
{
  "observation_id": "OBS-2026-BLK01-T004-003",
  "tree_id": "PALM-BLK01-R02-T004",
  "plantation_id": "COCO_NORTH_ESTATE_01",
  "timestamp": "2026-10-14T08:30:00Z",
  "day_index": 14,
  "days_since_first_observation": 14,
  "days_since_previous_observation": 7,
  "camera_metadata": {
    "distance_meters": 2.5,
    "azimuth_degrees": 0.0,
    "focal_length_mm": 26.0
  },
  "environmental_covariates": {
    "mean_temp_celsius_7d": 28.4,
    "cumulative_rainfall_mm_7d": 42.6,
    "mean_relative_humidity_pct_7d": 88.2
  },
  "yolo_detection": {
    "primary_disease": "leaf rot",
    "accepted_detection_count": 2,
    "mean_confidence": 0.84,
    "relative_affected_area_proxy": 0.0482
  },
  "expert_agronomic_staging": {
    "pathologist_severity_grade": 2,
    "foliar_necrosis_percentage": 5.0,
    "active_treatment_applied": "none"
  }
}
```

---

## 4. Leakage-Free Temporal Splitting Strategy

When real longitudinal dataset assets become available, dataset splitting must strictly adhere to **Specimen-Level Grouping (GroupKFold by `tree_id`)**:

```
+-------------------------------------------------------------+
|               ALL LONGITUDINAL PALM SPECIMENS               |
+-------------------------------------------------------------+
       |
       +---> TRAIN SPECIMENS (70% of Trees, e.g. T001 - T070)
       |     Contains ALL observation days (Day 0, 7, 14, 21...)
       |     for trees T001 through T070.
       |
       +---> VALIDATION SPECIMENS (15% of Trees, e.g. T071 - T085)
       |     Contains ALL observation days for trees T071 - T085.
       |
       +---> TEST SPECIMENS (15% of Trees, e.g. T086 - T100)
             Contains ALL observation days for trees T086 - T100.
             Completely unseen palms; tests generalization!
```

> [!CAUTION]
> **Strict Rule**: NEVER split by image index across time for the same palm (e.g., Day 0 in Train and Day 7 in Test). This causes catastrophic temporal leakage because static background foliage, trunk scars, and orchard lighting allow deep networks to identify the tree rather than learning disease progression dynamics.
