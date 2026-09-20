# Phase 11 Dataset Release, Class Distribution & Area Analysis

## 1. Dataset Release State
- **Release Version**: `Phase 11 Benchmark Ground-Truth Workspace v1.0.0`
- **Release Status**: **WORKSPACE OPERATIONAL / ANNOTATIONS PENDING**
- **Hard Training Gate**: **BLOCKED**
  ```text
  SEGMENTATION TRAINING = BLOCKED
  REASON = ACCEPTED REAL MASKS == 0
  ```

---

## 2. Benchmark Class & Split Distribution

| Disease Category | Class ID | Train Images | Val Images | Test Images | Total Images | Accepted Masks | Completion % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bud root dropping` | 0 | 13 | 4 | 3 | 20 | 0 | 0.0% |
| `bud rot` | 1 | 13 | 5 | 2 | 20 | 0 | 0.0% |
| `gray leaf spot` | 2 | 13 | 5 | 2 | 20 | 0 | 0.0% |
| `leaf rot` | 3 | 12 | 7 | 1 | 20 | 0 | 0.0% |
| `stembleeding` | 4 | 12 | 6 | 2 | 20 | 0 | 0.0% |
| **Total** | — | **63** | **27** | **10** | **100** | **0** | **0.0%** |

---

## 3. Box vs Mask Area Analysis ($\gamma$)
The empirical lesion inflation factor is formulated as:
$$\gamma = \frac{\text{Area}(\text{BoundingBox})}{\text{Area}(\text{Mask})}$$

### Current Empirical Value:
- **Status**: `AWAITING_DATA`
- **Mean $\gamma$**: `N/A`
- **Median $\gamma$**: `N/A`
- **Standard Deviation**: `N/A`
- **Interpretation**: True empirical $\gamma$ requires verified lesion masks. Synthetic approximations are strictly barred.

---

## 4. Severity Measurement Paradigm Comparison
- **Phase 5 Operational Metric**:
  $$\text{AffectedArea}_{\text{proxy}} = \frac{\text{Area}\left(\bigcup_k B_k\right)}{W \times H}$$
  (2D image-space bounding box union proxy; active in mobile app).
- **Phase 11 Formulation**:
  $$\text{AffectedArea}_{\text{true}} = \frac{\text{Area}\left(\bigcup_k M_k\right)}{W \times H}$$
  (2D visible necrotic lesion segmentation area measurement).
- **Current Severity Engine Status**: `PENDING_REAL_MASK_DATA`.
