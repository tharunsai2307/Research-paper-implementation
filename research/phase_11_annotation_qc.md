# Phase 11 Quality Control & Inter-Annotator Agreement Protocol

## 1. Double-Annotation QC Protocol
In botanical pathology segmentation, individual observer variance in identifying chlorotic transition zones can induce boundary bias. To mitigate this:
1. **Double-Annotation Cohort**: A randomized target subset of **20 images** (20% of the benchmark pool, sampled proportionally across all 5 disease categories) is earmarked for dual annotation.
2. **Blind Annotation Execution**:
   - **Annotator A** annotates the primary reference polygon into `annotations/annotator_A/`.
   - **Annotator B** independently annotates the same image into `annotations/annotator_B/` without access to Annotator A's coordinates.
3. **Exact Raster Agreement Metric**:
   $$\text{IoU} = \frac{\text{Area}(M_A \cap M_B)}{\text{Area}(M_A \cup M_B)}$$
   $$\text{Dice} = \frac{2 \times \text{Area}(M_A \cap M_B)}{\text{Area}(M_A) + \text{Area}(M_B)}$$
   Calculated via pixel rasterization on a standardized $1000 \times 1000$ coordinate canvas via `compute_polygon_iou_dice()`.

---

## 2. Acceptance Criterion & Disagreement Resolution
- **QC Threshold**: $\text{IoU} \ge 0.75$ is required for automatic acceptance.
- **Disagreement Workflow**:
  ```
  Annotator A Polygon  +  Annotator B Polygon
                    ↓
        Compute Exact Raster IoU
                    ↓
       Is IoU >= 0.75?
        ├── YES ──> ACCEPTED Ground Truth
        └── NO  ──> Flag: QC_REVIEW_REQUIRED
                          ↓
                 Senior Phytopathologist Review
                          ↓
                 Consensus Adjudication Polygon
                          ↓
                 ACCEPTED Ground Truth
  ```
- **Consensus Provenance**: Under no circumstance is the original annotation overwritten; consensus records are tagged with `reviewer_id`, timestamp, and resolution rationale in `qc_manifest.csv`.

---

## 3. Current Inter-Annotator Agreement State
In accordance with scientific integrity rules:
- **Completed Double Annotations**: **0**
- **Passing QC ($\text{IoU} \ge 0.75$)**: **0**
- **Failing QC ($\text{IoU} < 0.75$)**: **0**
- **QC Review Required**: **0**
- **Mean Inter-Annotator IoU**: `N/A`
- **Median Inter-Annotator IoU**: `N/A`
- **Minimum IoU**: `N/A`
- **Maximum IoU**: `N/A`

No fabricated agreement numbers or simulated overlaps are reported.
