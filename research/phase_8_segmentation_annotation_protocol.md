# Phase 8 Segmentation Annotation Protocol & Quality Control

## 1. Objective & Scope

Given that zero public segmentation datasets exist for the 5 target coconut palm diseases, this document establishes a standardized, reproducible **Manual Polygon/Mask Annotation Workflow** to guide future human-in-the-loop labeling campaigns.

---

## 2. Annotation Specifications

### 2.1 Annotation Targets & Units
- **Target Feature**: Visible diseased plant tissue displaying active pathology symptoms:
  1. `bud root dropping`: Premature fallen buttons/nuts with basal necrosis.
  2. `bud rot`: Necrotic lesions on spindle leaf, crown meristem discoloration, collapsed spear leaves.
  3. `gray leaf spot`: Individual elliptic fungal leaf spots with ash-grey centers and dark brown margins.
  4. `leaf rot`: Dark brown or black necrotic flaccid leaf margins, rotting leaflets, and apical blights.
  5. `stembleeding`: Dark brown/black rust-colored bleeding patches on the fibrous trunk bark.
- **Annotation Unit**: Closed 2D polygon vertices (`[[x1, y1], [x2, y2], ...]`) or pixel-level binary raster masks.
- **Coordinate Space**: Normalized image coordinates ($x, y \in [0.0, 1.0]$) to maintain resolution independence.

### 2.2 Boundary Delineation Rules
1. **Lesion Margins**: Polygons must strictly encompass necrotic/chlorotic tissue margins. Do **NOT** include healthy green fronds or background sky.
2. **Cluster Spotting**: For numerous minute lesions (common in early *gray leaf spot*), annotate clusters of $\le 5$ contiguous spots as a single polygon if individual separation is beneath optical resolution ($< 5\text{px}$).
3. **Vertex Density**: Minimum 8 vertices for convex lesions; minimum 16 vertices for irregular, branching foliar decay.

---

## 3. Mandatory Metadata Schema Contract

Every annotated image record must be accompanied by an export JSON containing:

```json
{
  "annotation_id": "ANN-2026-BLK01-IMG042-01",
  "image_id": "LeafRot007.jpg",
  "tree_id": "PALM-BLK01-R03-T08",
  "disease_class": "leaf rot",
  "class_id": 3,
  "annotator_id": "AGRONOMIST_EXPERT_04",
  "annotation_date": "2026-10-25T14:20:00Z",
  "mask_format": "POLYGON_NORMALIZED",
  "num_vertices": 18,
  "polygon_points": [
    [0.1534, 0.2662],
    [0.1842, 0.2510],
    [0.2105, 0.2801],
    [0.1912, 0.3204],
    [0.1501, 0.2988]
  ],
  "area_pixels": 4250,
  "polygon_area_ratio": 0.0162,
  "confidence_rating": 1.0,
  "quality_flag": "DOUBLE_REVIEW_VERIFIED"
}
```

---

## 4. Quality-Control (QC) & Multi-Review Protocol

To eliminate subjective annotator bias in plant pathology segmentation:
1. **Double Annotation Cohort**: A randomized $20\%$ split of all benchmark images must be independently segmented by two certified agronomists/botanists.
2. **Inter-Annotator Agreement**: Calculate Intersection-over-Union (IoU) between annotator masks:
   $$\text{IoU}_{\text{inter}} = \frac{\text{Area}(M_1 \cap M_2)}{\text{Area}(M_1 \cup M_2)}$$
   - Target Acceptance: $\text{IoU}_{\text{inter}} \ge 0.75$.
3. **Arbitration Gate**: Any sample with $\text{IoU}_{\text{inter}} < 0.75$ must be submitted to a senior phytopathologist for boundary arbitration before inclusion in training splits.
4. **Versioning**: Maintain cryptographic SHA-256 manifests for every released annotation package.
