# Phase 9 Segmentation Methodology & Bounding Box vs Mask Analysis

## 1. Mathematical Formulation: Bounding Box Area vs Real Mask Area

The central scientific hypothesis of extending object detection to fine-grained segmentation is that **rectangular bounding boxes systematically overestimate visible foliar necrosis coverage**.

### 1.1 Mathematical Formulation of Inflation Factor

Let an image $I$ with dimensions $W \times H$ contain $K$ disease lesions.
The bounding box area proxy $A_{\text{box}}$ is given by:
$$A_{\text{box}} = \text{Area}\left(\bigcup_{k=1}^K B_k\right)$$
where $B_k = [x_1, y_1, x_2, y_2]_k$ is the axis-aligned bounding box.

The true segmented lesion area $A_{\text{mask}}$ is given by:
$$A_{\text{mask}} = \sum_{y=1}^H \sum_{x=1}^W M_{\text{lesion}}(x, y)$$
where $M_{\text{lesion}} \in \{0, 1\}^{H \times W}$ is the binary ground-truth segmentation mask.

The **Box-to-Mask Inflation Factor** ($\gamma$) is defined as:
$$\gamma = \frac{A_{\text{box}}}{A_{\text{mask}}}$$

### 1.2 Theoretical Geometric Properties
- **Convex vs Non-Convex Geometry**: Coconut leaflets are narrow, elongated, and often curved. Necrotic lesions follow leaflet veins diagonally across the image.
- **Lower Bound**: By definition, any bounding box must enclose the entire polygon, so:
  $$A_{\text{box}} \ge A_{\text{mask}} \implies \gamma \ge 1.0$$
- **Expected Empirical Range**: For diagonal fungal streaks (*leaf rot*) and small scattered circular spots (*gray leaf spot*), the bounding box captures extensive interstitial healthy tissue and open sky, leading to expected inflation factors of:
  $$\gamma \in [2.5, 6.0]$$

---

## 2. Experimental Design for Future Model Benchmarking

When manual polygon annotations are completed, the following controlled experiments are prepared:

### Experiment SEG-001 (Baseline 640x640):
- **Model**: `YOLOv8n-seg` (COCO-pretrained segmentation backbone)
- **Input Resolution**: $640 \times 640$
- **Loss Functions**: Multi-task loss combining Box Regression Loss ($L_{\text{box}}$), Classification Loss ($L_{\text{cls}}$), and Mask BCE Loss ($L_{\text{mask}}$).
- **Evaluation Metrics**:
  - Mask Intersection-over-Union (mIoU)
  - Dice Similarity Coefficient ($F_1 = \frac{2|X \cap Y|}{|X| + |Y|}$)
  - Mask mAP@0.5 and mAP@0.5:0.95

### Experiment SEG-002 (Mobile-Optimized 512x512):
- **Model**: `YOLOv8n-seg`
- **Input Resolution**: $512 \times 512$
- **Comparison Focus**: CPU inference latency vs mask boundary precision.

---

## 3. Backward-Compatible Severity Engine Integration

The application severity engine (`src/segmentation/interface.py`) operates with a strict dual-mode fallback:
```
                               [Image Input]
                                     |
                         +-----------+-----------+
                         |                       |
            [Real Polygons Available]    [Real Polygons Pending]
                         |                       |
                         v                       v
               [Segmented Mask Area]    [Phase 5 Sweep-Line BBox Union]
                         |                       |
                         +-----------+-----------+
                                     |
                                     v
                        [Standardized Area Metric]
```
This architecture guarantees that the mobile application never fails, returns bitwise reproducible Phase 5 metrics currently, and will seamlessly adopt pixel-accurate masks once manual annotation concludes.
