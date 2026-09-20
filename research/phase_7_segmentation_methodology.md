# Phase 7 Segmentation Methodology: Interface Design & Bounding Box vs Mask Comparison

## 1. Overview & Current Status

In Phase 7, the research framework specifies an expansion from bounding-box detection to fine-grained lesion segmentation:
```
Image -> YOLOv8 Detection -> Disease Class -> Segmentation -> Disease Mask -> Refined Severity
```

As empirically proven in `research/phase_7_data_capability_audit.md`:
- **Genuine Ground-Truth Masks**: `NOT AVAILABLE` (Zero polygon or pixel mask annotations exist in the benchmark dataset).
- **Segmentation Model Training**: `PENDING REAL MASK DATA`.

In strict adherence to Step 3 (Case B), this document specifies:
1. The formal mathematical formulation of fine-grained mask severity vs. the existing bounding box area proxy.
2. The software architecture and schema contracts implemented in `src/segmentation/` to support future real mask data seamlessly.
3. Guidelines for prospective polygon annotation campaigns.

---

## 2. Severity Estimation: Bounding Box Proxy vs. True Mask Area

### 2.1 Existing Baseline: Bounding Box Union Area Proxy (Phase 5)
In Phase 5, lesion extent is operationalized as the **Relative 2D Image-Space Affected Area Proxy** ($\alpha_{\text{bbox}}$):
$$\alpha_{\text{bbox}} = \frac{\text{Area}\left(\bigcup_{i=1}^{K} B_i\right)}{W_{\text{img}} \times H_{\text{img}}}$$
where $B_i = [x_1, y_1, x_2, y_2]$ is the bounding box of accepted detection $i$, and $\text{Area}\left(\bigcup B_i\right)$ is the exact 2D planar union calculated via the Klee's measure sweep-line algorithm.

#### Limitations of Bounding Box Area:
- Rectangles encompass non-diseased tissue, healthy frond background, sky, and neighboring canopy foliage.
- For elongated or diagonal lesions (e.g., foliar blights tracking along a frond leaflet), a rectangular bounding box drastically overestimates the infected area relative to the whole photograph.

### 2.2 Advanced Formulation: True Pixel Mask Severity ($\alpha_{\text{mask}}$)
When ground-truth or predicted binary segmentation masks $M \in \{0, 1\}^{H \times W}$ are available, severity can be measured with high spatial precision:
$$\alpha_{\text{mask}} = \frac{\sum_{y=1}^{H}\sum_{x=1}^{W} M(x, y)}{A_{\text{denom}}}$$

#### Selection of the Denominator ($A_{\text{denom}}$):
1. **Full Image Area ($A_{\text{denom}} = W_{\text{img}} \times H_{\text{img}}$)**:
   - Comparable directly with the Phase 5 proxy.
   - Measures lesion footprint relative to the camera field of view.
2. **Visible Plant Canopy Area ($A_{\text{denom}} = A_{\text{canopy}}$)**:
   - Where $A_{\text{canopy}} = \sum M_{\text{canopy}}(x, y)$.
   - The most biologically and agronomically defensible metric, computing the percentage of visible palm tissue that has become necrotic or chlorotic:
   $$\text{Severity}_{\text{foliar}} = \frac{\text{Lesion Pixel Count}}{\text{Total Palm Canopy Pixel Count}} \times 100\%$$

---

## 3. Comparison Matrix: Bounding Box vs. Semantic Mask

| Property | Phase 5 Bounding Box Proxy | Phase 7 True Segmentation Mask |
| :--- | :--- | :--- |
| **Annotation Requirement** | 4-point bounding box (`[xc, yc, w, h]`) | Closed polygon vertices (`[[x1,y1], [x2,y2], ...]`) or RLE bitmask |
| **Current Dataset Status** | **AVAILABLE (120 bboxes, 100% verified)** | **UNAVAILABLE (0 masks)** |
| **Model Architecture** | YOLOv8n Detection (`best.pt`, 3.01M params) | YOLOv8n-Seg / Mask R-CNN (Future extension) |
| **Geometric Accuracy** | Convex rectangular bounding approximation | Pixel-accurate boundary tracing |
| **Background Inclusion** | High (includes sky and healthy leaflet gaps) | Zero (confined strictly to necrotic lesion margins) |
| **Computational Overhead** | Ultra-low (sweep-line union in $< 10\text{ ms}$) | Moderate (mask decoding and raster polygon fill) |
| **Research Claim** | "Relative 2D image-space affected-area proxy" | "Image-space segmented lesion coverage percentage" |
| **Clinical/Pathological Claim**| **DISCLAIMED** (Engineering proxy only) | **DISCLAIMED** (2D photographic proxy, not 3D tissue depth) |

---

## 4. Software Architecture for Future Segmentation

To ensure modular extensibility without breaking existing frozen pipelines, the system implements an abstract segmentation interface:

```
src/segmentation/
├── __init__.py
├── schemas.py           # Segmentation mask data structures and Polygon contracts
├── interface.py         # AbstractBaseSegmentationModel interface
└── fallback_proxy.py    # Fallback to Phase 5 exact sweep-line proxy when masks are pending
```

If a real polygon annotation dataset is provided in the future, developers simply subclass `AbstractBaseSegmentationModel` without refactoring the Phase 5/6 serving infrastructure or mobile client.
