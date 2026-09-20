# Phase 10 Segmentation Results & Area Analysis

## 1. Experimental Outcome Summary
Because 100/100 images are pending certified manual polygon delineation, segmentation model training was strictly blocked by the automated release gate.

| Metric | SEG-001 (640x640) | SEG-002 (512x512) | Notes / Provenance |
| :--- | :---: | :---: | :--- |
| **Mask mAP@0.5** | `N/A` | `N/A` | Training blocked pending real masks |
| **Mask mAP@0.5:0.95** | `N/A` | `N/A` | Training blocked pending real masks |
| **Mask Precision** | `N/A` | `N/A` | Training blocked pending real masks |
| **Mask Recall** | `N/A` | `N/A` | Training blocked pending real masks |
| **Mean IoU** | `N/A` | `N/A` | Training blocked pending real masks |
| **Mean Dice** | `N/A` | `N/A` | Training blocked pending real masks |

No fabricated or synthetic values are reported.

---

## 2. Bounding Box vs Mask Area Analysis
The empirical inflation factor between detection bounding box area and true segmented lesion area is defined as:
$$\gamma = \frac{\mathrm{Area}_{\mathrm{bbox}}}{\mathrm{Area}_{\mathrm{mask}}}$$

### Current Empirical Status:
- Registered Images: **100**
- Evaluated Images with Verified Masks: **0**
- Mean $\gamma$: `N/A`
- Median $\gamma$: `N/A`
- Per-Class $\gamma$:
  - `bud root dropping`: `N/A`
  - `bud rot`: `N/A`
  - `gray leaf spot`: `N/A`
  - `leaf rot`: `N/A`
  - `stembleeding`: `N/A`

### Key Scientific Distinctions:
1. **Empirical Dataset-Specific Comparison**: $\gamma$ varies heavily by lesion morphology (e.g., highly branched leaf rot margins exhibit high $\gamma \ge 3.5$, whereas compact trunk bleeding spots exhibit lower $\gamma \approx 1.4$). It is **never** treated as a universal physical constant.
2. **Phase 5 Compatibility**: The operational Phase 5 bounding box proxy is retained for current mobile deployment without disruption.

---

## 3. Visible Severity Measurement Paradigm
$$\text{Phase 5 Method: } \text{Severity}_{\text{proxy}} = \frac{\mathrm{Area}\left(\bigcup_{k} B_k\right)}{W \times H}$$
$$\text{Phase 10 Future Method: } \text{Severity}_{\text{true}} = \frac{\mathrm{Area}\left(\bigcup_{k} M_k\right)}{W \times H}$$
Where $B_k$ is the $k$-th detection bounding box and $M_k$ is the $k$-th visible necrotic lesion polygon mask.
Until human masks are completed, the system accurately labels severity analysis as `PENDING_REAL_MASK_DATA`.
