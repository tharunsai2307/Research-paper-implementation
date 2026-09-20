# Phase 10 Qualitative Error Analysis Framework

## 1. Error Taxonomy Architecture
To ensure rigorous diagnostic capability once YOLOv8-Seg models are trained on human-annotated masks, Phase 10 establishes a standardized 4-tier error taxonomy:

```
                          ┌───────────────────────────┐
                          │   Segmentation Error      │
                          │        Taxonomy           │
                          └─────────────┬─────────────┘
                                        │
         ┌──────────────────┬───────────┴───────────┬──────────────────┐
         ▼                  ▼                       ▼                  ▼
┌─────────────────┐┌─────────────────┐     ┌─────────────────┐┌─────────────────┐
│ False Positive  ││ False Negative  │     │ Boundary Error  ││ Class Confusion │
│ (FP: Ghost Mask)││(FN: Missed Les) │     │ (Low Dice/IoU)  ││ (Wrong Label)   │
└─────────────────┘└─────────────────┘     └─────────────────┘└─────────────────┘
```

---

## 2. Qualitative Error Categories

### Category 1: False Positive (Ghost Mask)
- **Definition**: The model segments background foliage, dried healthy frond tips, leaf stipules, or lichens as disease lesions where ground truth has no pathological necrosis.
- **Root Cause**: High visual similarity between naturally senescent coconut leaflets and fungal leaf rot necrotic tissue.
- **Mitigation Strategy**: Include healthy negative background images (`acquire_healthy_negatives.py`) in the segmentation training splits.

### Category 2: False Negative (Missed Lesion)
- **Definition**: Certified ground truth contains an active disease polygon, but the model outputs no overlapping mask instance above confidence threshold $\tau = 0.25$.
- **Root Cause**: Minute lesion size ($< 15\text{ px}$) in early *gray leaf spot* or extreme shadows cast by overlapping fronds.
- **Mitigation Strategy**: Multi-scale feature aggregation (FPN) and tile-based inference on high-resolution field imagery.

### Category 3: Boundary Delineation Error
- **Definition**: The model detects and classifies the disease correctly, but the predicted mask boundary deviates significantly from the true lesion margin ($\text{IoU} < 0.50$).
- **Root Cause**: Chlorotic transition zones (yellowing halos around fungal necrosis) where the boundary between diseased and healthy tissue is gradational.
- **Mitigation Strategy**: High-resolution mask heads (Proto loss weighting) and explicit boundary-weighted cross-entropy loss.

### Category 4: Disease Class Confusion
- **Definition**: The model accurately isolates the lesion contour but assigns an incorrect disease class ID (e.g. labeling *bud rot* spindle decay as *leaf rot*).
- **Root Cause**: Foliar symptoms of *bud rot* and *leaf rot* share similar dark brown necrosis in early progression stages.
- **Mitigation Strategy**: Incorporate anatomical spatial priors (crown meristem vs lower canopy fronds).

---

## 3. Real Test Image Tracking
In adherence to research integrity rules, actual quantitative failure counts and qualitative crop figures are reserved for when human ground truth annotations and model checkpoints exist. No artificial failure examples or fabricated masks were manufactured.
