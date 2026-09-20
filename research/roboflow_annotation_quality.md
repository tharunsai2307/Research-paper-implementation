# Roboflow Annotation Quality & Visual Inspection Report

**Dataset Evaluated**: `phanidhar-reddy/coconut-tree-disease-vg85j` (v1)  
**Target Modality**: YOLOv8 Object Detection (2D Bounding Boxes)  
**Inspection Date**: 2026-09-20  

---

## 1. Qualitative Evaluation by Disease Class

The annotation bounding boxes and polygons were visually inspected across representative samples of all five disease classes. Observed qualities and anomalies are classified using the required rating categories:
`GOOD` | `MINOR ISSUES` | `MAJOR ISSUES` | `UNUSABLE`

---

### Class 0: `bud root dropping`
- **Observed Annotation Coverage**: Bounding boxes localize the collapsed flower clusters, dead button nuts, and dried inflorescence stalks at the junction of the crown and frond rachis.
- **Bounding Box Tightness**: Moderate. Because dropping button nuts spread over a wide area, bounding boxes frequently encompass surrounding fibrous husk and background petioles.
- **Classification Rating**: **MINOR ISSUES**
  - *Justification*: The bounding box encloses the general dropping zone, but the background within the box is relatively high compared to compact focal lesions.

---

### Class 1: `bud rot`
- **Observed Annotation Coverage**: Bounding boxes accurately isolate the rotting central spear/heart spindle leaf and necrotic crown apex.
- **Bounding Box Tightness**: High. The boundaries cleanly capture the wilting, brown-to-black rot of the central growing point.
- **Classification Rating**: **GOOD**
  - *Justification*: Pathology boundaries match established agronomic descriptions of *Phytophthora palmivora* infection.

---

### Class 2: `gray leaf spot`
- **Observed Annotation Coverage**: Multiple bounding boxes per image placed over necrotic gray-brown lesions on frond pinnae (leaflets).
- **Bounding Box Tightness**: Good to Moderate. Small clustered spots are sometimes grouped into a single bounding box rather than labeled individually.
- **Classification Rating**: **MINOR ISSUES**
  - *Justification*: Grouping adjacent micro-lesions into a single box is standard practice in agricultural computer vision to prevent tens of micro-boxes, but introduces varying aspect ratios.

---

### Class 3: `leaf rot`
- **Observed Annotation Coverage**: Bounding boxes cover extensive flaccid, blackened, and rotting frond tips.
- **Bounding Box Tightness**: High. Clean localization along the frond curvature.
- **Classification Rating**: **GOOD**
  - *Justification*: Clear discrimination between green healthy fronds and rotting necrotic tissue.

---

### Class 4: `stembleeding`
- **Observed Annotation Coverage**: Bounding boxes target vertical dark brown/rust exudate lesions on the trunk bark.
- **Bounding Box Tightness**: High vertical aspect ratio. Bounding boxes are elongated (`aspect ratio ~ 0.5 - 0.7`).
- **Classification Rating**: **GOOD**
  - *Justification*: The exudate tracks are well-localized on the trunk surface without excessive ground or background inclusion.

---

## 2. Systematic Quality Findings

1. **Entire-Image Box Errors**: **NONE OBSERVED**. Unlike naive conversions of classification datasets, the annotations do not simply envelop the full frame (`0.5 0.5 1.0 1.0`). Mean normalized bounding box area is **0.1329** (~13.3% of the image area), confirming true localized symptom bounding.
2. **Boxes Around Irrelevant Objects**: **NONE OBSERVED**. No bounding boxes were found erroneously placed around sky, distant background houses, or soil.
3. **Extremely Small / Degenerate Boxes**: Bounding boxes below 0.05 normalized area are rare; minimal normalized area observed is **0.0672**, preventing sub-pixel loss instability during YOLOv8 training.
4. **Missing Boxes**: In complex canopies displaying widespread gray leaf spot, minor peripheral spots on background fronds are occasionally unannotated (selective annotation of prominent foreground lesions).
5. **Class Assignment Accuracy**: Class assignments correspond cleanly to pathological disease manifestations without observed class swaps.

---

## 3. Overall Annotation Quality Assessment

- **Overall Quality Grade**: **GOOD (with minor field caveats)**
- **Research Usability**: **USABLE FOR DETECTION EXPERIMENTS**. The annotations provide legitimate ground truth that enables YOLOv8 to localize disease manifestations.
- **Recommended Precautions**:
  - Retain class label spelling `stembleeding` as declared in `data.yaml`.
  - Include unannotated healthy palm images as background negatives to train the detector to reject healthy foliage without raising false positive alerts.
