# Automated Dataset Audit Report: Coconut Tree Disease

**Audit Directory**: `C:\Users\Hp\.gemini\antigravity-ide\scratch\coconut-disease-yolov8\data\raw\mendeley_coconut_disease`  
**Total Images Audited**: 100  
**Valid Images**: 100  
**Corrupted / Unreadable**: 0 corrupted, 0 unreadable  
**Duplicate Filename Groups**: 0  

## 1. Class Distribution

| Class Name | Image Count | Proportion |
| :--- | :---: | :---: |
| **Bud Root Dropping** | 20 | 20.00% |
| **Bud Rot** | 20 | 20.00% |
| **Gray Leaf Spot** | 20 | 20.00% |
| **Leaf Rot** | 20 | 20.00% |
| **Stem Bleeding** | 20 | 20.00% |

## 2. Image Formats and Dimensions

- **Supported Formats**: {'JPEG': 100}
- **Minimum Resolution**: 768x1024
- **Maximum Resolution**: 768x1024
- **Mean Resolution**: 768.0 x 1024.0
- **Unique Dimensions**: 1 distinct aspect ratios / sizes

## 3. Annotation & Object Detection Integrity

- **Annotations Detected**: NO
- **Total Annotation Files**: 0
- **Empty Annotation Files**: 0
- **Malformed Coordinates**: 0
- **Total Bounding Boxes**: 0
- **Mean Bounding Boxes per Image**: 0.00
- **Mean Normalized Bounding Box Area**: 0.0000

## 4. Quality Anomalies & Recommendations

- **Data Integrity**: Clean. 100% of examined image files were successfully parsed by PIL and OpenCV.
- **Annotation Status**: **Classification-Only Source**. Bounding box ground truth does NOT exist in the raw dataset hierarchy. Manual or AI-assisted bounding box annotation is required before YOLOv8 object detection training can proceed.