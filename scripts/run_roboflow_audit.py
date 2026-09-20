"""
CLI script to run full Roboflow dataset audit (Tasks 4 through 8).
Generates:
- outputs/dataset_analysis/roboflow_annotation_summary.json
- outputs/dataset_analysis/roboflow_class_distribution.csv
- outputs/dataset_analysis/roboflow_samples/ (annotated visual overlays)
- outputs/dataset_analysis/roboflow_image_statistics.csv
- outputs/dataset_analysis/roboflow_leakage_report.json
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset.roboflow_analyzer import RoboflowDatasetAnalyzer

def main():
    dataset_dir = BASE_DIR / "data" / "raw" / "roboflow_coconut_detection"
    output_dir = BASE_DIR / "outputs" / "dataset_analysis"

    if not dataset_dir.exists():
        print(f"[-] Dataset directory missing: {dataset_dir}")
        sys.exit(1)

    analyzer = RoboflowDatasetAnalyzer(dataset_dir, output_dir)
    analyzer.load_yaml_config()
    
    # Task 4: Structure & Annotation Audit
    annotation_summary = analyzer.audit_structure_and_annotations()
    
    # Task 5: Class Analysis
    class_stats = analyzer.analyze_classes()
    
    # Task 6: Visual Annotation Validation
    visual_results = analyzer.validate_visual_annotations(samples_per_class=3)
    
    # Task 7: Image Quality & Blur/Exposure
    img_stats = analyzer.analyze_image_quality()
    
    # Task 8: Data Leakage Detection
    leakage_report = analyzer.detect_cross_split_leakage(hamming_threshold=4)

    print("\n" + "=" * 60)
    print("[+] PHASE 2A DATASET ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total Images Audited: {annotation_summary['total_images']}")
    print(f"Total Labels Audited: {annotation_summary['total_labels']}")
    print(f"Total Bounding Boxes: {annotation_summary['total_bounding_boxes']}")
    print(f"Malformed Annotations: {annotation_summary['malformed_annotations']}")
    print(f"Exact Cross-Split Leaks: {len(leakage_report['exact_cross_split_duplicates'])}")
    print(f"Near-Duplicate Burst Leaks: {len(leakage_report['near_duplicate_cross_split_pairs'])}")
    print("=" * 60)

if __name__ == "__main__":
    main()
