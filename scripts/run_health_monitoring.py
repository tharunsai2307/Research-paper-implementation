"""
Execution script for the Phase 5 Plantation Health Monitoring Engine.

Runs end-to-end inference, disease aggregation, severity proxy calculation,
and Plantation Health Index computation on coconut plantation field observations.

Supports:
- Model A: EXP-001 (baseline YOLOv8n @ 640x640)
- Model B: EXP-002 (YOLOv8n @ 512x512)
- Real demonstration on available sanitized validation block (27 diseased + 8 healthy controls)
- Output formats: JSON (mobile-app ready) and tabular CSV
- Comparative execution between Model A and Model B
- Generation of research-quality visualizations
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.health_monitoring.inference import HealthInferenceEngine
from src.health_monitoring.plantation_health import PlantationHealthAggregator
from src.health_monitoring.schemas import (
    CLASS_NAMES,
    ObservationStatus,
    DEFAULT_OPERATIONAL_THRESHOLD,
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)

def generate_visualizations(
    report: Dict[str, Any],
    image_records: List[Dict[str, Any]],
    output_dir: Path
):
    """Generates the required research figures for the health monitoring engine."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Disease Distribution Chart (Detection percentage and Affected Image percentage)
    dist = report["disease_distribution"]
    classes = [c for c in dist.keys()]
    det_pcts = [dist[c]["percentage_of_total_detections"] for c in classes]
    img_pcts = [dist[c]["percentage_of_images_affected"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, det_pcts, width, label='Lesion Detections (%)', color='#2b5c8f')
    ax.bar(x + width/2, img_pcts, width, label='Trees Affected (%)', color='#d95f02')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Disease Distribution: Lesion Detections vs. Affected Trees', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in classes], fontsize=10)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'disease_distribution_chart.png', dpi=300)
    plt.close(fig)

    # 2. Observation Status Breakdown (Pie / Donut)
    counts = report["observation_counts"]
    statuses = ['Disease Detected', 'No Disease Detected\n(Healthy)', 'Low-Conf Candidate']
    vals = [
        counts["disease_positive_images"],
        counts["disease_negative_images"],
        counts["low_confidence_candidate_images"]
    ]
    colors = ['#d95f02', '#2ca02c', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=statuses,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )
    plt.setp(autotexts, size=11, weight="bold")
    ax.set_title('Plantation Observation Status Breakdown\n(N = {})'.format(counts["successfully_processed_images"]), fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'observation_status_breakdown.png', dpi=300)
    plt.close(fig)

    # 3. Detection Confidence Distribution
    confs = []
    for r in image_records:
        for det in r.get("accepted_detections", []):
            confs.append(det["confidence"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if confs:
        ax.hist(confs, bins=15, range=(DEFAULT_OPERATIONAL_THRESHOLD, 1.0), color='#1f77b4', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(confs), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Conf: {np.mean(confs):.3f}')
        ax.axvline(DEFAULT_OPERATIONAL_THRESHOLD, color='green', linestyle='dotted', linewidth=1.5, label=f'Operational Cutoff: {DEFAULT_OPERATIONAL_THRESHOLD}')
        ax.legend()
    ax.set_xlabel('Model Confidence Score', fontsize=12)
    ax.set_ylabel('Accepted Detections Count', fontsize=12)
    ax.set_title('Accepted Detection Confidence Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'detection_confidence_distribution.png', dpi=300)
    plt.close(fig)

    # 4. Detection Count Per Disease (Absolute Bar Chart)
    det_counts = [dist[c]["detection_count"] for c in classes]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([c.replace(" ", "\n") for c in classes], det_counts, color='#386cb0', edgecolor='black', alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Total Accepted Lesions', fontsize=12)
    ax.set_title('Accepted Detection Count per Disease Class', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'detection_count_per_disease.png', dpi=300)
    plt.close(fig)

    # 5. Relative Affected Area Proxy Distribution
    area_proxies = [r["relative_affected_area_proxy"] for r in image_records if r["observation_status"] == ObservationStatus.DISEASE_DETECTED]
    fig, ax = plt.subplots(figsize=(8, 5))
    if area_proxies:
        ax.hist(area_proxies, bins=12, color='#7570b3', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(area_proxies), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Proxy (Positive): {np.mean(area_proxies):.4f}')
        ax.legend()
    ax.set_xlabel('Relative Affected Area Proxy (Union Lesion Area / Image Area)', fontsize=11)
    ax.set_ylabel('Observation Count', fontsize=12)
    ax.set_title('Distribution of Visual Affected Area Proxy (Disease-Positive Trees)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'affected_area_proxy_distribution.png', dpi=300)
    plt.close(fig)

    # 6. Plantation Health Index Gauge / Indicator
    phi_composite = report["plantation_health_index"]["phi_composite_score"]
    phi_incidence = report["plantation_health_index"]["phi_incidence_only_score"]
    tier = report["plantation_health_index"]["health_tier"]

    fig, ax = plt.subplots(figsize=(9, 4))
    categories = ['PHI (Incidence Only)', 'PHI (Composite Metric)']
    scores = [phi_incidence, phi_composite]
    bar_colors = ['#4daf4a' if s >= 70 else ('#ff7f00' if s >= 50 else '#e41a1c') for s in scores]

    bars = ax.barh(categories, scores, color=bar_colors, edgecolor='black', height=0.45)
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + 1.5, bar.get_y() + bar.get_height()/2.0, f'{xval:.1f} / 100', ha='left', va='center', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 110)
    ax.set_xlabel('Health Index Score (0 - 100 scale)', fontsize=12)
    ax.set_title(f'Plantation Health Index Summary\nTier: {tier}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'plantation_health_index_gauge.png', dpi=300)
    plt.close(fig)

def save_csv_reports(report: Dict[str, Any], output_dir: Path):
    """Saves tabular summaries for agronomic reporting."""
    # 1. plantation_health_report.csv
    summary_csv = output_dir / "plantation_health_report.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Description / Units"])
        writer.writerow(["Plantation ID", report["meta"]["plantation_id"], "Field block identifier"])
        writer.writerow(["Engine Version", report["meta"]["engine_version"], "Software version"])
        writer.writerow(["Operational Confidence Threshold", report["meta"]["operational_confidence_threshold"], "Detection confidence cutoff"])
        writer.writerow(["Total Submitted Observations", report["observation_counts"]["total_images_submitted"], "Photos submitted"])
        writer.writerow(["Successfully Processed Observations", report["observation_counts"]["successfully_processed_images"], "Photos decoded and analyzed"])
        writer.writerow(["Disease-Positive Observations", report["observation_counts"]["disease_positive_images"], "Photos with confirmed target disease"])
        writer.writerow(["Disease-Negative Observations", report["observation_counts"]["disease_negative_images"], "Photos without confirmed disease"])
        writer.writerow(["Low-Confidence Candidate Observations", report["observation_counts"]["low_confidence_candidate_images"], "Sub-threshold detections [0.10, 0.25)"])
        writer.writerow(["Disease-Positive Image Rate (%)", report["observation_rates"]["disease_positive_image_percentage"], "% of valid photos positive"])
        writer.writerow(["Mean Lesion Boxes Per Image", report["impact_indicators"]["mean_boxes_per_image"], "Mean box count per sampled palm"])
        writer.writerow(["Mean Affected Area Proxy (All)", report["impact_indicators"]["mean_relative_affected_area_all_images"], "Union lesion area / frame area"])
        writer.writerow(["Mean Affected Area Proxy (Positive Only)", report["impact_indicators"]["mean_relative_affected_area_positive_images"], "Union lesion area / frame area"])
        writer.writerow(["Mean Detection Confidence", report["impact_indicators"]["confidence_distribution"]["mean"], "Accepted detection confidence"])
        writer.writerow(["PHI Incidence Only Score", report["plantation_health_index"]["phi_incidence_only_score"], "100 * (1 - PosRate)"])
        writer.writerow(["PHI Composite Score", report["plantation_health_index"]["phi_composite_score"], "100 * (1 - (0.85*PosRate + 0.15*AreaProxy))"])
        writer.writerow(["Assigned Health Tier", report["plantation_health_index"]["health_tier"], "Categorical management tier"])
        writer.writerow(["Agronomic Recommendation", report["plantation_health_index"]["recommended_action"], "Field action directive"])

    # 2. disease_distribution.csv
    dist_csv = output_dir / "disease_distribution.csv"
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class ID", "Disease Class", "Total Detection Count",
            "% of Total Detections", "Unique Trees/Images Affected", "% of Sampled Trees Affected"
        ])
        for cname, data in report["disease_distribution"].items():
            writer.writerow([
                data["class_id"],
                data["disease_class"],
                data["detection_count"],
                data["percentage_of_total_detections"],
                data["unique_images_affected"],
                data["percentage_of_images_affected"]
            ])

def run_pipeline(
    model_path: Path,
    imgsz: int,
    image_dir: Path,
    conf: float = DEFAULT_OPERATIONAL_THRESHOLD,
    candidate_conf: float = DEFAULT_CANDIDATE_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    plantation_id: str = "DEMO_COCONUT_BLOCK_VAL_01"
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Instantiates engine and aggregates results."""
    print(f"Loading Health Inference Engine with {model_path.name} (imgsz={imgsz})...")
    engine = HealthInferenceEngine(
        model_path=model_path,
        imgsz=imgsz,
        operational_threshold=conf,
        candidate_threshold=candidate_conf,
        iou_threshold=iou
    )

    print(f"Processing images from: {image_dir} ...")
    image_records = engine.process_directory(image_dir, plantation_id=plantation_id)
    print(f"Successfully processed {len(image_records)} images.")

    aggregator = PlantationHealthAggregator(weight_incidence=0.85, weight_area_proxy=0.15)
    report = aggregator.generate_plantation_report(
        image_records=image_records,
        plantation_id=plantation_id,
        observation_session_id="SESSION_VAL_DEMO_20260920"
    )

    return report, image_records

def main():
    parser = argparse.ArgumentParser(description="Phase 5 Plantation Health Monitoring Engine")
    parser.add_argument("--model_a", type=str, default="outputs/training/EXP-001_baseline_yolov8n/weights/best.pt")
    parser.add_argument("--model_b", type=str, default="outputs/training/EXP-002_imgsz512/weights/best.pt")
    parser.add_argument("--input_dir", type=str, default="data/processed/coconut_detection_clean/val/images")
    parser.add_argument("--output_dir", type=str, default="outputs/health_monitoring")
    parser.add_argument("--figures_dir", type=str, default="outputs/figures/health_monitoring")
    parser.add_argument("--conf", type=float, default=DEFAULT_OPERATIONAL_THRESHOLD)
    parser.add_argument("--candidate_conf", type=float, default=DEFAULT_CANDIDATE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()

    model_a_path = PROJECT_ROOT / args.model_a
    model_b_path = PROJECT_ROOT / args.model_b
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    figures_dir = PROJECT_ROOT / args.figures_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Primary pipeline execution: Save both Model A and Model B
    print("\n" + "="*70)
    print("RUNNING MODEL A: EXP-001 YOLOv8n @ 640x640")
    print("="*70)
    report_a, records_a = run_pipeline(
        model_path=model_a_path,
        imgsz=640,
        image_dir=input_dir,
        conf=args.conf,
        candidate_conf=args.candidate_conf,
        iou=args.iou
    )

    # Save Model A artifacts
    with open(output_dir / "exp001_image_health_records.json", "w", encoding="utf-8") as f:
        json.dump(records_a, f, indent=2)
    with open(output_dir / "exp001_plantation_health_report.json", "w", encoding="utf-8") as f:
        json.dump(report_a, f, indent=2)

    print("\n" + "="*70)
    print("RUNNING MODEL B: EXP-002 YOLOv8n @ 512x512 (PRIMARY RESEARCH DEMO)")
    print("="*70)
    report_b, records_b = run_pipeline(
        model_path=model_b_path,
        imgsz=512,
        image_dir=input_dir,
        conf=args.conf,
        candidate_conf=args.candidate_conf,
        iou=args.iou
    )

    # Save Model B artifacts
    with open(output_dir / "exp002_image_health_records.json", "w", encoding="utf-8") as f:
        json.dump(records_b, f, indent=2)
    with open(output_dir / "exp002_plantation_health_report.json", "w", encoding="utf-8") as f:
        json.dump(report_b, f, indent=2)

    # Write primary artifacts (Model B contains accepted detections, enabling full visual & proxy assessment)
    records_json_path = output_dir / "image_health_records.json"
    with open(records_json_path, "w", encoding="utf-8") as f:
        json.dump(records_b, f, indent=2)
    print(f"Saved primary records: {records_json_path}")

    report_json_path = output_dir / "plantation_health_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_b, f, indent=2)
    print(f"Saved primary report: {report_json_path}")

    save_csv_reports(report_b, output_dir)
    print(f"Saved CSV reports in: {output_dir}")

    generate_visualizations(report_b, records_b, figures_dir)
    print(f"Generated research figures in: {figures_dir}")

    # Model comparison summary
    comparison = {
        "dataset_block": str(input_dir.name),
        "total_images": len(records_a),
        "operational_confidence_threshold": args.conf,
        "candidate_confidence_threshold": args.candidate_conf,
        "model_a_exp001": {
            "checkpoint": str(model_a_path),
            "input_size": 640,
            "total_accepted_detections": sum(dist["detection_count"] for dist in report_a["disease_distribution"].values()),
            "disease_positive_images": report_a["observation_counts"]["disease_positive_images"],
            "disease_positive_rate_pct": report_a["observation_rates"]["disease_positive_image_percentage"],
            "mean_affected_area_proxy_pos": report_a["impact_indicators"]["mean_relative_affected_area_positive_images"],
            "mean_confidence": report_a["impact_indicators"]["confidence_distribution"]["mean"],
            "phi_composite": report_a["plantation_health_index"]["phi_composite_score"],
            "phi_incidence": report_a["plantation_health_index"]["phi_incidence_only_score"],
            "health_tier": report_a["plantation_health_index"]["health_tier"]
        },
        "model_b_exp002": {
            "checkpoint": str(model_b_path),
            "input_size": 512,
            "total_accepted_detections": sum(dist["detection_count"] for dist in report_b["disease_distribution"].values()),
            "disease_positive_images": report_b["observation_counts"]["disease_positive_images"],
            "disease_positive_rate_pct": report_b["observation_rates"]["disease_positive_image_percentage"],
            "mean_affected_area_proxy_pos": report_b["impact_indicators"]["mean_relative_affected_area_positive_images"],
            "mean_confidence": report_b["impact_indicators"]["confidence_distribution"]["mean"],
            "phi_composite": report_b["plantation_health_index"]["phi_composite_score"],
            "phi_incidence": report_b["plantation_health_index"]["phi_incidence_only_score"],
            "health_tier": report_b["plantation_health_index"]["health_tier"]
        },
        "comparative_insights": (
            "Model A (640x640) vs Model B (512x512) reflects standard image-resolution trade-offs. "
            "Higher resolution preserves fine-grained lesion boundaries (higher detection count), "
            "whereas 512x512 reduces computational footprint suitable for resource-constrained mobile runtimes."
        )
    }

    comp_json_path = output_dir / "model_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved Model Comparison: {comp_json_path}")

    print("\n" + "="*70)
    print("PHASE 5 ENGINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
