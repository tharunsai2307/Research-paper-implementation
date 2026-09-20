"""
CLI script to detect data leakage and near-duplicates across dataset splits and classes.
Exports outputs/dataset_analysis/leakage_report.json.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset.leakage import LeakageDetector

def main():
    target_dir = BASE_DIR / "data" / "raw" / "mendeley_coconut_disease"
    output_path = BASE_DIR / "outputs" / "dataset_analysis" / "leakage_report.json"

    if not target_dir.exists():
        print(f"[-] Target dataset directory does not exist: {target_dir}")
        sys.exit(1)

    detector = LeakageDetector(target_dir)
    report = detector.check_leakage(hamming_threshold=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Leakage detection report saved: {output_path}")
    print(f"    - Total images analyzed: {report.get('total_images_analyzed')}")
    print(f"    - Exact duplicate groups: {report.get('exact_duplicate_groups')}")
    print(f"    - Near-duplicate pairs: {report.get('near_duplicate_pairs')}")
    print(f"    - Cross-class leak groups: {report.get('cross_class_leakage_groups')}")

if __name__ == "__main__":
    main()
