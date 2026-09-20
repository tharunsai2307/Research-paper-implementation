"""
CLI script to execute full dataset audit on raw and processed datasets.
Outputs machine-readable JSON, CSV, and human-readable Markdown reports.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset.audit import DatasetAuditor

def main():
    target_dir = BASE_DIR / "data" / "raw" / "mendeley_coconut_disease"
    output_dir = BASE_DIR / "outputs" / "dataset_analysis"
    markdown_report = BASE_DIR / "research" / "dataset_audit.md"

    if not target_dir.exists():
        print(f"[-] Target dataset directory does not exist: {target_dir}")
        sys.exit(1)

    auditor = DatasetAuditor(target_dir)
    summary = auditor.audit()
    auditor.export_reports(output_dir, markdown_report)

    print("\n[+] Audit completed successfully!")
    print(f"    - Total images: {summary.get('total_images')}")
    print(f"    - Valid images: {summary.get('valid_images')}")
    print(f"    - Corrupted images: {summary.get('corrupted_images')}")
    print(f"    - Classes: {list(summary.get('class_distribution', {}).keys())}")

if __name__ == "__main__":
    main()
