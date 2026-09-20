"""
Roboflow Dataset Acquisition and Ingestion Script.
Supports automated download via Roboflow API key or automated extraction
of a manually exported ZIP archive into data/raw/roboflow_coconut_detection/.
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ROBOFLOW_DIR = BASE_DIR / "data" / "raw" / "roboflow_coconut_detection"

def download_via_api(api_key: str, version: int = 1):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[-] The 'roboflow' package is required for direct API export. Install with: pip install roboflow")
        sys.exit(1)

    print(f"[*] Initializing Roboflow API download for workspace 'phanidhar-reddy', project 'coconut-tree-disease-vg85j' (version {version})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("phanidhar-reddy").project("coconut-tree-disease-vg85j")
    version_obj = project.version(version)
    
    # Download in yolov8 format into target directory
    RAW_ROBOFLOW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = version_obj.download("yolov8", location=str(RAW_ROBOFLOW_DIR))
    print(f"[+] Successfully downloaded Roboflow dataset to: {dataset.location}")

def extract_manual_zip(zip_path: Path):
    if not zip_path.exists():
        print(f"[-] Specified ZIP archive not found: {zip_path}")
        sys.exit(1)

    print(f"[*] Extracting {zip_path.name} to {RAW_ROBOFLOW_DIR}...")
    RAW_ROBOFLOW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_ROBOFLOW_DIR)
    print(f"[+] Successfully extracted archive to: {RAW_ROBOFLOW_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Acquire or extract the Roboflow Coconut Tree Disease Dataset.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--zip", type=str, help="Path to manually downloaded Roboflow ZIP archive")
    parser.add_argument("--version", type=int, default=1, help="Dataset version (default: 1)")
    args = parser.parse_args()

    if args.zip:
        extract_manual_zip(Path(args.zip))
    elif args.api_key:
        download_via_api(args.api_key, args.version)
    else:
        # Check if a zip file already exists in data/raw/
        raw_zips = list((BASE_DIR / "data" / "raw").glob("*coconut*tree*disease*.zip")) + list((BASE_DIR / "data" / "raw").glob("*.zip"))
        if raw_zips:
            print(f"[*] Found local archive in data/raw/: {raw_zips[0].name}")
            extract_manual_zip(raw_zips[0])
        else:
            print("=" * 78)
            print("[-] AUTHENTICATION REQUIRED FOR ROBOFLOW DIRECT DOWNLOAD")
            print("=" * 78)
            print("Roboflow Universe requires authentication to download datasets.")
            print("You have two methods to provide the dataset:\n")
            print("METHOD 1: Manual Browser Export (No coding required)")
            print("1. Open: https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j")
            print("2. Click on the 'Dataset' tab (Version 1).")
            print("3. Click the 'Download Dataset' button.")
            print("4. Select 'YOLOv8' format and click 'Download as ZIP'.")
            print("5. Save the downloaded ZIP file into:")
            print(f"   {BASE_DIR / 'data' / 'raw'}")
            print("6. Re-run this script: python scripts/download_roboflow.py\n")
            print("METHOD 2: Automated Download via API Key")
            print("Run: python scripts/download_roboflow.py --api-key <YOUR_ROBOFLOW_API_KEY>\n")
            print("=" * 78)

if __name__ == "__main__":
    main()
