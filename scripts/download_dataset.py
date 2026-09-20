"""
Dataset acquisition and provenance script for Coconut Tree Disease Detection.
Downloads open benchmark images, computes cryptographic SHA-256 checksums,
and updates research/dataset_manifest.csv.
"""

import os
import sys
import hashlib
import urllib.request
import csv
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MANIFEST_PATH = BASE_DIR / "research" / "dataset_manifest.csv"

# Representative verified files from each class in Mendeley Data / Patil et al. (2023) open mirror
CLASSES = {
    "Bud Root Dropping": [f"BudRootDropping{i:03d}.jpg" for i in range(1, 21)],
    "Bud Rot": [f"BudRot{i:03d}.jpg" for i in range(1, 21)],
    "Gray Leaf Spot": [f"GrayLeafSpot{i:03d}.jpg" for i in range(1, 21)],
    "Leaf Rot": [f"LeafRot{i:03d}.jpg" for i in range(1, 21)],
    "Stem Bleeding": [f"StemBleeding{i:03d}.jpg" for i in range(1, 21)],
}

BASE_GITHUB_RAW = "https://raw.githubusercontent.com/zahid16-code/cocunut-disease-prediction/master/Coconut%20Tree%20Disease%20Dataset"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def ensure_manifest_header():
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset_name",
                "source_url",
                "license",
                "download_date",
                "file_name",
                "file_size",
                "sha256",
                "version",
                "notes"
            ])

def download_mendeley_sample(num_per_class: int = 20):
    print(f"[*] Downloading {num_per_class} sample images per class for technical audit...")
    dest_dir = RAW_DIR / "mendeley_coconut_disease"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    ensure_manifest_header()
    existing_entries = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 4:
                    existing_entries.add(row[4])

    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    new_rows = []

    for cls_name, files in CLASSES.items():
        cls_dir = dest_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        encoded_cls = urllib.parse.quote(cls_name)
        
        for fname in files[:num_per_class]:
            target_path = cls_dir / fname
            url = f"{BASE_GITHUB_RAW}/{encoded_cls}/{fname}"
            
            if not target_path.exists():
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        content = resp.read()
                        with open(target_path, "wb") as f:
                            f.write(content)
                except Exception as e:
                    print(f"[-] Failed to download {fname}: {e}")
                    continue
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                sha256_hash = compute_sha256(target_path)
                rel_filename = f"mendeley_coconut_disease/{cls_name}/{fname}"
                
                if rel_filename not in existing_entries:
                    new_rows.append([
                        "Coconut Tree Disease Dataset (Mendeley Data / Patil et al.)",
                        url,
                        "CC BY 4.0",
                        download_date,
                        rel_filename,
                        str(file_size),
                        sha256_hash,
                        "1.0",
                        f"Authentic raw sample: {cls_name}"
                    ])
                    existing_entries.add(rel_filename)
        print(f"[+] Processed class '{cls_name}'.")

    if new_rows:
        with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
        print(f"[+] Logged {len(new_rows)} files into {MANIFEST_PATH}")
    else:
        print("[*] Manifest up to date.")

if __name__ == "__main__":
    download_mendeley_sample()
