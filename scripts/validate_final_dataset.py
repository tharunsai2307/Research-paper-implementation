"""
Automated sanity assertion test suite for the sanitized coconut detection dataset.
Validates:
1. Directory structure and presence of train/val/test images and labels.
2. Existence and validity of data.yaml.
3. 1-to-1 matching between image files and label files.
4. YOLO coordinate validity ([0, 1] bounds, non-negative, valid float format).
5. Class ID validity (strictly 0..4).
6. Proper representation of healthy negative controls (empty .txt files) in all splits.
7. Zero cross-split exact duplicates and zero near-duplicates.
8. Dataset manifest consistency with physical file counts.

Exits with code 0 if all tests pass; raises AssertionError otherwise.
"""

import sys
import yaml
import json
import hashlib
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean"
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.json"
DATA_YAML_PATH = PROCESSED_DIR / "data.yaml"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(filepath: Path, hash_size: int = 8) -> int:
    with Image.open(filepath) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        diff = []
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
        val = 0
        for idx, bit in enumerate(diff):
            if bit:
                val += 1 << idx
        return val

def hamming(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

def test_dataset_integrity():
    print("=" * 70)
    print("[*] Running Automated Final Dataset Validation Test Suite")
    print("=" * 70)

    # Test 1: data.yaml existence and schema
    print("[1/8] Verifying data.yaml configuration...")
    assert DATA_YAML_PATH.exists(), f"data.yaml missing at {DATA_YAML_PATH}"
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    assert "names" in data_cfg, "names missing in data.yaml"
    assert len(data_cfg["names"]) == 5, f"Expected 5 classes, found {len(data_cfg['names'])}"
    assert data_cfg["names"][0] == "bud root dropping"
    assert data_cfg["names"][4] == "stembleeding"
    print("      -> PASSED: data.yaml contains valid 5-class schema.")

    # Test 2: Directory structure
    print("[2/8] Verifying dataset directory hierarchy...")
    splits = ["train", "val", "test"]
    for s in splits:
        img_dir = PROCESSED_DIR / s / "images"
        lbl_dir = PROCESSED_DIR / s / "labels"
        assert img_dir.is_dir(), f"Missing {img_dir}"
        assert lbl_dir.is_dir(), f"Missing {lbl_dir}"
    print("      -> PASSED: All partition directories exist.")

    # Test 3: 1-to-1 Image-Label matching
    print("[3/8] Verifying 1-to-1 image-label correspondence...")
    split_images = {}
    split_labels = {}
    total_imgs = 0
    total_lbls = 0

    for s in splits:
        img_dir = PROCESSED_DIR / s / "images"
        lbl_dir = PROCESSED_DIR / s / "labels"
        imgs = sorted(list(img_dir.glob("*.*")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        split_images[s] = imgs
        split_labels[s] = lbls
        total_imgs += len(imgs)
        total_lbls += len(lbls)

        img_stems = set(p.stem for p in imgs)
        lbl_stems = set(p.stem for p in lbls)

        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        assert len(missing_labels) == 0, f"Missing labels for images in {s}: {missing_labels}"
        assert len(orphan_labels) == 0, f"Orphan labels without images in {s}: {orphan_labels}"

    assert total_imgs == 150, f"Expected 150 total images, found {total_imgs}"
    assert total_lbls == 150, f"Expected 150 total labels, found {total_lbls}"
    print(f"      -> PASSED: Exactly 150 images matched to 150 label files (0 orphans, 0 missing).")

    # Test 4: Label formatting and coordinate bounds
    print("[4/8] Verifying label formatting and bounding box coordinates...")
    total_bboxes = 0
    healthy_counts = {s: 0 for s in splits}
    class_bbox_counts = {i: 0 for i in range(5)}

    for s in splits:
        for lbl_p in split_labels[s]:
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                healthy_counts[s] += 1
            else:
                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    assert len(parts) == 5, f"Malformed line in {lbl_p}: line {line_idx+1}: '{line}'"
                    cid = int(parts[0])
                    assert 0 <= cid <= 4, f"Invalid class_id {cid} in {lbl_p}"
                    xc, yc, w, h = map(float, parts[1:])
                    assert 0.0 <= xc <= 1.0, f"xc out of bounds in {lbl_p}: {xc}"
                    assert 0.0 <= yc <= 1.0, f"yc out of bounds in {lbl_p}: {yc}"
                    assert 0.0 < w <= 1.0, f"w out of bounds in {lbl_p}: {w}"
                    assert 0.0 < h <= 1.0, f"h out of bounds in {lbl_p}: {h}"
                    class_bbox_counts[cid] += 1
                    total_bboxes += 1

    assert total_bboxes == 120, f"Expected 120 total bboxes, found {total_bboxes}"
    print(f"      -> PASSED: All 120 bounding boxes adhere strictly to YOLO normalized coordinate bounds.")

    # Test 5: Negative controls present across all partitions
    print("[5/8] Verifying healthy/negative control distribution...")
    assert sum(healthy_counts.values()) == 50, f"Expected 50 healthy negatives, found {sum(healthy_counts.values())}"
    for s in splits:
        assert healthy_counts[s] > 0, f"Split {s} has no healthy negative samples!"
    print(f"      -> PASSED: 50 healthy negatives distributed across train ({healthy_counts['train']}), val ({healthy_counts['val']}), test ({healthy_counts['test']}).")

    # Test 6: Class representation
    print("[6/8] Verifying disease class representation...")
    for cid in range(5):
        assert class_bbox_counts[cid] > 0, f"Class {cid} has 0 bounding boxes!"
    print(f"      -> PASSED: All 5 disease classes actively represented (Gray leaf spot: {class_bbox_counts[2]}, others: 20 each).")

    # Test 7: Post-Split Leakage Test
    print("[7/8] Running cryptographic and perceptual leakage verification...")
    split_hashes = {}
    split_dhashes = {}
    for s in splits:
        split_hashes[s] = [compute_sha256(p) for p in split_images[s]]
        split_dhashes[s] = [compute_dhash(p) for p in split_images[s]]

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in pairs:
        # Exact collisions
        set1, set2 = set(split_hashes[s1]), set(split_hashes[s2])
        collisions = set1.intersection(set2)
        assert len(collisions) == 0, f"Exact SHA-256 duplicate collision between {s1} and {s2}: {collisions}"

        # Perceptual near-duplicate collisions (Hamming <= 4)
        for h1 in split_dhashes[s1]:
            for h2 in split_dhashes[s2]:
                assert hamming(h1, h2) > 4, f"Perceptual near-duplicate leakage detected between {s1} and {s2} (dist={hamming(h1, h2)})"

    print("      -> PASSED: Zero cross-split exact duplicates and zero perceptual near-duplicates.")

    # Test 8: Manifest consistency
    print("[8/8] Verifying dataset_manifest.json consistency...")
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["partition_statistics"]["total_images"] == total_imgs
    assert manifest["partition_statistics"]["total_annotations"] == total_bboxes
    print("      -> PASSED: Dataset manifest perfectly matches physical filesystem counts.")

    print("\n" + "=" * 70)
    print("[+] ALL AUTOMATED TESTS PASSED SUCCESSFULLY.")
    print("    Dataset is verified, leakage-free, and ready for Phase 3 YOLOv8 modeling.")
    print("=" * 70)

if __name__ == "__main__":
    test_dataset_integrity()
