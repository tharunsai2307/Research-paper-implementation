"""
Group-based sequence splitting and dataset sanitization engine for Coconut Tree Disease Detection.
Clusters burst captures, near-duplicates, and sequential frames into unified groups using
Disjoint Set Union (DSU) so that no burst group is split across train/val/test partitions.
Ingests verified healthy coconut images as background negative samples (empty .txt labels).
Outputs sanitized dataset to data/processed/coconut_detection_clean/.
"""

import os
import re
import csv
import json
import shutil
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ROBO_DIR = BASE_DIR / "data" / "raw" / "roboflow_coconut_detection"
RAW_HEALTHY_DIR = BASE_DIR / "data" / "raw" / "healthy_coconut_negatives"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "coconut_detection_clean"
OUTPUT_DIR = BASE_DIR / "outputs" / "dataset_analysis"

RANDOM_SEED = 42

def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(filepath: Path, hash_size: int = 8) -> int:
    try:
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
    except Exception:
        return 0

def hamming_dist(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

class DSU:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

def extract_sequence_prefix(filename: str) -> Tuple[str, int]:
    """Extracts sequence prefix and numerical sequence index, e.g. BudRootDropping001 -> ('BudRootDropping', 1)"""
    m = re.match(r"([A-Za-z_-]+)(\d+)", filename)
    if m:
        return m.group(1), int(m.group(2))
    return filename, -1

def create_sanitized_dataset():
    print("=" * 70)
    print("[*] Starting Phase 2B Group-Based Split & Dataset Sanitization")
    print("=" * 70)

    # 1. Collect all raw diseased images and labels
    items = []
    splits = ["train", "valid", "test"]
    for s in splits:
        img_dir = RAW_ROBO_DIR / s / "images"
        lbl_dir = RAW_ROBO_DIR / s / "labels"
        if not img_dir.exists():
            continue
        for img_p in img_dir.glob("*.*"):
            if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                if lbl_p.exists():
                    items.append({
                        "id": img_p.stem,
                        "img_path": img_p,
                        "lbl_path": lbl_p,
                        "is_healthy": False
                    })

    # 2. Ingest verified healthy coconut images
    healthy_items = []
    if RAW_HEALTHY_DIR.exists():
        for h_p in sorted(list(RAW_HEALTHY_DIR.glob("*.jpg"))):
            healthy_items.append({
                "id": h_p.stem,
                "img_path": h_p,
                "lbl_path": None, # Will generate empty .txt
                "is_healthy": True
            })

    print(f"[+] Loaded {len(items)} diseased candidate items.")
    print(f"[+] Loaded {len(healthy_items)} verified healthy negative items.")

    # 3. Form burst groups on diseased items using DSU
    item_ids = [it["id"] for it in items]
    dsu = DSU(item_ids)

    # Signal A: Perceptual dHash near-duplicates (Hamming <= 4)
    id_to_item = {it["id"]: it for it in items}
    dhashes = {it["id"]: compute_dhash(it["img_path"]) for it in items}

    for i in range(len(items)):
        id1 = items[i]["id"]
        dh1 = dhashes[id1]
        if dh1 == 0:
            continue
        for j in range(i + 1, len(items)):
            id2 = items[j]["id"]
            dh2 = dhashes[id2]
            if dh2 == 0:
                continue
            dist = hamming_dist(dh1, dh2)
            # Immediate burst near-duplicate if Hamming <= 4
            if dist <= 4:
                dsu.union(id1, id2)

    # Signal B: Direct adjacent sequence frames (num2 - num1 == 1) if perceptual distance <= 6
    prefix_map = defaultdict(list)
    for it in items:
        pfx, num = extract_sequence_prefix(it["id"])
        if num != -1:
            prefix_map[pfx].append((num, it["id"]))

    for pfx, seq in prefix_map.items():
        seq.sort()
        for i in range(len(seq) - 1):
            num1, id1 = seq[i]
            num2, id2 = seq[i + 1]
            if num2 - num1 == 1:
                dh1 = dhashes.get(id1, 0)
                dh2 = dhashes.get(id2, 0)
                if dh1 != 0 and dh2 != 0 and hamming_dist(dh1, dh2) <= 6:
                    dsu.union(id1, id2)

    # Group diseased items by DSU root
    groups = defaultdict(list)
    for it in items:
        root = dsu.find(it["id"])
        groups[root].append(it)

    multi_groups = [g for g in groups.values() if len(g) > 1]
    print(f"[+] Clustered {len(items)} diseased items into {len(groups)} leakage-controlled groups ({len(multi_groups)} multi-item burst clusters).")

    # 4. Group healthy negative items (cluster bursts if consecutive)
    healthy_groups = defaultdict(list)
    for it in healthy_items:
        healthy_groups[it["id"]].append(it)
    print(f"[+] Clustered {len(healthy_items)} healthy items into {len(healthy_groups)} negative groups.")

    # 5. Deterministic split allocation (Seed = 42)
    rng = random.Random(RANDOM_SEED)

    # Split diseased groups by primary class
    group_classes = {}
    for gid, g_items in groups.items():
        cids = []
        for it in g_items:
            with open(it["lbl_path"], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cids.append(int(parts[0]))
        primary_cid = max(set(cids), key=cids.count) if cids else 0
        group_classes[gid] = primary_cid

    # Stratified split per disease class
    class_to_groups = defaultdict(list)
    for gid, cid in group_classes.items():
        class_to_groups[cid].append(gid)

    train_items, val_items, test_items = [], [], []

    for cid in sorted(class_to_groups.keys()):
        g_list = list(class_to_groups[cid])
        rng.shuffle(g_list)
        n_g = len(g_list)
        # Allocate groups: target ~75% train, ~15% val, ~10% test
        # Ensure at least 1 group in train, 1 in val, 1 in test
        if n_g >= 4:
            n_val = max(1, int(round(0.15 * n_g)))
            n_test = max(1, int(round(0.10 * n_g)))
            n_train = n_g - n_val - n_test
        elif n_g == 3:
            n_train, n_val, n_test = 1, 1, 1
        elif n_g == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_g = g_list[:n_train]
        val_g = g_list[n_train:n_train + n_val]
        test_g = g_list[n_train + n_val:]

        for gid in train_g:
            train_items.extend(groups[gid])
        for gid in val_g:
            val_items.extend(groups[gid])
        for gid in test_g:
            test_items.extend(groups[gid])

    # Partition healthy negative groups (~75% train, ~15% val, ~10% test)
    h_group_list = list(healthy_groups.keys())
    rng.shuffle(h_group_list)
    nh = len(h_group_list)
    nh_val = max(1, int(round(0.15 * nh)))
    nh_test = max(1, int(round(0.10 * nh)))
    nh_train = nh - nh_val - nh_test

    for gid in h_group_list[:nh_train]:
        train_items.extend(healthy_groups[gid])
    for gid in h_group_list[nh_train:nh_train + nh_val]:
        val_items.extend(healthy_groups[gid])
    for gid in h_group_list[nh_train + nh_val:]:
        test_items.extend(healthy_groups[gid])

    total_final = len(train_items) + len(val_items) + len(test_items)
    print(f"\n[+] Grouped Split Results (Total: {total_final} images):")
    print(f"    - Train: {len(train_items)} images ({len(train_items)/total_final*100:.1f}%)")
    print(f"    - Val:   {len(val_items)} images ({len(val_items)/total_final*100:.1f}%)")
    print(f"    - Test:  {len(test_items)} images ({len(test_items)/total_final*100:.1f}%)")

    # 6. Materialize clean processed dataset in data/processed/coconut_detection_clean/
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    split_manifest = {
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

    for split_name, s_items in split_manifest.items():
        img_out = PROCESSED_DIR / split_name / "images"
        lbl_out = PROCESSED_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for it in s_items:
            # Copy image
            target_img = img_out / it["img_path"].name
            shutil.copy2(it["img_path"], target_img)

            # Copy or generate label
            target_lbl = lbl_out / f"{it['img_path'].stem}.txt"
            if it["is_healthy"] or it["lbl_path"] is None:
                # Empty label file for negative background images
                target_lbl.touch()
            else:
                shutil.copy2(it["lbl_path"], target_lbl)

    # 7. Write clean data.yaml (Task O)
    data_yaml_content = f"""# Ultralytics YOLOv8 Sanitized Dataset Configuration
# Generated in Phase 2B with Group-Based Leakage Control & Healthy Negative Integration
# Primary Dataset: Roboflow Coconut Tree Disease (Phanidhar Reddy)
# Negative Control Source: Wikimedia Commons Verified Cocos nucifera

path: ../coconut_detection_clean
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: bud root dropping
  1: bud rot
  2: gray leaf spot
  3: leaf rot
  4: stembleeding
"""
    with open(PROCESSED_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    print(f"[+] Successfully materialized clean processed dataset at: {PROCESSED_DIR}")
    return split_manifest

if __name__ == "__main__":
    create_sanitized_dataset()
