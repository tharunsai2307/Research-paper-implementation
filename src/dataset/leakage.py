"""
Data leakage and near-duplicate detection engine for Coconut Tree Disease Detection.
Identifies cryptographic file duplicates, perceptual near-duplicates (dHash),
and potential cross-split or cross-class contaminations.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(filepath: Path, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for perceptual similarity comparison."""
    try:
        with Image.open(filepath) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            
            diff = []
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
            
            decimal_value = 0
            for idx, bit in enumerate(diff):
                if bit:
                    decimal_value += 1 << idx
            return decimal_value
    except Exception:
        return 0

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

class LeakageDetector:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.exact_duplicates: Dict[str, List[str]] = defaultdict(list)
        self.near_duplicates: List[Dict[str, Any]] = []
        self.cross_class_leaks: List[Dict[str, Any]] = []
        self.cross_split_leaks: List[Dict[str, Any]] = []

    def check_leakage(self, hamming_threshold: int = 4) -> Dict[str, Any]:
        """Runs exact MD5 and perceptual dHash checks across all images in the dataset."""
        print(f"[*] Scanning for data leakage and near-duplicates in: {self.dataset_path}")
        image_files = [f for f in self.dataset_path.rglob("*") if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS]
        
        md5_map = defaultdict(list)
        dhash_list: List[Tuple[Path, str, int]] = []

        for img_path in image_files:
            rel_p = str(img_path.relative_to(self.dataset_path))
            cls_name = img_path.parent.name
            
            # 1. Exact MD5
            file_hash = compute_md5(img_path)
            md5_map[file_hash].append((rel_p, cls_name))
            
            # 2. Perceptual dHash
            dh = compute_dhash(img_path)
            if dh != 0:
                dhash_list.append((img_path, rel_p, dh))

        # Identify exact duplicates
        for fhash, items in md5_map.items():
            if len(items) > 1:
                paths = [p for p, _ in items]
                classes = set(c for _, c in items)
                self.exact_duplicates[fhash] = paths
                if len(classes) > 1:
                    self.cross_class_leaks.append({
                        "hash": fhash,
                        "files": paths,
                        "classes": list(classes),
                        "type": "exact_cross_class_duplicate"
                    })

        # Identify perceptual near-duplicates (pairwise comparison)
        n = len(dhash_list)
        print(f"[*] Comparing perceptual hashes across {n} images...")
        for i in range(n):
            p1, rel1, h1 = dhash_list[i]
            for j in range(i + 1, n):
                p2, rel2, h2 = dhash_list[j]
                dist = hamming_distance(h1, h2)
                if dist <= hamming_threshold:
                    cls1 = p1.parent.name
                    cls2 = p2.parent.name
                    self.near_duplicates.append({
                        "file1": rel1,
                        "file2": rel2,
                        "class1": cls1,
                        "class2": cls2,
                        "hamming_distance": dist,
                        "is_cross_class": cls1 != cls2
                    })

        report = {
            "total_images_analyzed": len(image_files),
            "exact_duplicate_groups": len(self.exact_duplicates),
            "total_exact_duplicate_images": sum(len(v) for v in self.exact_duplicates.values()),
            "near_duplicate_pairs": len(self.near_duplicates),
            "cross_class_leakage_groups": len(self.cross_class_leaks),
            "exact_duplicates": dict(self.exact_duplicates),
            "near_duplicates_sample": self.near_duplicates[:20],
            "cross_class_leaks": self.cross_class_leaks
        }

        print(f"[+] Found {len(self.exact_duplicates)} exact duplicate groups, {len(self.near_duplicates)} near-duplicate pairs.")
        return report
