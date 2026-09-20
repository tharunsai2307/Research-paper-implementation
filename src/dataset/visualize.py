"""
Visual dataset inspection module for Coconut Tree Disease Detection.
Generates representative class grids and overlays bounding-box ground truth
without altering raw files.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class DatasetVisualizer:
    def __init__(self, dataset_path: str | Path, output_dir: str | Path):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_class_grid(self, samples_per_class: int = 3, filename: str = "representative_class_grid.png") -> Path:
        """Generates a multi-class comparison grid showing authentic disease samples."""
        print(f"[*] Generating representative class grid ({samples_per_class} per class)...")
        class_folders = [d for d in self.dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        
        if not class_folders:
            raise ValueError(f"No class subdirectories found in {self.dataset_path}")

        n_classes = len(class_folders)
        fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(samples_per_class * 4, n_classes * 3.5))
        if n_classes == 1:
            axes = np.expand_dims(axes, axis=0)
        if samples_per_class == 1:
            axes = np.expand_dims(axes, axis=1)

        for row_idx, cls_dir in enumerate(class_folders):
            cls_name = cls_dir.name
            img_files = sorted([f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS])
            selected = img_files[:samples_per_class]

            for col_idx in range(samples_per_class):
                ax = axes[row_idx, col_idx]
                if col_idx < len(selected):
                    img_path = selected[col_idx]
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, _ = img_rgb.shape
                        ax.imshow(img_rgb)
                        ax.set_title(f"{cls_name}\n({img_path.name})\n{w}x{h}", fontsize=9)
                    else:
                        ax.text(0.5, 0.5, "Unreadable Image", ha="center", va="center")
                else:
                    ax.text(0.5, 0.5, "No Image", ha="center", va="center")
                ax.axis("off")

        plt.suptitle("Representative Coconut Tree Disease Classes (Raw Field Imagery)", fontsize=14, y=0.99)
        plt.tight_layout()
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Saved representative grid to {save_path}")
        return save_path

    def visualize_bounding_boxes(self, img_path: Path, label_path: Path, class_names: Optional[List[str]] = None, save_name: str = "bbox_sample.png") -> Optional[Path]:
        """Draws YOLO bounding box overlays and saves visual verification output."""
        img = cv2.imread(str(img_path))
        if img is None or not label_path.exists():
            return None

        h, w, _ = img.shape
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            label_str = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label_str, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        save_path = self.output_dir / save_name
        cv2.imwrite(str(save_path), img)
        return save_path
