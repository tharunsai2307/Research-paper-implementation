"""
CLI script to generate visual inspection grids of authentic dataset samples.
Saves grids to outputs/dataset_analysis/sample_grids/.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset.visualize import DatasetVisualizer

def main():
    target_dir = BASE_DIR / "data" / "raw" / "mendeley_coconut_disease"
    output_dir = BASE_DIR / "outputs" / "dataset_analysis" / "sample_grids"

    if not target_dir.exists():
        print(f"[-] Target dataset directory does not exist: {target_dir}")
        sys.exit(1)

    visualizer = DatasetVisualizer(target_dir, output_dir)
    grid_path = visualizer.generate_class_grid(samples_per_class=4, filename="representative_class_grid.png")
    print(f"\n[+] Visual inspection grid generated: {grid_path}")

if __name__ == "__main__":
    main()
