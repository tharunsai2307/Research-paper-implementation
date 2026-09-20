"""
Generate Phase 10 Figures and Visualizations.
Outputs to outputs/phase_10/figures/:
1. mask_status_breakdown.png: Status of 100 registered images across classes
2. split_distribution_phase10.png: Partitioning across Train (63), Val (27), Test (10)
3. box_vs_mask_inflation_concept.png: Diagram explaining bbox proxy vs visible polygon lesion area
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs/phase_10/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_figures():
    print("Generating Phase 10 Research Figures...")

    # Figure 1: Mask Status Breakdown
    classes = ["bud root dropping", "bud rot", "gray leaf spot", "leaf rot", "stembleeding"]
    completed = [0, 0, 0, 0, 0]
    pending = [20, 20, 20, 20, 20]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(classes))
    width = 0.5
    
    ax.bar(x, completed, width, label="Completed Masks (0)", color="#10b981")
    ax.bar(x, pending, width, bottom=completed, label="Pending Annotations (100)", color="#ef4444", alpha=0.85)
    
    ax.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
    ax.set_title("Phase 10: Ground-Truth Mask Annotation Status by Disease Class\n(Training Gate Status: BLOCKED)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n").title() for c in classes], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 25)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i in range(len(classes)):
        ax.text(i, 10, "20 Pending\n(Gate Blocked)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        
    plt.tight_layout()
    f1_path = FIGURES_DIR / "mask_status_breakdown.png"
    plt.savefig(f1_path)
    plt.close()
    print(f"  [CREATED] {f1_path}")

    # Figure 2: Split Distribution
    train_counts = [13, 13, 13, 12, 12]
    val_counts = [4, 5, 5, 7, 6]
    test_counts = [3, 2, 2, 1, 2]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x, train_counts, width=0.6, label="Train (63 / 63%)", color="#2563eb")
    ax.bar(x, val_counts, width=0.6, bottom=train_counts, label="Validation (27 / 27%)", color="#f59e0b")
    ax.bar(x, test_counts, width=0.6, bottom=np.array(train_counts) + np.array(val_counts), label="Test (10 / 10% - Untouched)", color="#10b981")
    
    ax.set_ylabel("Image Count", fontsize=12, fontweight="bold")
    ax.set_title("Phase 10: Leakage-Free Dataset Split Allocation Across Disease Classes", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n").title() for c in classes], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 24)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    f2_path = FIGURES_DIR / "split_distribution_phase10.png"
    plt.savefig(f2_path)
    plt.close()
    print(f"  [CREATED] {f2_path}")

    # Figure 3: Box vs Mask Inflation Concept
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Bounding Box (Phase 5 Proxy)
    bbox = patches.Rectangle((2, 2), 6, 6, linewidth=2.5, edgecolor="#ef4444", facecolor="none", linestyle="--", label="Detection Bounding Box (Phase 5 Proxy)")
    ax.add_patch(bbox)
    
    # Realistic lesion polygon
    poly_pts = np.array([
        [2.5, 3.0], [3.2, 4.5], [4.0, 5.8], [5.5, 6.8], [7.2, 6.2],
        [7.5, 4.8], [6.8, 3.5], [5.2, 2.4], [3.6, 2.2]
    ])
    polygon = patches.Polygon(poly_pts, closed=True, edgecolor="#059669", facecolor="#34d399", alpha=0.4, linewidth=2.5, label="Visible Lesion Boundary (Ground Truth Polygon)")
    ax.add_patch(polygon)
    
    ax.text(5.0, 9.2, "Box vs Mask Severity Area Comparison", ha="center", fontsize=14, fontweight="bold")
    ax.text(5.0, 8.6, r"$\gamma = \frac{\mathrm{Area}_{\mathrm{bbox}}}{\mathrm{Area}_{\mathrm{mask}}} \geq 1.0$ (Empirical Inflation Factor)", ha="center", fontsize=12, color="#1e293b")
    
    ax.text(2.2, 7.6, "BBox Area: $W \\times H$\n(Overestimates Necrosis)", color="#ef4444", fontsize=10, fontweight="bold")
    ax.text(5.0, 4.5, "True Necrotic Lesion\n(Actual Pathology Area)", ha="center", color="#065f46", fontsize=10, fontweight="bold")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    
    plt.tight_layout()
    f3_path = FIGURES_DIR / "box_vs_mask_inflation_concept.png"
    plt.savefig(f3_path)
    plt.close()
    print(f"  [CREATED] {f3_path}")
    print("All figures successfully generated.")

if __name__ == "__main__":
    generate_figures()
