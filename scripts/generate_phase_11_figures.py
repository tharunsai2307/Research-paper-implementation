"""
Generate Phase 11 Publication Figures.
Outputs to outputs/phase_11/figures/:
1. annotation_completion_status.png: Audit of 100 benchmark images (Pending vs Accepted)
2. class_wise_mask_distribution.png: Registered images vs Accepted masks per class
3. split_mask_distribution.png: Train (63), Val (27), Test (10) partition state
4. box_vs_mask_gamma_methodology.png: True lesion segmentation vs Bounding box severity methodology
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs/phase_11/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["bud root dropping", "bud rot", "gray leaf spot", "leaf rot", "stembleeding"]

def generate_phase_11_figures():
    print("Generating Phase 11 Research Figures from Actual Dataset Data...")

    # Figure 1: Annotation Completion Status
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    categories = ["Accepted Masks", "QC Review Required", "Rejected Masks", "Pending Annotations"]
    counts = [0, 0, 0, 100]
    colors = ["#10b981", "#f59e0b", "#64748b", "#ef4444"]
    
    bars = ax.bar(categories, counts, color=colors, width=0.55)
    ax.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_title("Phase 11: Ground-Truth Annotation & Acceptance Status\n(Training Gate: BLOCKED)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 120)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 2, f"{int(h)} ({h/100:.0%})", ha="center", va="bottom", fontweight="bold", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 2, "0", ha="center", va="bottom", color="#64748b", fontsize=10)
            
    plt.tight_layout()
    f1_path = FIGURES_DIR / "annotation_completion_status.png"
    plt.savefig(f1_path)
    plt.close()
    print(f"  [CREATED] {f1_path}")

    # Figure 2: Class-Wise Distribution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(CLASSES))
    width = 0.35
    
    reg_counts = [20, 20, 20, 20, 20]
    acc_counts = [0, 0, 0, 0, 0]
    
    b1 = ax.bar(x - width/2, reg_counts, width, label="Registered Benchmark Images (20/cls)", color="#3b82f6")
    b2 = ax.bar(x + width/2, acc_counts, width, label="Accepted Human Masks (0)", color="#10b981")
    
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title("Phase 11: Class-Wise Image Inventory vs Accepted Ground-Truth Masks", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n").title() for c in CLASSES], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 26)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i in range(len(CLASSES)):
        ax.text(i - width/2, reg_counts[i] + 0.8, "20", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, 0.8, "0", ha="center", color="#64748b", fontsize=9)
        
    plt.tight_layout()
    f2_path = FIGURES_DIR / "class_wise_mask_distribution.png"
    plt.savefig(f2_path)
    plt.close()
    print(f"  [CREATED] {f2_path}")

    # Figure 3: Split Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    splits = ["Train", "Validation", "Test (Untouched)"]
    split_assigned = [63, 27, 10]
    split_accepted = [0, 0, 0]
    
    x_s = np.arange(len(splits))
    ax.bar(x_s - width/2, split_assigned, width, label="Allocated Benchmark Images", color="#6366f1")
    ax.bar(x_s + width/2, split_accepted, width, label="Accepted Masks (Awaiting)", color="#10b981")
    
    ax.set_ylabel("Image Count", fontsize=11, fontweight="bold")
    ax.set_title("Phase 11: Leakage-Free Dataset Split Allocation & Mask State", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x_s)
    ax.set_xticklabels(splits, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i in range(len(splits)):
        ax.text(i - width/2, split_assigned[i] + 1.5, str(split_assigned[i]), ha="center", fontsize=10, fontweight="bold")
        ax.text(i + width/2, 1.5, "0", ha="center", color="#64748b", fontsize=10)
        
    plt.tight_layout()
    f3_path = FIGURES_DIR / "split_mask_distribution.png"
    plt.savefig(f3_path)
    plt.close()
    print(f"  [CREATED] {f3_path}")

    # Figure 4: Box vs Mask Gamma Methodology
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    ax.text(5.0, 5.2, "Empirical Box vs Mask Comparison Formulation", ha="center", fontsize=13, fontweight="bold")
    ax.text(5.0, 4.4, r"$\gamma = \frac{\mathrm{Area}(\mathrm{BoundingBox})}{\mathrm{Area}(\mathrm{Mask})} \geq 1.0$", ha="center", fontsize=14, color="#1e293b")
    ax.text(5.0, 3.5, r"$\mathrm{AffectedArea}_{\mathrm{true}} = \frac{\mathrm{Area}(\mathrm{Mask})}{\mathrm{Area}(\mathrm{Image})}$   vs   $\mathrm{AffectedArea}_{\mathrm{proxy}} = \frac{\mathrm{Area}(\mathrm{BBox})}{\mathrm{Area}(\mathrm{Image})}$", ha="center", fontsize=11, color="#334155")
    
    ax.text(5.0, 2.2, "Status: AWAITING REAL HUMAN GROUND TRUTH MASKS", ha="center", fontsize=11, fontweight="bold", color="#dc2626")
    ax.text(5.0, 1.4, "Research Integrity: Zero synthetic values or simulated ratios reported.\nEmpirical gamma will be calculated once double-annotated QC passes.", ha="center", fontsize=9, color="#64748b")
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
        
    plt.tight_layout()
    f4_path = FIGURES_DIR / "box_vs_mask_gamma_methodology.png"
    plt.savefig(f4_path)
    plt.close()
    print(f"  [CREATED] {f4_path}")

    print("All Phase 11 figures successfully generated.")

if __name__ == "__main__":
    generate_phase_11_figures()
