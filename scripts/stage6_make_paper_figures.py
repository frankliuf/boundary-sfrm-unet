from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path("experiments/summaries")
OUT = Path("figures/paper1")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str):
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, text, fc="#FFFFFF", ec="#202020", lw=1.0, fontsize=8):
    rect = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fontsize)
    return rect


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10, linewidth=0.9, color="#202020"))


def draw_mini_prediction(ax, x, y, w, h, edge="#202020", accent="#5B2A86"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FAFAFA", edgecolor=edge, linewidth=0.7))
    for i in range(1, 4):
        ax.plot([x + w * i / 4, x + w * i / 4], [y, y + h], color="#E0E0E0", linewidth=0.35)
        ax.plot([x, x + w], [y + h * i / 4, y + h * i / 4], color="#E0E0E0", linewidth=0.35)
    theta = np.linspace(0, 2 * np.pi, 120)
    ax.plot(x + w * (0.50 + 0.24 * np.cos(theta)), y + h * (0.50 + 0.30 * np.sin(theta)), color="#D62728", linewidth=1.0)
    ax.plot(x + w * (0.50 + 0.18 * np.cos(theta + 0.25)), y + h * (0.52 + 0.22 * np.sin(theta)), color="#148F3F", linewidth=0.9)
    ax.plot([x + w * 0.63, x + w * 0.78], [y + h * 0.48, y + h * 0.48], color=accent, linewidth=2.0)


def figure1_framework():
    fig, ax = plt.subplots(figsize=(10.2, 5.25))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(0.2, 6.55, "A", fontsize=11, weight="bold")
    ax.text(0.55, 6.55, "Dense segmentation output contains both global and spatial failure signals", fontsize=10, weight="bold")

    add_box(ax, (0.55, 3.05), (1.35, 0.72), "Input\nimage", fc="#F7F7F7", fontsize=7.3)
    add_box(ax, (2.35, 3.05), (1.55, 0.72), "Base\nsegmenter", fc="#F7F7F7", fontsize=7.3)
    draw_mini_prediction(ax, 4.45, 2.58, 1.55, 1.55)
    ax.text(5.22, 4.35, "probability + mask", ha="center", fontsize=7.2)
    arrow(ax, (1.90, 3.41), (2.35, 3.41))
    arrow(ax, (3.90, 3.41), (4.45, 3.41))

    # Global compression path.
    add_box(ax, (6.65, 4.70), (1.65, 0.65), "Global\ncompression", fc="#FFFFFF", ec="#606060", fontsize=7.2)
    add_box(ax, (8.70, 4.70), (1.45, 0.65), "mean / max\nentropy", fc="#FFFFFF", ec="#606060", fontsize=7.0)
    add_box(ax, (10.55, 4.70), (1.05, 0.65), "case\nrank", fc="#F7F7F7", ec="#606060", fontsize=7.0)
    arrow(ax, (6.00, 3.72), (6.65, 5.02))
    arrow(ax, (8.30, 5.02), (8.70, 5.02))
    arrow(ax, (10.15, 5.02), (10.55, 5.02))
    ax.text(6.65, 5.72, "good for macro-area collapse", fontsize=7.2, color="#404040")

    # Spatial audit path.
    add_box(ax, (6.55, 1.35), (1.95, 0.78), "SFRM spatial\ndecomposition", fc="#FFFFFF", ec="#5B2A86", lw=1.25, fontsize=7.2)
    feature_labels = ["boundary", "topology", "uncertainty\nclusters", "overconfidence"]
    for i, lab in enumerate(feature_labels):
        add_box(ax, (8.85 + 0.72 * i, 1.18), (0.58, 1.12), lab, fc="#FBF8FE", ec="#5B2A86", lw=0.75, fontsize=5.9)
    add_box(ax, (10.55, 2.95), (1.05, 0.82), "local\nrisk map", fc="#FBF8FE", ec="#5B2A86", lw=1.0, fontsize=7.0)
    arrow(ax, (6.00, 3.10), (6.55, 1.74))
    arrow(ax, (8.50, 1.74), (8.85, 1.74))
    arrow(ax, (10.30, 2.30), (10.72, 2.95))
    ax.text(6.55, 0.75, "preserves micro-structural boundary failure", fontsize=7.2, color="#5B2A86")

    # Explicit contrast statement.
    ax.plot([6.2, 11.8], [4.18, 4.18], color="#D8D8D8", linewidth=0.7)
    ax.text(6.25, 3.98, "same prediction, different information retained", fontsize=7.2, color="#505050")
    ax.text(0.55, 0.45, "B  Paper 1: audit and failure discovery", fontsize=8, weight="bold")
    ax.text(4.85, 0.45, "C  Paper 2: feedback repair with model-actionable risk maps", fontsize=8, color="#5B2A86", weight="bold")
    save(fig, "fig1_sfrm_framework")

def figure2_primary_bars():
    t1 = read_csv(ROOT / "stage5_paper_tables/table1_primary_low_boundary_vs_global_entropy.csv")
    t2 = read_csv(ROOT / "stage5_paper_tables/table2_low_boundary_vs_trained_global_predictor.csv")
    labels = [f"{r['dataset']}\n{r['model_family']}" for r in t1]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), sharey=True)
    for ax, table, comp_title in [
        (axes[0], t1, "A. Conventional global max entropy"),
        (axes[1], t2, "B. Trained global-feature predictor"),
    ]:
        sfrm = np.array([float(r["sfrm_recall_mean"]) for r in table])
        comp = np.array([float(r["comparator_recall_mean"]) for r in table])
        ax.bar(x - width / 2, comp, width, label="Comparator", color="#C7C7C7", edgecolor="black", linewidth=0.4)
        ax.bar(x + width / 2, sfrm, width, label="SFRM", color="#5B2A86", edgecolor="black", linewidth=0.4)
        ax.set_title(comp_title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 0.42)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recall at 10% review budget", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Primary endpoint: low-boundary-quality cases", fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig2_primary_low_boundary_results")


def load_metric_rows(prefix: str):
    return read_csv(ROOT / f"stage3_predictor_{prefix}/predictor_review_budget_metrics.csv")


def curve_values(rows, label, predictor):
    items = [r for r in rows if r["label"] == label and r["predictor"] == predictor]
    items.sort(key=lambda r: float(r["budget"]))
    return [float(r["budget"]) for r in items], [float(r["recall"]) for r in items]


def figure3_review_curves():
    configs = [
        ("consep_grid_proto_seed42", "CoNSeP prototype"),
        ("monuseg_proto_seed42", "MoNuSeg prototype"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.45), sharey=True)
    label = "low_boundary_dice_le_q25"
    series = [
        ("predefined::global_max_entropy", "Global max entropy", "#9E9E9E", "--"),
        ("logistic_l2::global_features", "Global-feature predictor", "#404040", "-"),
        ("logistic_l2::sfrm_features", "SFRM predictor", "#5B2A86", "-"),
    ]
    for ax, (prefix, title) in zip(axes, configs):
        rows = load_metric_rows(prefix)
        for predictor, name, color, style in series:
            budgets, recalls = curve_values(rows, label, predictor)
            ax.plot(np.array(budgets) * 100, recalls, style, color=color, marker="o", markersize=3, linewidth=1.2, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Reviewed cases (%)", fontsize=8)
        ax.grid(color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recall of low-boundary cases", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Fixed-budget review simulation", fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig3_review_budget_curves")


def figure4_external_validation():
    rows = [r for r in read_csv(ROOT / "stage5_paper_tables/table4_external_3d_validation.csv") if r["endpoint"] == "low_boundary_dice_le_q25"]
    labels = ["Global max\nentropy" if "global_max" in r["comparator"] else "Trained global\nfeatures" for r in rows]
    sfrm = [float(r["sfrm_recall"]) for r in rows]
    comp = [float(r["comparator_recall"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.bar(x - width / 2, comp, width, color="#C7C7C7", edgecolor="black", linewidth=0.4, label="Comparator")
    ax.bar(x + width / 2, sfrm, width, color="#5B2A86", edgecolor="black", linewidth=0.4, label="SFRM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Recall at 10% review budget", fontsize=8)
    ax.set_title("External 3D FeTS/BraTS validation", fontsize=9, weight="bold")
    ax.set_ylim(0, 0.45)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_external_3d_validation")


if __name__ == "__main__":
    figure1_framework()
    figure2_primary_bars()
    figure3_review_curves()
    figure4_external_validation()
    print(f"wrote figures to {OUT}")

