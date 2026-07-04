from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "paper1"
TABLE_DIR = ROOT / "outputs" / "paper_tables"


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(ax, xy, width, height, text, edge, face="#FFFFFF", fontsize=12, lw=2.0, rounding=0.03):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, color="#5B6470", lw=2.0, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=18,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Top pipeline.
    _box(ax, (0.04, 0.77), 0.16, 0.11, "Input image $X$", edge="#3B82F6", fontsize=16)
    _box(ax, (0.28, 0.75), 0.24, 0.14, "Coarse U-Net\nencoder + decoder", edge="#2563EB", fontsize=17)
    _box(ax, (0.61, 0.79), 0.13, 0.08, "Prob. $P$", edge="#7C3AED", fontsize=15)
    _box(ax, (0.79, 0.79), 0.12, 0.08, "Mask $M$", edge="#7C3AED", fontsize=15)
    _arrow(ax, (0.20, 0.825), (0.28, 0.825))
    _arrow(ax, (0.52, 0.825), (0.61, 0.825))
    _arrow(ax, (0.74, 0.825), (0.79, 0.825))

    # Failure branch.
    _box(ax, (0.31, 0.56), 0.18, 0.10, "Failure head\n$F_b$, $F_s$", edge="#EF4444", fontsize=15)
    _arrow(ax, (0.40, 0.75), (0.40, 0.66), color="#5B6470")

    # Risk tensor block.
    _box(ax, (0.16, 0.41), 0.68, 0.11, "SFRM risk tensor  $R = [P, H(P), F_b, F_s]$", edge="#0F766E", fontsize=18)
    ax.text(0.50, 0.385, "refinement uses raw risk; audit uses calibrated risk", ha="center", va="center", fontsize=12, color="#475569")
    _arrow(ax, (0.675, 0.79), (0.675, 0.52), color="#5B6470")
    _arrow(ax, (0.40, 0.56), (0.40, 0.52), color="#5B6470")

    # Decoder refinement.
    _box(ax, (0.08, 0.22), 0.23, 0.12, "Refine decoder $l=3$\ncoarse skip modulation", edge="#2563EB", fontsize=14)
    _box(ax, (0.38, 0.22), 0.23, 0.12, "Refine decoder $l=2$\nrisk-gated update", edge="#2563EB", fontsize=14)
    _box(ax, (0.68, 0.22), 0.23, 0.12, "Refine decoder $l=1$\nboundary correction", edge="#2563EB", fontsize=14)
    _arrow(ax, (0.26, 0.41), (0.20, 0.34), color="#0F766E")
    _arrow(ax, (0.50, 0.41), (0.50, 0.34), color="#0F766E")
    _arrow(ax, (0.74, 0.41), (0.80, 0.34), color="#0F766E")
    _arrow(ax, (0.31, 0.28), (0.38, 0.28))
    _arrow(ax, (0.61, 0.28), (0.68, 0.28))

    # Output.
    _box(ax, (0.36, 0.06), 0.28, 0.10, "Refined mask  $\\hat{Y}$", edge="#16A34A", fontsize=18)
    _arrow(ax, (0.50, 0.22), (0.50, 0.16))

    # Training-only note.
    _box(ax, (0.02, 0.05), 0.24, 0.11, "training-only failure targets\nboundary band + leakage / bridge", edge="#DC2626", fontsize=13, lw=1.8)
    _arrow(ax, (0.26, 0.11), (0.31, 0.11), color="#DC2626", lw=1.8)
    ax.text(
        0.50,
        0.97,
        "Boundary-SFRM-UNet: structured failure extraction and decoder feedback",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.935,
        "White-background line diagram; no icons; risk maps are fused at multiple decoder scales",
        ha="center",
        va="top",
        fontsize=11,
        color="#475569",
    )
    _save(fig, "fig_boundary_sfrm_unet_architecture")


def figure_results_summary() -> None:
    data = json.loads((TABLE_DIR / "feedback_refinement_summary.json").read_text(encoding="utf-8"))
    datasets = [row["dataset"] for row in data]
    deltas = {
        "Boundary Dice": [row["delta_boundary_dice"] for row in data],
        "AJI": [row["delta_aji"] for row in data],
        "PQ": [row["delta_pq"] for row in data],
        "Confounder FPR": [-row["delta_confounder_fpr"] for row in data],
    }

    colors = {
        "Boundary Dice": "#2563EB",
        "AJI": "#7C3AED",
        "PQ": "#0891B2",
        "Confounder FPR": "#16A34A",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(datasets))
    width = 0.18

    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, (name, vals) in zip(offsets, deltas.items(), strict=True):
        ax.bar(x + offset * width, vals, width=width, label=name, color=colors[name], edgecolor="black", linewidth=0.8)

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Improvement over baseline", fontsize=12)
    ax.set_title(
        "Boundary-SFRM-UNet improves structural metrics\nand suppresses confounder leakage",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax.legend(frameon=False, ncol=2, fontsize=11, loc="upper left")
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "fig_boundary_sfrm_unet_results")


def figure_monuseg_supporting() -> None:
    rows = json.loads((TABLE_DIR / "monuseg_cue_comparison_fixed_epoch8.json").read_text(encoding="utf-8"))
    models = [
        "Baseline\nU-Net",
        "Two-pass\nno risk",
        "Entropy-only\nrefinement",
        "Boundary-\nSFRM v3",
        "Learned failure\nhead v3",
    ]
    lookup = {row["model"]: row for row in rows}
    ordered = [lookup["Baseline U-Net"], lookup["Two-pass, no risk"], lookup["Entropy-only refinement"], lookup["Boundary-SFRM v3"], lookup["Learned failure head v3"]]
    metrics = {
        "Boundary Dice": [row["boundary_dice"] for row in ordered],
        "AJI": [row["aji"] for row in ordered],
        "PQ": [row["pq"] for row in ordered],
        "Confounder FPR": [row["confounder_fpr"] for row in ordered],
    }
    colors = {
        "Boundary Dice": "#2563EB",
        "AJI": "#7C3AED",
        "PQ": "#0891B2",
        "Confounder FPR": "#DC2626",
    }

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8))
    fig.patch.set_facecolor("white")
    x = np.arange(len(models))

    for ax, (metric, vals) in zip(axes, metrics.items(), strict=True):
        ax.set_facecolor("white")
        bars = ax.bar(x, vals, color=colors[metric], edgecolor="black", linewidth=0.8, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if metric == "Confounder FPR":
            ax.set_ylim(0.66, 0.74)
            best_idx = int(np.argmin(vals))
        else:
            ymin = min(vals)
            ymax = max(vals)
            pad = max(0.01, (ymax - ymin) * 0.35)
            ax.set_ylim(ymin - pad, ymax + pad)
            best_idx = int(np.argmax(vals))
        bars[best_idx].set_linewidth(1.6)
        bars[best_idx].set_edgecolor("#111827")
        for idx, v in enumerate(vals):
            ax.text(idx, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)

    fig.suptitle(
        "MoNuSeg supporting study: structured SFRM cues outperform generic second-pass refinement",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "All refinement variants are shown at the same endpoint (epoch 8) to isolate cue quality under an identical optimization budget. "
        "Under this fixed-budget comparison, structured SFRM cues continue improving and the teacher-guided learned failure head is strongest overall.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout()
    _save(fig, "fig_boundary_sfrm_unet_monuseg_support")


def main() -> None:
    figure_architecture()
    figure_results_summary()
    figure_monuseg_supporting()


if __name__ == "__main__":
    main()
