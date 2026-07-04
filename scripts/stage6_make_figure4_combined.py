from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from stage6_source_diverse_qualitative import add_contours, overlay_mask, sfrm_boundary_risk

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = Path("figures/paper1")
OUT.mkdir(parents=True, exist_ok=True)


def read_cases(path: Path, n: int):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[:n]


def load_image(images_dir: Path, patch_id: str):
    return np.asarray(Image.open(images_dir / f"{patch_id}.png").convert("RGB"))


def save(fig, name: str):
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_row(axes, case, images_dir: Path, maps_dir: Path, dataset_label: str):
    patch_id = case["patch_id"]
    image = load_image(images_dir, patch_id)
    with np.load(maps_dir / f"{patch_id}.npz") as data:
        pred = data["pred"].astype(bool)
        gt = data["gt"].astype(bool)
        entropy = data["entropy"].astype(np.float32)
    fp = pred & ~gt
    fn = gt & ~pred
    risk = sfrm_boundary_risk(pred, entropy, 2)

    axes[0].imshow(image)
    add_contours(axes[0], gt, "#148F3F", 0.62)
    add_contours(axes[0], pred, "#D62728", 0.62)
    axes[0].text(
        0.012,
        0.988,
        f"{dataset_label}\nD={float(case['eval__dice']):.2f}, BD={float(case['eval__boundary_dice']):.2f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )

    axes[1].imshow(entropy, cmap="magma", vmin=0, vmax=0.693)
    axes[1].text(
        0.012,
        0.988,
        f"global={float(case['global_score']):.2f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.0},
    )

    risk_overlay = overlay_mask(image, risk, (118, 42, 131), 0.72)
    axes[2].imshow(risk_overlay)
    add_contours(axes[2], pred, "#D62728", 0.50)
    axes[2].text(
        0.012,
        0.988,
        f"SFRM={float(case['sfrm_score']):.2f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="white",
        bbox={"facecolor": "#4B176D", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
    )

    err_overlay = overlay_mask(image, fp, (215, 48, 39), 0.58)
    err_overlay = overlay_mask(err_overlay, fn, (44, 123, 182), 0.58)
    axes[3].imshow(err_overlay)
    add_contours(axes[3], gt, "#148F3F", 0.48)
    add_contours(axes[3], pred, "#D62728", 0.48)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.35)
            spine.set_color("#CFCFCF")


def main():
    consep = read_cases(Path("experiments/summaries/stage6_qualitative_consep_proto_seed42_low_boundary/source_diverse_selected_cases.csv"), 2)
    monuseg = read_cases(Path("experiments/summaries/stage6_qualitative_monuseg_proto_seed42_low_boundary/source_diverse_selected_cases.csv"), 2)
    rows = [(c, "CoNSeP") for c in consep] + [(m, "MoNuSeg") for m in monuseg]
    fig, axes = plt.subplots(len(rows), 4, figsize=(10.4, 2.35 * len(rows)), dpi=300)
    titles = ["Image + GT/pred contours", "Global entropy", "SFRM boundary risk", "FP/FN error overlay"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=8.2, pad=4)
    for i, (case, dataset) in enumerate(rows):
        if dataset == "CoNSeP":
            images = Path(r"D:\paper_MedIA Vol. 107–113\data\consep_test_grid_patches\images")
            maps = Path("experiments/maps/stage0_consep_grid_proto_seed42_full")
        else:
            images = Path(r"D:\paper_MedIA Vol. 107–113\data\monuseg_test_patches\images")
            maps = Path("experiments/maps/stage0_monuseg_proto_seed42_full")
        draw_row(axes[i], case, images, maps, dataset)
    fig.suptitle("SFRM localizes boundary failures missed by global entropy", fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.35, w_pad=0.22, h_pad=0.32)
    save(fig, "fig4_source_diverse_qualitative")
    print("wrote", OUT / "fig4_source_diverse_qualitative.pdf")


if __name__ == "__main__":
    main()
