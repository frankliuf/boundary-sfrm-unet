from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

from stage1_review_budget_simulation import build_labels, build_scores, read_table, review_metrics
from stage3_lightweight_predictor import (
    make_models,
    numeric_matrix,
    out_of_fold_predictions,
    source_id,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ROOT = Path("experiments/summaries")
OUT = Path("figures/paper1")
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "key": "monuseg_point_seed42",
        "dataset": "MoNuSeg",
        "model": "Point",
        "features": ROOT / "stage0_monuseg_point_seed42_features_full.csv",
        "predictions": ROOT / "stage3_predictor_monuseg_point_seed42/oof_predictions.csv",
    },
    {
        "key": "monuseg_proto_seed42",
        "dataset": "MoNuSeg",
        "model": "Prototype",
        "features": ROOT / "stage0_monuseg_proto_seed42_features_full.csv",
        "predictions": ROOT / "stage3_predictor_monuseg_proto_seed42/oof_predictions.csv",
    },
    {
        "key": "consep_grid_point_seed42",
        "dataset": "CoNSeP",
        "model": "Point",
        "features": ROOT / "stage0_consep_grid_point_seed42_features_full.csv",
        "predictions": ROOT / "stage3_predictor_consep_grid_point_seed42/oof_predictions.csv",
    },
    {
        "key": "consep_grid_proto_seed42",
        "dataset": "CoNSeP",
        "model": "Prototype",
        "features": ROOT / "stage0_consep_grid_proto_seed42_features_full.csv",
        "predictions": ROOT / "stage3_predictor_consep_grid_proto_seed42/oof_predictions.csv",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, name: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def feature_columns(rows: list[dict[str, str]]) -> list[str]:
    return [name for name in rows[0] if name.startswith("feat__")]


def family_columns(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    cols = feature_columns(rows)

    def starts(prefix: str) -> list[str]:
        return [name for name in cols if name.startswith(prefix)]

    return {
        "Global": starts("feat__global_uncertainty__"),
        "Boundary": starts("feat__boundary_risk__"),
        "Cluster": starts("feat__uncertainty_cluster__"),
        "Topology": starts("feat__topology_risk__"),
        "Morphology": starts("feat__anatomical_topological_consistency__"),
        "All SFRM": [
            name
            for name in cols
            if not name.startswith("feat__global_uncertainty__")
            and (
                name.startswith("feat__boundary_risk__")
                or name.startswith("feat__uncertainty_cluster__")
                or name.startswith("feat__topology_risk__")
                or name.startswith("feat__anatomical_topological_consistency__")
                or name.startswith("feat__feature_ambiguity__")
            )
        ],
    }


def read_oof(path: Path, label: str, predictor: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] == label and row["predictor"] == predictor:
                out[row["patch_id"]] = float(row["prediction"])
    return out


def oof_vector(rows: list[dict[str, str]], path: Path, label: str, predictor: str) -> np.ndarray:
    pred = read_oof(path, label, predictor)
    values = []
    for row in rows:
        pid = row["patch_id"]
        if pid not in pred:
            raise KeyError(f"Missing prediction for {pid}, {label}, {predictor}")
        values.append(pred[pid])
    return np.asarray(values, dtype=np.float64)


def add_contours(ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float) -> None:
    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndi.binary_dilation(mask.astype(bool), footprint) ^ ndi.binary_erosion(mask.astype(bool), footprint)


def sfrm_boundary_risk(pred: np.ndarray, entropy: np.ndarray) -> np.ndarray:
    band = boundary_band(pred, 2)
    if not np.any(band):
        return band
    boundary_entropy = entropy[band]
    low = float(np.nanquantile(boundary_entropy, 0.35))
    high = float(np.nanquantile(boundary_entropy, 0.85))
    return band & ((entropy <= low) | (entropy >= high))


def uncertainty_cluster_mask(entropy: np.ndarray) -> np.ndarray:
    thresh = max(float(np.nanquantile(entropy, 0.85)), 0.50)
    high = entropy >= thresh
    labels, n = ndi.label(high)
    if n == 0:
        return high
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[np.where(counts >= 24)[0]] = True
    keep[0] = False
    return keep[labels]


def image_path(images_dir: Path, patch_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = images_dir / f"{patch_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for {patch_id} in {images_dir}")


def choose_case(rows: list[dict[str, str]], candidates: np.ndarray, priority: np.ndarray) -> int:
    idx = np.where(candidates)[0]
    if idx.size == 0:
        raise RuntimeError("No candidate case found for qualitative panel")
    return int(idx[np.argsort(-priority[idx], kind="mergesort")[0]])


def figure_risk_map_examples() -> None:
    cfg = CONFIGS[-1]  # CoNSeP prototype is the strongest stress test for local boundary failures.
    rows = read_table(cfg["features"])
    labels = build_labels(rows, bad_dice_threshold=0.65, top_error_quantile=0.75)
    scores = build_scores(rows)
    sfrm = oof_vector(rows, cfg["predictions"], "low_boundary_dice_le_q25", "logistic_l2::sfrm_features")
    global_max = scores["global_max_entropy"]
    n = len(rows)
    k = max(1, int(math.ceil(n * 0.10)))
    sfrm_top = np.zeros(n, dtype=bool)
    global_top = np.zeros(n, dtype=bool)
    sfrm_top[np.argsort(-sfrm, kind="mergesort")[:k]] = True
    global_top[np.argsort(-global_max, kind="mergesort")[:k]] = True

    dice = np.asarray([float(row["eval__dice"]) for row in rows])
    bdice = np.asarray([float(row["eval__boundary_dice"]) for row in rows])
    bad_dice = labels["bad_dice_lt_0.65"]
    low_boundary = labels["low_boundary_dice_le_q25"]

    macro_idx = choose_case(rows, bad_dice & global_top, global_max + (1.0 - dice))
    micro_mask = low_boundary & sfrm_top & ~global_top & (dice >= 0.45)
    if not np.any(micro_mask):
        micro_mask = low_boundary & sfrm_top & ~global_top
    micro_idx = choose_case(rows, micro_mask, (sfrm - global_max) + (1.0 - bdice) + 0.25 * dice)
    foreground = np.asarray([float(row["feat__global_uncertainty__foreground_area_frac"]) for row in rows])
    components = np.asarray([float(row["feat__topology_risk__component_count"]) for row in rows])
    normal_mask = (
        (dice >= np.nanquantile(dice, 0.75))
        & (bdice >= np.nanquantile(bdice, 0.75))
        & (foreground >= 0.05)
        & (components >= 3)
        & ~sfrm_top
        & ~global_top
    )
    normal_idx = choose_case(rows, normal_mask, dice + bdice - sfrm - global_max)

    selected = [
        ("Macro-area failure", macro_idx),
        ("Micro-boundary failure", micro_idx),
        ("Low-risk control", normal_idx),
    ]
    images_dir = Path(r"D:\paper_MedIA Vol. 107–113\data\consep_test_grid_patches\images")
    maps_dir = Path("experiments/maps/stage0_consep_grid_proto_seed42_full")

    fig, axes = plt.subplots(3, 5, figsize=(11.2, 6.4), dpi=300)
    titles = [
        "Image + contours",
        "Prediction / GT",
        "Entropy map",
        "Boundary risk",
        "Uncertainty clusters",
    ]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=8.2, pad=4)

    rows_out: list[dict[str, Any]] = []
    for i, (case_type, idx) in enumerate(selected):
        row = rows[idx]
        pid = row["patch_id"]
        image = np.asarray(Image.open(image_path(images_dir, pid)).convert("RGB"))
        with np.load(maps_dir / f"{pid}.npz") as data:
            pred = data["pred"].astype(bool)
            gt = data["gt"].astype(bool)
            entropy = data["entropy"].astype(np.float32)
        risk = sfrm_boundary_risk(pred, entropy)
        cluster = uncertainty_cluster_mask(entropy)

        axes[i, 0].imshow(image)
        add_contours(axes[i, 0], gt, "#148F3F", 0.55)
        add_contours(axes[i, 0], pred, "#D62728", 0.55)
        axes[i, 0].text(
            0.01,
            0.99,
            f"{case_type}\nD={dice[idx]:.2f}, BD={bdice[idx]:.2f}",
            transform=axes[i, 0].transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )

        pred_overlay = overlay_mask(image, pred, (214, 96, 77), 0.34)
        axes[i, 1].imshow(pred_overlay)
        add_contours(axes[i, 1], gt, "#148F3F", 0.55)
        add_contours(axes[i, 1], pred, "#D62728", 0.55)

        axes[i, 2].imshow(entropy, cmap="magma", vmin=0, vmax=0.693)
        axes[i, 2].text(
            0.01,
            0.99,
            f"global={global_max[idx]:.2f}",
            transform=axes[i, 2].transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.0},
        )

        axes[i, 3].imshow(overlay_mask(image, risk, (118, 42, 131), 0.72))
        add_contours(axes[i, 3], pred, "#D62728", 0.45)
        axes[i, 3].text(
            0.01,
            0.99,
            f"SFRM={sfrm[idx]:.2f}",
            transform=axes[i, 3].transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="white",
            bbox={"facecolor": "#4B176D", "edgecolor": "none", "alpha": 0.68, "pad": 1.0},
        )

        axes[i, 4].imshow(overlay_mask(image, cluster, (0, 128, 128), 0.66))
        add_contours(axes[i, 4], pred, "#D62728", 0.45)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.35)
                spine.set_color("#CFCFCF")
        rows_out.append(
            {
                "case_type": case_type,
                "patch_id": pid,
                "dice": float(dice[idx]),
                "boundary_dice": float(bdice[idx]),
                "global_max_entropy": float(global_max[idx]),
                "sfrm_probability": float(sfrm[idx]),
            }
        )

    fig.suptitle("SFRM converts segmentation outputs into inspectable spatial risk maps", fontsize=9.4, fontweight="bold", y=0.985)
    fig.tight_layout(pad=0.35, w_pad=0.25, h_pad=0.34, rect=[0, 0, 1, 0.955])
    save(fig, "fig5_sfrm_risk_map_examples")
    write_csv(ROOT / "stage8_additional_figures/risk_map_example_cases.csv", rows_out)


def figure_descriptor_correlation() -> None:
    rows_all: list[dict[str, str]] = []
    for cfg in CONFIGS:
        rows_all.extend(read_table(cfg["features"]))
    families = family_columns(rows_all)
    ordered: list[str] = []
    for family in ["Boundary", "Cluster", "Topology", "Morphology"]:
        ordered.extend(families[family])
    matrix = numeric_matrix(rows_all, ordered)
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    short_labels: list[str] = []
    feature_key_rows: list[dict[str, str]] = []
    for family, prefix in [
        ("Boundary", "B"),
        ("Cluster", "C"),
        ("Topology", "T"),
        ("Morphology", "M"),
    ]:
        fam_cols = [name for name in ordered if name in families[family]]
        for idx, name in enumerate(fam_cols, start=1):
            code = f"{prefix}{idx}"
            short_labels.append(code)
            feature_key_rows.append({"code": code, "family": family, "feature": name})
    write_csv(ROOT / "stage8_additional_figures/descriptor_correlation_feature_key.csv", feature_key_rows)

    fig, ax = plt.subplots(figsize=(7.6, 6.9), dpi=300)
    im = ax.imshow(corr, cmap="vlag" if "vlag" in plt.colormaps() else "coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)
    starts = []
    pos = 0
    for family in ["Boundary", "Cluster", "Topology", "Morphology"]:
        starts.append((family, pos, pos + len(families[family])))
        pos += len(families[family])
    for _, start, end in starts:
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), end - start, end - start, fill=False, edgecolor="black", linewidth=0.8))
        ax.axhline(end - 0.5, color="white", linewidth=0.7)
        ax.axvline(end - 0.5, color="white", linewidth=0.7)
    for family, start, end in starts:
        center = (start + end - 1) / 2
        ax.text(center, -1.65, family, ha="center", va="bottom", fontsize=6.2, rotation=0)
        ax.text(-1.65, center, family, ha="right", va="center", fontsize=6.2, rotation=90)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson r", fontsize=7)
    ax.set_title("SFRM descriptor correlation structure", fontsize=10, fontweight="bold")
    fig.tight_layout(pad=0.4)
    save(fig, "fig6_sfrm_descriptor_correlation")


def figure_family_ablation() -> None:
    label = "low_boundary_dice_le_q25"
    budget = 0.10
    model = make_models(42)["logistic_l2"]
    output_rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        rows = read_table(cfg["features"])
        labels = build_labels(rows, bad_dice_threshold=0.65, top_error_quantile=0.75)
        y = labels[label].astype(int)
        groups = np.asarray([source_id(row["patch_id"]) for row in rows])
        for family, columns in family_columns(rows).items():
            if not columns:
                continue
            x = numeric_matrix(rows, columns)
            pred, folds_used, cv_mode = out_of_fold_predictions(x, y, groups, model, max_folds=5, random_state=42)
            metrics = review_metrics(labels[label], pred, budget)
            output_rows.append(
                {
                    "dataset": cfg["dataset"],
                    "model": cfg["model"],
                    "feature_family": family,
                    "n_features": len(columns),
                    "cv_mode": cv_mode,
                    "folds_used": folds_used,
                    "recall_at_10": metrics["recall"],
                    "precision_at_10": metrics["precision"],
                    "captured": metrics["captured"],
                    "positives": metrics["positives"],
                }
            )
    write_csv(ROOT / "stage8_additional_figures/family_ablation_low_boundary.csv", output_rows)

    families = ["Global", "Boundary", "Cluster", "Topology", "Morphology", "All SFRM"]
    cfg_labels = [f"{cfg['dataset']}\n{cfg['model']}" for cfg in CONFIGS]
    x = np.arange(len(families))
    width = 0.18
    colors = ["#999999", "#5B2A86", "#008080", "#D95F02"]
    fig, ax = plt.subplots(figsize=(9.4, 3.7), dpi=300)
    for i, cfg_label in enumerate(cfg_labels):
        vals = []
        for family in families:
            row = next(r for r in output_rows if f"{r['dataset']}\n{r['model']}" == cfg_label and r["feature_family"] == family)
            vals.append(float(row["recall_at_10"]))
        ax.bar(x + (i - 1.5) * width, vals, width, label=cfg_label, color=colors[i], edgecolor="black", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=7)
    ax.set_ylabel("Recall at 10% review budget", fontsize=8)
    ax.set_title("Per-family SFRM ablation for low-boundary-Dice detection", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.4, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0, max(0.45, ax.get_ylim()[1]))
    fig.tight_layout()
    save(fig, "fig7_family_ablation_low_boundary")


def figure_dice_boundary_scatter() -> None:
    label = "low_boundary_dice_le_q25"
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), dpi=300, sharex=True, sharey=True)
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        rows = read_table(cfg["features"])
        dice = np.asarray([float(row["eval__dice"]) for row in rows])
        bdice = np.asarray([float(row["eval__boundary_dice"]) for row in rows])
        sfrm = oof_vector(rows, cfg["predictions"], label, "logistic_l2::sfrm_features")
        scatter = ax.scatter(dice, bdice, c=sfrm, cmap="magma_r", s=14, linewidths=0.25, edgecolors="#4A4A4A", alpha=0.88, vmin=0, vmax=1)
        ax.axvline(0.65, color="#606060", linestyle="--", linewidth=0.7)
        ax.axhline(float(np.nanquantile(bdice, 0.25)), color="#606060", linestyle=":", linewidth=0.7)
        ax.set_title(f"{cfg['dataset']} {cfg['model']}", fontsize=8.5)
        ax.grid(color="#ECECEC", linewidth=0.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    axes[1, 0].set_xlabel("Global Dice", fontsize=8)
    axes[1, 1].set_xlabel("Global Dice", fontsize=8)
    axes[0, 0].set_ylabel("Boundary Dice", fontsize=8)
    axes[1, 0].set_ylabel("Boundary Dice", fontsize=8)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("SFRM-predicted failure probability", fontsize=7)
    fig.suptitle("Scale heterogeneity: global Dice can mask local boundary failure", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    save(fig, "fig8_global_vs_boundary_dice_scatter")


def main() -> None:
    (ROOT / "stage8_additional_figures").mkdir(parents=True, exist_ok=True)
    figure_risk_map_examples()
    figure_descriptor_correlation()
    figure_family_ablation()
    figure_dice_boundary_scatter()
    print(f"wrote additional figures to {OUT}")


if __name__ == "__main__":
    main()
