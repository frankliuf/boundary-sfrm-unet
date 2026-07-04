from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

from stage1_review_budget_simulation import build_labels, build_scores, read_table

PATCH_SOURCE_RE = re.compile(r"^(?P<source>.+?)_y\d+_x\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select source-diverse qualitative SFRM cases.")
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--label", type=str, default="low_boundary_dice_le_q25")
    parser.add_argument("--sfrm-predictor", type=str, default="logistic_l2::sfrm_features")
    parser.add_argument("--global-predictor", type=str, default="predefined::global_max_entropy")
    parser.add_argument("--review-fraction", type=float, default=0.10)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--min-dice", type=float, default=0.0)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
    return parser.parse_args()


def source_id(patch_id: str) -> str:
    match = PATCH_SOURCE_RE.match(patch_id)
    if match:
        return match.group("source")
    parts = patch_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else patch_id


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_image(images_dir: Path, patch_id: str) -> np.ndarray:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = images_dir / f"{patch_id}{suffix}"
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    raise FileNotFoundError(f"No image file found for patch_id={patch_id} in {images_dir}")


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    dilated = ndi.binary_dilation(mask.astype(bool), structure=footprint)
    eroded = ndi.binary_erosion(mask.astype(bool), structure=footprint)
    return np.logical_xor(dilated, eroded)


def add_contours(ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float) -> None:
    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def read_oof_predictions(path: Path, label: str, predictor: str) -> dict[str, float]:
    output: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] == label and row["predictor"] == predictor:
                output[row["patch_id"]] = float(row["prediction"])
    return output


def score_vector(rows: list[dict[str, str]], label: str, predictor: str, predictions_csv: Path) -> np.ndarray:
    if predictor.startswith("predefined::"):
        score_name = predictor.split("::", 1)[1]
        scores = build_scores(rows)
        if score_name not in scores:
            raise KeyError(f"Unknown predefined score: {score_name}")
        return scores[score_name]
    predictions = read_oof_predictions(predictions_csv, label, predictor)
    values = []
    for row in rows:
        pid = row["patch_id"]
        if pid not in predictions:
            raise KeyError(f"Missing OOF prediction for {label}, {predictor}, {pid}")
        values.append(predictions[pid])
    return np.asarray(values, dtype=np.float64)


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_table(args.features_csv)
    labels = build_labels(rows, args.bad_dice_threshold, args.top_error_quantile)
    if args.label not in labels:
        raise KeyError(f"Unknown label {args.label}")
    y = labels[args.label]
    sfrm_scores = score_vector(rows, args.label, args.sfrm_predictor, args.predictions_csv)
    global_scores = score_vector(rows, args.label, args.global_predictor, args.predictions_csv)
    n = len(rows)
    k = max(1, int(math.ceil(n * args.review_fraction)))
    sfrm_top = np.zeros(n, dtype=bool)
    global_top = np.zeros(n, dtype=bool)
    sfrm_top[np.argsort(-sfrm_scores, kind="mergesort")[:k]] = True
    global_top[np.argsort(-global_scores, kind="mergesort")[:k]] = True

    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not y[i] or not (sfrm_top[i] and not global_top[i]):
            continue
        dice_value = float(row["eval__dice"])
        if dice_value < args.min_dice:
            continue
        bd = float(row["eval__boundary_dice"])
        gap = float(sfrm_scores[i] - global_scores[i])
        candidates.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "patch_id": row["patch_id"],
                "source_id": source_id(row["patch_id"]),
                "label": args.label,
                "category": "sfrm_hit_global_miss",
                "sfrm_predictor": args.sfrm_predictor,
                "global_predictor": args.global_predictor,
                "sfrm_score": float(sfrm_scores[i]),
                "global_score": float(global_scores[i]),
                "score_gap": gap,
                "priority": gap + (1.0 - bd) + 0.35 * dice_value,
                "eval__dice": dice_value,
                "eval__boundary_dice": bd,
                "eval__boundary_error_area_frac": float(row["eval__boundary_error_area_frac"]),
                "eval__lecr_boundary_error": float(row["eval__lecr_boundary_error"]),
            }
        )
    candidates.sort(key=lambda item: -float(item["priority"]))
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for item in candidates:
        if item["source_id"] in used_sources:
            continue
        selected.append(item)
        used_sources.add(item["source_id"])
        if len(selected) >= args.max_cases:
            return selected
    if args.min_dice > 0 and len(selected) < args.max_cases:
        relaxed = argparse.Namespace(**{**vars(args), "min_dice": 0.0})
        return select_cases(relaxed)
    for item in candidates:
        if item not in selected:
            selected.append(item)
        if len(selected) >= args.max_cases:
            break
    return selected


def sfrm_boundary_risk(pred: np.ndarray, entropy: np.ndarray, radius: int) -> np.ndarray:
    pred_boundary = boundary_band(pred, radius)
    if not pred_boundary.any():
        return pred_boundary
    boundary_entropy = entropy[pred_boundary]
    low_threshold = float(np.nanquantile(boundary_entropy, 0.35))
    high_threshold = float(np.nanquantile(boundary_entropy, 0.85))
    return pred_boundary & ((entropy <= low_threshold) | (entropy >= high_threshold))


def render_composite(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    if not selected:
        raise RuntimeError("No selected cases to render")
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 4, figsize=(10.8, 2.55 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.asarray([axes])
    for col, title in enumerate(["Image + contours", "Global entropy", "SFRM boundary risk", "FP / FN errors"]):
        axes[0, col].set_title(title, fontsize=8)
    for row_idx, case in enumerate(selected):
        patch_id = str(case["patch_id"])
        image = load_image(args.images_dir, patch_id)
        with np.load(args.maps_dir / f"{patch_id}.npz") as data:
            pred = data["pred"].astype(bool)
            gt = data["gt"].astype(bool)
            entropy = data["entropy"].astype(np.float32)
        fp = pred & ~gt
        fn = gt & ~pred
        risk = sfrm_boundary_risk(pred, entropy, args.boundary_radius)

        axes[row_idx, 0].imshow(image)
        add_contours(axes[row_idx, 0], gt, "#148F3F", 0.65)
        add_contours(axes[row_idx, 0], pred, "#D62728", 0.65)
        axes[row_idx, 0].text(
            0.01,
            0.99,
            f"{case['source_id']}\nD={case['eval__dice']:.2f}, BD={case['eval__boundary_dice']:.2f}",
            transform=axes[row_idx, 0].transAxes,
            va="top",
            ha="left",
            fontsize=5.8,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        axes[row_idx, 1].imshow(entropy, cmap="magma", vmin=0.0, vmax=0.693)
        risk_overlay = overlay_mask(image, risk, (118, 42, 131), 0.68)
        axes[row_idx, 2].imshow(risk_overlay)
        add_contours(axes[row_idx, 2], pred, "#D62728", 0.55)
        err_overlay = overlay_mask(image, fp, (215, 48, 39), 0.58)
        err_overlay = overlay_mask(err_overlay, fn, (44, 123, 182), 0.58)
        axes[row_idx, 3].imshow(err_overlay)
        add_contours(axes[row_idx, 3], gt, "#148F3F", 0.5)
        add_contours(axes[row_idx, 3], pred, "#D62728", 0.5)
        for col in range(4):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            for spine in axes[row_idx, col].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color("#D0D0D0")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.4, w_pad=0.25, h_pad=0.35)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "source_diverse_qualitative_panel.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "source_diverse_qualitative_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selected = select_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "source_diverse_selected_cases.csv", selected)
    render_composite(args, selected)
    print(f"selected={len(selected)}")
    print(f"sources={len({item['source_id'] for item in selected})}")
    print(f"wrote={args.output_dir / 'source_diverse_selected_cases.csv'}")
    print(f"wrote={args.output_dir / 'source_diverse_qualitative_panel.png'}")


if __name__ == "__main__":
    main()

