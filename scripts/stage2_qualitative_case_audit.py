from __future__ import annotations

import argparse
import csv
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and render diagnostic cases for SFRM qualitative auditing."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--label", type=str, default="gray_high_dice_high_boundary_error")
    parser.add_argument("--sfrm-score", type=str, default="boundary_overconfidence_score")
    parser.add_argument("--global-score", type=str, default="global_mean_entropy")
    parser.add_argument("--review-fraction", type=float, default=0.10)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--low-entropy-quantile", type=float, default=0.25)
    return parser.parse_args()


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
    binary = mask.astype(bool)
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    dilated = ndi.binary_dilation(binary, structure=footprint)
    eroded = ndi.binary_erosion(binary, structure=footprint)
    return np.logical_xor(dilated, eroded)


def add_contours(ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float, label: str) -> None:
    added = False
    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)
        added = True
    if added:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def render_case(
    image: np.ndarray,
    arrays: dict[str, np.ndarray],
    row: dict[str, str],
    score_values: dict[str, float],
    output_path: Path,
    boundary_radius: int,
    low_entropy_quantile: float,
) -> None:
    pred = arrays["pred"].astype(bool)
    gt = arrays["gt"].astype(bool)
    entropy = arrays["entropy"].astype(np.float32)
    prob = arrays["prob"].astype(np.float32)
    error = arrays["error"].astype(bool)

    pred_boundary = boundary_band(pred, boundary_radius)
    if pred_boundary.any():
        threshold = float(np.nanquantile(entropy[pred_boundary], low_entropy_quantile))
    else:
        threshold = float(np.nanquantile(entropy, low_entropy_quantile))
    overconf_boundary = pred_boundary & (entropy <= threshold)
    fp = pred & ~gt
    fn = gt & ~pred

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=180)
    axes = axes.ravel()

    axes[0].imshow(image)
    add_contours(axes[0], gt, "#1a9850", 1.0, "GT")
    add_contours(axes[0], pred, "#d73027", 1.0, "Pred")
    axes[0].set_title("Image + contours", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=6, frameon=False)

    im1 = axes[1].imshow(entropy, cmap="magma", vmin=0.0, vmax=max(0.693, float(entropy.max())))
    axes[1].set_title("Entropy map", fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    over = overlay_mask(image, overconf_boundary, (85, 40, 170), 0.65)
    axes[2].imshow(over)
    add_contours(axes[2], pred, "#d73027", 0.9, "Pred")
    axes[2].set_title("Deployable low-entropy boundary", fontsize=9)

    err_rgb = image.copy()
    err_rgb = overlay_mask(err_rgb, fp, (215, 48, 39), 0.55)
    err_rgb = overlay_mask(err_rgb, fn, (44, 123, 182), 0.55)
    axes[3].imshow(err_rgb)
    add_contours(axes[3], gt, "#1a9850", 0.8, "GT")
    add_contours(axes[3], pred, "#d73027", 0.8, "Pred")
    axes[3].set_title("Evaluation error: FP red / FN blue", fontsize=9)

    axes[4].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
    add_contours(axes[4], gt, "white", 0.7, "GT")
    axes[4].set_title("Foreground probability", fontsize=9)

    axes[5].axis("off")
    text = "\n".join(
        [
            f"patch_id: {row['patch_id']}",
            f"Dice: {float(row['eval__dice']):.4f}",
            f"Boundary Dice: {float(row['eval__boundary_dice']):.4f}",
            f"Boundary error frac: {float(row['eval__boundary_error_area_frac']):.4f}",
            f"LECR-boundary: {float(row['eval__lecr_boundary_error']):.4f}",
            f"global entropy score: {score_values['global']:.3f}",
            f"SFRM score: {score_values['sfrm']:.3f}",
            f"overconf boundary pixels: {int(overconf_boundary.sum())}",
            f"entropy threshold: {threshold:.4f}",
        ]
    )
    axes[5].text(0.02, 0.98, text, va="top", ha="left", fontsize=8, family="monospace")

    for ax in axes[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_table(args.features_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.features_csv}")

    labels = build_labels(rows, args.bad_dice_threshold, args.top_error_quantile)
    scores = build_scores(rows)
    if args.label not in labels:
        raise KeyError(f"Unknown label {args.label}. Available: {sorted(labels)}")
    if args.sfrm_score not in scores or args.global_score not in scores:
        raise KeyError(f"Unknown scores. Available: {sorted(scores)}")

    label_values = labels[args.label]
    sfrm = scores[args.sfrm_score]
    global_score = scores[args.global_score]
    n = len(rows)
    k = max(1, int(math.ceil(n * args.review_fraction)))
    sfrm_top = np.zeros(n, dtype=bool)
    global_top = np.zeros(n, dtype=bool)
    sfrm_top[np.argsort(-sfrm, kind="mergesort")[:k]] = True
    global_top[np.argsort(-global_score, kind="mergesort")[:k]] = True

    case_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not label_values[i]:
            continue
        if sfrm_top[i] and not global_top[i]:
            category = "sfrm_hit_global_miss"
        elif global_top[i] and not sfrm_top[i]:
            category = "global_hit_sfrm_miss"
        elif sfrm_top[i] and global_top[i]:
            category = "both_hit"
        else:
            category = "both_miss"
        boundary_error = float(row["eval__boundary_error_area_frac"])
        priority = (sfrm[i] - global_score[i]) + boundary_error
        case_rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "patch_id": row["patch_id"],
                "category": category,
                "label": args.label,
                "sfrm_score_name": args.sfrm_score,
                "global_score_name": args.global_score,
                "sfrm_score": float(sfrm[i]),
                "global_score": float(global_score[i]),
                "score_gap": float(sfrm[i] - global_score[i]),
                "priority": float(priority),
                "eval__dice": float(row["eval__dice"]),
                "eval__boundary_dice": float(row["eval__boundary_dice"]),
                "eval__boundary_error_area_frac": boundary_error,
                "eval__lecr_boundary_error": float(row["eval__lecr_boundary_error"]),
                "eval__lecr_uncertainty": float(row["eval__lecr_uncertainty"]),
            }
        )

    if not case_rows:
        raise RuntimeError(f"No positive cases found for label {args.label}")

    ordered = sorted(
        case_rows,
        key=lambda r: (
            r["category"] != "sfrm_hit_global_miss",
            -float(r["priority"]),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selected_cases.csv", ordered)

    rendered = 0
    for case in ordered:
        if case["category"] != "sfrm_hit_global_miss":
            continue
        patch_id = str(case["patch_id"])
        npz_path = args.maps_dir / f"{patch_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        image = load_image(args.images_dir, patch_id)
        with np.load(npz_path) as z:
            arrays = {name: z[name] for name in z.files}
        row = next(r for r in rows if r["patch_id"] == patch_id)
        render_case(
            image=image,
            arrays=arrays,
            row=row,
            score_values={"sfrm": float(case["sfrm_score"]), "global": float(case["global_score"])},
            output_path=args.output_dir / "figures" / f"{patch_id}.png",
            boundary_radius=args.boundary_radius,
            low_entropy_quantile=args.low_entropy_quantile,
        )
        rendered += 1
        if rendered >= args.top_n:
            break

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "features_csv": str(args.features_csv),
        "maps_dir": str(args.maps_dir),
        "label": args.label,
        "sfrm_score": args.sfrm_score,
        "global_score": args.global_score,
        "review_fraction": args.review_fraction,
        "review_count": k,
        "positive_cases": int(label_values.sum()),
        "category_counts": {
            category: sum(1 for row in case_rows if row["category"] == category)
            for category in ["sfrm_hit_global_miss", "global_hit_sfrm_miss", "both_hit", "both_miss"]
        },
        "rendered_sfrm_hit_global_miss": rendered,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(summary, handle, indent=2)

    print(f"positive_cases={summary['positive_cases']}")
    print(f"category_counts={summary['category_counts']}")
    print(f"rendered={rendered}")
    print(f"wrote={args.output_dir / 'selected_cases.csv'}")


if __name__ == "__main__":
    main()
