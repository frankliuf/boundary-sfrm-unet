from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from skimage import measure, morphology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract leakage-free SFRM features from cached 3D segmentation NPZ artifacts."
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing per-case .npz files with pred, label, probability, uncertainty, and error.",
    )
    parser.add_argument(
        "--site-labels",
        type=str,
        nargs="+",
        required=True,
        help="One site label per input directory, used only in metadata and patch_id construction.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--uncertainty-quantile", type=float, default=0.85)
    parser.add_argument("--probability-key", type=str, default="probability")
    parser.add_argument("--uncertainty-key", type=str, default="uncertainty")
    return parser.parse_args()


def binary_entropy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def margin_uncertainty(prob: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * np.abs(prob.astype(np.float32) - 0.5)


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = float(np.logical_and(pred_b, gt_b).sum())
    denom = float(pred_b.sum() + gt_b.sum())
    return (2.0 * inter + eps) / (denom + eps)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_b = mask.astype(bool)
    if not mask_b.any():
        return np.zeros_like(mask_b, dtype=bool)
    footprint = morphology.ball(max(1, radius))
    dilated = morphology.dilation(mask_b, footprint)
    eroded = morphology.erosion(mask_b, footprint)
    return np.logical_xor(dilated, eroded)


def connected_component_stats(mask: np.ndarray) -> dict[str, float]:
    mask_b = mask.astype(bool)
    labeled = measure.label(mask_b, connectivity=1)
    regions = measure.regionprops(labeled)
    areas = np.asarray([r.area for r in regions], dtype=np.float64)
    total = float(mask_b.size)
    if not regions:
        return {
            "component_count": 0.0,
            "largest_component_area_frac": 0.0,
            "small_component_count": 0.0,
            "mean_component_area_frac": 0.0,
            "component_area_cv": 0.0,
            "euler_number": 0.0,
        }
    small_threshold = max(32.0, 0.0001 * total)
    mean_area = float(areas.mean())
    return {
        "component_count": float(len(regions)),
        "largest_component_area_frac": float(areas.max() / total),
        "small_component_count": float((areas < small_threshold).sum()),
        "mean_component_area_frac": float(mean_area / total),
        "component_area_cv": float(areas.std() / (mean_area + 1e-7)),
        "euler_number": float(measure.euler_number(mask_b, connectivity=1)),
    }


def topology_stability(prob: np.ndarray, thresholds: tuple[float, ...] = (0.45, 0.5, 0.55)) -> dict[str, float]:
    counts: list[float] = []
    areas: list[float] = []
    eulers: list[float] = []
    for threshold in thresholds:
        mask = prob >= threshold
        stats = connected_component_stats(mask)
        counts.append(stats["component_count"])
        areas.append(float(mask.mean()))
        eulers.append(float(measure.euler_number(mask.astype(bool), connectivity=1)))
    return {
        "threshold_component_count_std": float(np.std(counts)),
        "threshold_area_frac_std": float(np.std(areas)),
        "threshold_euler_std": float(np.std(eulers)),
    }


def morphology_opening_residual(mask: np.ndarray) -> float:
    opened = morphology.opening(mask.astype(bool), morphology.ball(1))
    return float(np.logical_xor(mask.astype(bool), opened).mean())


def morphology_closing_residual(mask: np.ndarray) -> float:
    closed = morphology.closing(mask.astype(bool), morphology.ball(1))
    return float(np.logical_xor(mask.astype(bool), closed).mean())


def uncertainty_cluster_stats(uncertainty: np.ndarray, quantile: float) -> dict[str, float]:
    threshold = float(np.quantile(uncertainty, quantile))
    high = uncertainty >= threshold
    stats = connected_component_stats(high)
    return {
        "high_uncertainty_area_frac": float(high.mean()),
        "high_uncertainty_threshold": threshold,
        "high_uncertainty_component_count": stats["component_count"],
        "high_uncertainty_largest_component_area_frac": stats["largest_component_area_frac"],
        "high_uncertainty_small_component_count": stats["small_component_count"],
    }


def deployable_features(
    prob: np.ndarray,
    uncertainty: np.ndarray,
    pred_fg: np.ndarray,
    boundary_radius: int,
    uncertainty_quantile: float,
) -> dict[str, dict[str, float]]:
    entropy = binary_entropy(prob)
    margin = margin_uncertainty(prob)
    pred_boundary = boundary_band(pred_fg, boundary_radius)
    pred_stats = connected_component_stats(pred_fg)
    cluster_stats = uncertainty_cluster_stats(uncertainty, uncertainty_quantile)
    stability = topology_stability(prob)

    boundary_entropy = entropy[pred_boundary] if pred_boundary.any() else np.asarray([], dtype=np.float32)
    boundary_margin = margin[pred_boundary] if pred_boundary.any() else np.asarray([], dtype=np.float32)
    boundary_uncertainty = uncertainty[pred_boundary] if pred_boundary.any() else np.asarray([], dtype=np.float32)

    return {
        "global_uncertainty": {
            "mean_entropy": float(entropy.mean()),
            "max_entropy": float(entropy.max()),
            "mean_margin_uncertainty": float(margin.mean()),
            "max_margin_uncertainty": float(margin.max()),
            "foreground_area_frac": float(pred_fg.mean()),
            "mean_cached_uncertainty": float(uncertainty.mean()),
            "max_cached_uncertainty": float(uncertainty.max()),
        },
        "boundary_risk": {
            "pred_boundary_area_frac": float(pred_boundary.mean()),
            "boundary_mean_entropy": float(boundary_entropy.mean()) if boundary_entropy.size else 0.0,
            "boundary_max_entropy": float(boundary_entropy.max()) if boundary_entropy.size else 0.0,
            "boundary_mean_margin_uncertainty": float(boundary_margin.mean()) if boundary_margin.size else 0.0,
            "boundary_high_entropy_frac": float((boundary_entropy >= np.quantile(entropy, 0.85)).mean())
            if boundary_entropy.size
            else 0.0,
            "boundary_mean_cached_uncertainty": float(boundary_uncertainty.mean()) if boundary_uncertainty.size else 0.0,
            "boundary_max_cached_uncertainty": float(boundary_uncertainty.max()) if boundary_uncertainty.size else 0.0,
        },
        "uncertainty_cluster": cluster_stats,
        "topology_risk": {
            **pred_stats,
            **stability,
        },
        "anatomical_topological_consistency": {
            "opening_residual_frac": morphology_opening_residual(pred_fg),
            "closing_residual_frac": morphology_closing_residual(pred_fg),
        },
    }


def evaluation_metrics(
    prob: np.ndarray,
    uncertainty: np.ndarray,
    pred_fg: np.ndarray,
    label_fg: np.ndarray,
    boundary_radius: int,
) -> dict[str, float]:
    error = np.logical_xor(pred_fg.astype(bool), label_fg.astype(bool))
    gt_boundary = boundary_band(label_fg, boundary_radius)
    pred_boundary = boundary_band(pred_fg, boundary_radius)
    union_boundary = np.logical_or(gt_boundary, pred_boundary)
    total_uncertainty = float(uncertainty.sum()) + 1e-7
    error_uncertainty = float(uncertainty[error].sum()) if error.any() else 0.0
    return {
        "dice": dice(pred_fg, label_fg),
        "boundary_dice": dice(pred_boundary, gt_boundary),
        "error_area_frac": float(error.mean()),
        "boundary_error_area_frac": float(np.logical_and(error, union_boundary).mean()),
        "lecr_uncertainty": error_uncertainty / total_uncertainty,
        "lecr_boundary_error": float(np.logical_and(error, union_boundary).sum() / (error.sum() + 1e-7)),
    }


def flatten_entry(
    patch_id: str,
    site: str,
    features: dict[str, dict[str, float]],
    metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {"patch_id": patch_id, "site": site}
    for family, values in features.items():
        for name, value in values.items():
            row[f"feat__{family}__{name}"] = value
    for name, value in metrics.items():
        row[f"eval__{name}"] = value
    return row


def assert_no_leakage(features: dict[str, dict[str, float]]) -> None:
    forbidden = ("gt", "ground_truth", "label", "dice", "error", "lecr")
    leakage = []
    for family, values in features.items():
        for name in values:
            lowered = f"{family}__{name}".lower()
            if any(token in lowered for token in forbidden):
                leakage.append(lowered)
    if leakage:
        raise RuntimeError(f"Deployable feature names contain leakage tokens: {leakage}")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    args = parse_args()
    if len(args.input_dirs) != len(args.site_labels):
        raise ValueError("--input-dirs and --site-labels must have the same length")

    rows: list[dict[str, Any]] = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as jsonl:
        for input_dir, site in zip(args.input_dirs, args.site_labels):
            files = sorted(input_dir.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No NPZ files found in {input_dir}")
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    pred = data["pred"]
                    label = data["label"]
                    prob = data[args.probability_key].astype(np.float32)
                    uncertainty = data[args.uncertainty_key].astype(np.float32)
                pred_fg = pred > 0
                label_fg = label > 0
                features = deployable_features(
                    prob=prob,
                    uncertainty=uncertainty,
                    pred_fg=pred_fg,
                    boundary_radius=args.boundary_radius,
                    uncertainty_quantile=args.uncertainty_quantile,
                )
                assert_no_leakage(features)
                metrics = evaluation_metrics(
                    prob=prob,
                    uncertainty=uncertainty,
                    pred_fg=pred_fg,
                    label_fg=label_fg,
                    boundary_radius=args.boundary_radius,
                )
                case_id = path.stem
                patch_id = f"{site}-{case_id}"
                entry = {
                    "patch_id": patch_id,
                    "site": site,
                    "source_npz": str(path),
                    "deployable_features": features,
                    "evaluation_metrics": metrics,
                }
                jsonl.write(json.dumps(entry, ensure_ascii=False, default=json_safe) + "\n")
                rows.append(flatten_entry(patch_id, site, features, metrics))

    if not rows:
        raise RuntimeError("No cases were extracted.")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote_csv={args.output_csv}")
    print(f"wrote_jsonl={args.output_jsonl}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
