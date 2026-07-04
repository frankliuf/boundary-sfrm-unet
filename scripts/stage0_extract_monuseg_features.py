from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage import measure, morphology
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract leakage-free SFRM features and audit-only metrics for MoNuSeg patches."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(".."))
    parser.add_argument("--images-dir", type=Path, default=Path("../data/monuseg_test_patches/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("../data/monuseg_test_patches/masks"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../outputs/confounder_prompting/monuseg_split_point_seed42/best_model.pt"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("experiments/summaries/stage0_monuseg_features.csv"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("experiments/summaries/stage0_monuseg_cache.jsonl"))
    parser.add_argument(
        "--output-map-dir",
        type=Path,
        default=None,
        help="Optional directory for compressed per-patch probability, entropy, prediction, GT, and error maps.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--uncertainty-quantile", type=float, default=0.85)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_confounder_project_to_path(workspace_root: Path) -> None:
    candidates = (ROOT, workspace_root / "confounder_prompting")
    for candidate in candidates:
        if (candidate / "confounder_mining").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return
    raise FileNotFoundError("Cannot find the bundled or sibling confounder_mining package")


def binary_entropy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def safe_patch_filename(patch_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in patch_id)


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
    if radius <= 0:
        return mask_b ^ morphology.binary_erosion(mask_b)
    # Match confounder_prompting.metrics._mask_boundary, which uses a
    # square max-pooling footprint with kernel size 2 * radius + 1.
    kernel_size = radius * 2 + 1
    selem = np.ones((kernel_size, kernel_size), dtype=bool)
    dilated = morphology.dilation(mask_b, selem)
    eroded = morphology.erosion(mask_b, selem)
    return np.logical_xor(dilated, eroded)


def connected_component_stats(mask: np.ndarray) -> dict[str, float]:
    labeled = measure.label(mask.astype(bool), connectivity=2)
    regions = measure.regionprops(labeled)
    areas = np.array([r.area for r in regions], dtype=np.float64)
    total = float(mask.size)
    if len(regions) == 0:
        return {
            "component_count": 0.0,
            "largest_component_area_frac": 0.0,
            "small_component_count": 0.0,
            "mean_component_area_frac": 0.0,
            "component_area_cv": 0.0,
            "mean_eccentricity": 0.0,
            "mean_solidity": 0.0,
            "euler_number": 0.0,
        }
    small_threshold = max(8.0, 0.001 * total)
    area_mean = float(areas.mean())
    return {
        "component_count": float(len(regions)),
        "largest_component_area_frac": float(areas.max() / total),
        "small_component_count": float((areas < small_threshold).sum()),
        "mean_component_area_frac": float(area_mean / total),
        "component_area_cv": float(areas.std() / (area_mean + 1e-7)),
        "mean_eccentricity": float(np.mean([r.eccentricity for r in regions])),
        "mean_solidity": float(np.mean([r.solidity for r in regions])),
        "euler_number": float(measure.euler_number(mask.astype(bool), connectivity=2)),
    }


def topology_stability(prob: np.ndarray, thresholds: tuple[float, ...] = (0.45, 0.5, 0.55)) -> dict[str, float]:
    counts = []
    areas = []
    eulers = []
    for threshold in thresholds:
        mask = prob >= threshold
        counts.append(connected_component_stats(mask)["component_count"])
        areas.append(float(mask.mean()))
        eulers.append(float(measure.euler_number(mask.astype(bool), connectivity=2)))
    return {
        "threshold_component_count_std": float(np.std(counts)),
        "threshold_area_frac_std": float(np.std(areas)),
        "threshold_euler_std": float(np.std(eulers)),
    }


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


def deployable_features(prob: np.ndarray, pred: np.ndarray, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    entropy = binary_entropy(prob)
    margin = margin_uncertainty(prob)
    pred_boundary = boundary_band(pred, args.boundary_radius)
    pred_stats = connected_component_stats(pred)
    cluster_stats = uncertainty_cluster_stats(entropy, args.uncertainty_quantile)
    stability = topology_stability(prob)

    boundary_entropy = entropy[pred_boundary] if pred_boundary.any() else np.array([], dtype=np.float32)
    boundary_margin = margin[pred_boundary] if pred_boundary.any() else np.array([], dtype=np.float32)

    return {
        "global_uncertainty": {
            "mean_entropy": float(entropy.mean()),
            "max_entropy": float(entropy.max()),
            "mean_margin_uncertainty": float(margin.mean()),
            "max_margin_uncertainty": float(margin.max()),
            "foreground_area_frac": float(pred.mean()),
        },
        "boundary_risk": {
            "pred_boundary_area_frac": float(pred_boundary.mean()),
            "boundary_mean_entropy": float(boundary_entropy.mean()) if boundary_entropy.size else 0.0,
            "boundary_max_entropy": float(boundary_entropy.max()) if boundary_entropy.size else 0.0,
            "boundary_mean_margin_uncertainty": float(boundary_margin.mean()) if boundary_margin.size else 0.0,
            "boundary_high_entropy_frac": float((boundary_entropy >= np.quantile(entropy, 0.85)).mean())
            if boundary_entropy.size
            else 0.0,
        },
        "uncertainty_cluster": cluster_stats,
        "topology_risk": {
            **pred_stats,
            **stability,
        },
        "anatomical_topological_consistency": {
            "opening_residual_frac": morphology_opening_residual(pred),
            "closing_residual_frac": morphology_closing_residual(pred),
        },
    }


def morphology_opening_residual(mask: np.ndarray) -> float:
    opened = morphology.opening(mask.astype(bool), morphology.disk(1))
    return float(np.logical_xor(mask.astype(bool), opened).mean())


def morphology_closing_residual(mask: np.ndarray) -> float:
    closed = morphology.closing(mask.astype(bool), morphology.disk(1))
    return float(np.logical_xor(mask.astype(bool), closed).mean())


def evaluation_metrics(prob: np.ndarray, pred: np.ndarray, gt: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    entropy = binary_entropy(prob)
    error = np.logical_xor(pred.astype(bool), gt.astype(bool))
    gt_boundary = boundary_band(gt, args.boundary_radius)
    pred_boundary = boundary_band(pred, args.boundary_radius)
    union_boundary = np.logical_or(gt_boundary, pred_boundary)
    total_uncertainty = float(entropy.sum()) + 1e-7
    error_uncertainty = float(entropy[error].sum()) if error.any() else 0.0

    return {
        "dice": dice(pred, gt),
        "boundary_dice": dice(pred_boundary, gt_boundary),
        "error_area_frac": float(error.mean()),
        "boundary_error_area_frac": float(np.logical_and(error, union_boundary).mean()),
        "lecr_uncertainty": error_uncertainty / total_uncertainty,
        "lecr_boundary_error": float(np.logical_and(error, union_boundary).sum() / (error.sum() + 1e-7)),
    }


def flatten_entry(patch_id: str, features: dict[str, dict[str, float]], metrics: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"patch_id": patch_id}
    for family, values in features.items():
        for name, value in values.items():
            row[f"feat__{family}__{name}"] = value
    for name, value in metrics.items():
        row[f"eval__{name}"] = value
    return row


def assert_no_leakage(features: dict[str, dict[str, float]]) -> None:
    forbidden = (
        "__gt__",
        "ground_truth",
        "false_positive",
        "false_negative",
        "__dice",
        "dice__",
        "__error",
        "error__",
        "__lecr",
        "lecr__",
    )
    for family, values in features.items():
        for name in values:
            lowered = f"{family}__{name}".lower()
            if any(token in lowered for token in forbidden):
                raise AssertionError(f"Deployable feature appears to leak evaluation information: {lowered}")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    add_confounder_project_to_path(workspace_root)

    from confounder_mining.dataset import NucleiPointDataset
    from confounder_mining.unet import SmallUNet

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = NucleiPointDataset(args.images_dir, args.masks_dir, augment=False)
    if args.limit > 0:
        dataset.samples = dataset.samples[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = SmallUNet(base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_map_dir is not None:
        args.output_map_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as jsonl:
        with torch.no_grad():
            for batch in loader:
                image = batch["image"].to(device)
                logits = model(image)
                prob_batch = torch.sigmoid(logits).cpu().numpy()
                mask_batch = batch["mask"].cpu().numpy()
                ids = list(batch["id"])
                for i, patch_id in enumerate(ids):
                    prob = prob_batch[i, 0]
                    pred = prob >= 0.5
                    gt = mask_batch[i, 0] > 0.5
                    entropy = binary_entropy(prob)
                    error = np.logical_xor(pred, gt)
                    features = deployable_features(prob, pred, args)
                    assert_no_leakage(features)
                    metrics = evaluation_metrics(prob, pred, gt, args)
                    if args.output_map_dir is not None:
                        np.savez_compressed(
                            args.output_map_dir / f"{safe_patch_filename(patch_id)}.npz",
                            prob=prob.astype(np.float32),
                            entropy=entropy.astype(np.float32),
                            pred=pred.astype(np.uint8),
                            gt=gt.astype(np.uint8),
                            error=error.astype(np.uint8),
                        )
                    entry = {
                        "patch_id": patch_id,
                        "deployable_features": features,
                        "evaluation_metrics": metrics,
                    }
                    jsonl.write(json.dumps(entry, ensure_ascii=False, default=json_safe) + "\n")
                    rows.append(flatten_entry(patch_id, features, metrics))

    if not rows:
        raise RuntimeError("No rows were extracted.")

    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote_csv={args.output_csv}")
    print(f"wrote_jsonl={args.output_jsonl}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
