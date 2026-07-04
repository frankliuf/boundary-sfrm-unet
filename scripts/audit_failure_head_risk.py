from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit learned failure-head risk calibration.")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--confounders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-thresholds", type=float, nargs="*", default=[0.45, 0.47])
    parser.add_argument("--norm-thresholds", type=float, nargs="*", default=[0.8, 0.9])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def robust_norm(x: np.ndarray, low_q: float = 0.10, high_q: float = 0.99) -> np.ndarray:
    q_low = float(np.quantile(x, low_q))
    q_high = float(np.quantile(x, high_q))
    if q_high - q_low < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - q_low) / (q_high - q_low)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def mask_stats(risk: np.ndarray, target: np.ndarray, threshold: float) -> tuple[float, float, float]:
    pred = risk >= threshold
    pos = target > 0.5
    tp = float(np.logical_and(pred, pos).sum())
    fp = float(np.logical_and(pred, ~pos).sum())
    fn = float(np.logical_and(~pred, pos).sum())
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    frac = float(pred.mean())
    return recall, precision, frac


def flatten_summary(x: np.ndarray) -> dict[str, float]:
    flat = x.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(flat.max()),
    }


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    project_root = workspace_root / "failure_region_reliability"
    sys.path.insert(0, str(project_root))
    if not (project_root / "confounder_mining").is_dir():
        sys.path.insert(0, str((workspace_root / "confounder_prompting").resolve()))

    spec = importlib.util.spec_from_file_location("train_boundary_sfrm_unet", project_root / "scripts" / "train_boundary_sfrm_unet.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load train_boundary_sfrm_unet.py")
    train_boundary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_boundary)

    from confounder_mining.dataset import NucleiPointDataset
    from src.models import BoundarySFRMUNet

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    variant = ckpt_args["variant"]
    if variant not in {"learned_failure_head", "learned_failure_head_calibrated"}:
        raise ValueError(f"Checkpoint variant must be failure-head based, got {variant}")

    model = BoundarySFRMUNet(
        in_channels=3,
        base_channels=int(ckpt_args["base_channels"]),
        risk_channels=train_boundary.risk_channels_for_variant(variant),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = NucleiPointDataset(args.images, args.masks, confounders_dir=args.confounders, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    raw_primary_all: list[np.ndarray] = []
    norm_primary_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    rows: list[dict[str, float | str]] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            instances = batch["instance"].to(device)
            confounders = batch["confounder"].to(device)

            coarse_state = model.forward_coarse(image)
            coarse_prob = torch.sigmoid(coarse_state.coarse_logits)
            failure_prob = torch.sigmoid(coarse_state.failure_logits)
            raw_primary = failure_prob.max(dim=1).values.cpu().numpy()[0]
            norm_primary = robust_norm(raw_primary)
            failure_target = train_boundary.build_failure_target_batch(
                coarse_prob=coarse_prob,
                target=target,
                instances=instances,
                confounders=confounders,
                boundary_radius=int(ckpt_args["boundary_radius"]),
                contact_restrict=bool(ckpt_args["failure_contact_restrict"]),
                device=device,
            ).max(dim=1).values.cpu().numpy()[0]

            raw_primary_all.append(raw_primary[None])
            norm_primary_all.append(norm_primary[None])
            target_all.append(failure_target[None])

            row: dict[str, float | str] = {
                "patch_id": str(batch["id"][0]),
                "raw_mean": float(raw_primary.mean()),
                "raw_max": float(raw_primary.max()),
                "raw_q99": float(np.quantile(raw_primary, 0.99)),
                "norm_mean": float(norm_primary.mean()),
                "norm_max": float(norm_primary.max()),
                "target_frac": float(failure_target.mean()),
            }
            for thr in args.raw_thresholds:
                recall, precision, frac = mask_stats(raw_primary, failure_target, float(thr))
                row[f"raw_recall_at_{thr:.2f}"] = recall
                row[f"raw_precision_at_{thr:.2f}"] = precision
                row[f"raw_frac_at_{thr:.2f}"] = frac
            for thr in args.norm_thresholds:
                recall, precision, frac = mask_stats(norm_primary, failure_target, float(thr))
                row[f"norm_recall_at_{thr:.2f}"] = recall
                row[f"norm_precision_at_{thr:.2f}"] = precision
                row[f"norm_frac_at_{thr:.2f}"] = frac
            rows.append(row)

    raw_primary_np = np.concatenate(raw_primary_all, axis=0)
    norm_primary_np = np.concatenate(norm_primary_all, axis=0)
    target_np = np.concatenate(target_all, axis=0)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "variant": variant,
        "sample_count": int(len(rows)),
        "raw_primary": flatten_summary(raw_primary_np),
        "normalized_primary": flatten_summary(norm_primary_np),
        "target_fraction": flatten_summary(target_np),
        "thresholds": {},
    }

    for thr in args.raw_thresholds:
        recalls, precisions, fracs = [], [], []
        for row in rows:
            recalls.append(float(row[f"raw_recall_at_{thr:.2f}"]))
            precisions.append(float(row[f"raw_precision_at_{thr:.2f}"]))
            fracs.append(float(row[f"raw_frac_at_{thr:.2f}"]))
        summary["thresholds"][f"raw@{thr:.2f}"] = {
            "mean_recall": float(np.mean(recalls)),
            "mean_precision": float(np.mean(precisions)),
            "mean_high_risk_frac": float(np.mean(fracs)),
        }

    for thr in args.norm_thresholds:
        recalls, precisions, fracs = [], [], []
        for row in rows:
            recalls.append(float(row[f"norm_recall_at_{thr:.2f}"]))
            precisions.append(float(row[f"norm_precision_at_{thr:.2f}"]))
            fracs.append(float(row[f"norm_frac_at_{thr:.2f}"]))
        summary["thresholds"][f"norm@{thr:.2f}"] = {
            "mean_recall": float(np.mean(recalls)),
            "mean_precision": float(np.mean(precisions)),
            "mean_high_risk_frac": float(np.mean(fracs)),
        }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    with (args.output_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
