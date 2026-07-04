from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-time channel masking for Boundary-SFRM-UNet.")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", choices=("cryonuseg", "consep"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_train_boundary_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "train_boundary_sfrm_unet",
        project_root / "scripts" / "train_boundary_sfrm_unet.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load train_boundary_sfrm_unet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_config(workspace_root: Path, dataset: str) -> dict[str, Path]:
    data_root = workspace_root / "data"
    run_root = workspace_root / "failure_region_reliability" / "experiments" / "boundary_sfrm_runs"
    if dataset == "cryonuseg":
        return {
            "images": data_root / "cryonuseg_val_split_full" / "images",
            "masks": data_root / "cryonuseg_val_split_full" / "masks",
            "confounders": data_root / "cryonuseg_val_split_confounders",
            "baseline_ckpt": run_root / "cryonuseg_fullsup_unet_seed42" / "best_boundary_model.pt",
            "sfrm_ckpt": run_root / "cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42" / "best_boundary_model.pt",
        }
    if dataset == "consep":
        return {
            "images": data_root / "consep_val_split_full" / "images",
            "masks": data_root / "consep_val_split_full" / "masks",
            "confounders": data_root / "consep_val_split_confounders",
            "baseline_ckpt": run_root / "consep_fullsup_unet_seed42" / "best_boundary_model.pt",
            "sfrm_ckpt": run_root / "consep_fullsup_learned_failure_head_v3_teacher10_seed42" / "best_boundary_model.pt",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def summarise(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys:
        out[key] = float(np.mean([float(row[key]) for row in rows]))
    return out


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    project_root = workspace_root / "failure_region_reliability"
    sys.path.insert(0, str(project_root))
    if not (project_root / "confounder_mining").is_dir():
        sys.path.insert(0, str((workspace_root / "confounder_prompting").resolve()))

    train_boundary = load_train_boundary_module(project_root)

    from confounder_mining.dataset import NucleiPointDataset
    from confounder_mining.instance_metrics import instance_metrics_from_binary
    from confounder_mining.unet import SmallUNet
    from src.metrics import boundary_band_numpy, dice_numpy
    from src.models import BoundarySFRMUNet

    cfg = dataset_config(workspace_root, args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sfrm_ckpt = torch.load(cfg["sfrm_ckpt"], map_location=device, weights_only=False)
    base_ckpt = torch.load(cfg["baseline_ckpt"], map_location=device, weights_only=False)
    ckpt_args = sfrm_ckpt["args"]
    variant = ckpt_args["variant"]
    if variant not in {"learned_failure_head", "learned_failure_head_calibrated"}:
        raise ValueError(f"Expected failure-head checkpoint, got variant={variant}")

    dataset = NucleiPointDataset(cfg["images"], cfg["masks"], confounders_dir=cfg["confounders"], augment=False)
    if args.limit > 0:
        dataset.samples = dataset.samples[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline_model = SmallUNet(base_channels=int(ckpt_args["base_channels"])).to(device)
    baseline_model.load_state_dict(base_ckpt["model"])
    baseline_model.eval()

    model = BoundarySFRMUNet(
        in_channels=3,
        base_channels=int(ckpt_args["base_channels"]),
        risk_channels=train_boundary.risk_channels_for_variant(variant),
    ).to(device)
    model.load_state_dict(sfrm_ckpt["model"])
    model.eval()

    channel_masks = {
        "full": torch.tensor([1.0, 1.0, 1.0, 1.0], device=device).view(1, 4, 1, 1),
        "drop_P": torch.tensor([0.0, 1.0, 1.0, 1.0], device=device).view(1, 4, 1, 1),
        "drop_H": torch.tensor([1.0, 0.0, 1.0, 1.0], device=device).view(1, 4, 1, 1),
        "drop_Fb": torch.tensor([1.0, 1.0, 0.0, 1.0], device=device).view(1, 4, 1, 1),
        "drop_Fs": torch.tensor([1.0, 1.0, 1.0, 0.0], device=device).view(1, 4, 1, 1),
    }
    metric_keys = [
        "dice",
        "boundary_dice",
        "aji",
        "pq",
        "confounder_fpr",
    ]
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            instances = batch["instance"].cpu().numpy()
            confounders = batch["confounder"].cpu().numpy()[:, 0]
            patch_ids = list(batch["id"])

            baseline_logits = baseline_model(image)
            baseline_prob = torch.sigmoid(baseline_logits)
            baseline_pred = baseline_prob.cpu().numpy()[:, 0] >= 0.5

            coarse_state = model.forward_coarse(image)
            coarse_prob = torch.sigmoid(coarse_state.coarse_logits)
            risk_tensor, _ = train_boundary.build_risk_tensor_batch(
                coarse_state=coarse_state,
                coarse_prob=coarse_prob,
                variant=variant,
                boundary_radius=int(ckpt_args["boundary_radius"]),
                uncertainty_quantile=float(ckpt_args["uncertainty_quantile"]),
                device=device,
            )

            target_np = target.cpu().numpy()[:, 0] >= 0.5

            for mask_name, mask_tensor in channel_masks.items():
                masked_risk = risk_tensor * mask_tensor
                refined_logits = model.decode_refine(coarse_state, masked_risk)
                refined_pred = torch.sigmoid(refined_logits).cpu().numpy()[:, 0] >= 0.5

                for i, patch_id in enumerate(patch_ids):
                    gt = target_np[i]
                    pred = refined_pred[i]
                    gt_boundary = boundary_band_numpy(gt, int(ckpt_args["boundary_radius"]))
                    pred_boundary = boundary_band_numpy(pred, int(ckpt_args["boundary_radius"]))
                    confounder_region = confounders[i] > 0.5
                    aji, pq = instance_metrics_from_binary(pred, instances[i])
                    row = {
                        "dataset": args.dataset,
                        "variant": mask_name,
                        "patch_id": patch_id,
                        "baseline_dice": dice_numpy(baseline_pred[i], gt),
                        "dice": dice_numpy(pred, gt),
                        "boundary_dice": dice_numpy(pred_boundary, gt_boundary),
                        "aji": aji,
                        "pq": pq,
                        "confounder_fpr": float(pred[confounder_region].mean()) if confounder_region.any() else 0.0,
                    }
                    grouped_rows[mask_name].append(row)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "checkpoint": str(cfg["sfrm_ckpt"]),
        "variant": variant,
        "sample_count": int(sum(len(v) for v in grouped_rows.values()) / max(len(grouped_rows), 1)),
        "channel_order": ["P", "H(P)", "F_b", "F_s"],
        "results": {},
    }

    full_summary = summarise(grouped_rows["full"], metric_keys)
    for mask_name, rows in grouped_rows.items():
        agg = summarise(rows, metric_keys)
        agg["delta_boundary_vs_full"] = agg["boundary_dice"] - full_summary["boundary_dice"]
        agg["delta_aji_vs_full"] = agg["aji"] - full_summary["aji"]
        agg["delta_pq_vs_full"] = agg["pq"] - full_summary["pq"]
        agg["delta_fpr_vs_full"] = agg["confounder_fpr"] - full_summary["confounder_fpr"]
        summary["results"][mask_name] = agg

    with (args.output_dir / f"{args.dataset}_channel_ablation_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    flat_rows: list[dict[str, Any]] = []
    for mask_name, rows in grouped_rows.items():
        flat_rows.extend(rows)
    with (args.output_dir / f"{args.dataset}_channel_ablation_per_case.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
