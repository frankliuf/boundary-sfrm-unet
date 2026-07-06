from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    workspace_root = ROOT.parent
    parser = argparse.ArgumentParser(description="Evaluate Boundary-SFRM-UNet and baseline on a held-out split.")
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--confounders", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--sfrm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_metadata_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["id"]: row for row in rows}


def mean_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    from confounder_mining.dataset import NucleiPointDataset
    from confounder_mining.unet import SmallUNet
    from scripts.train_boundary_sfrm_unet import evaluate, risk_channels_for_variant
    from src.models import BoundarySFRMUNet

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_map = load_metadata_map(args.metadata)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sfrm_ckpt = torch.load(args.sfrm_checkpoint, map_location=device, weights_only=False)
    baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)

    sfrm_args = sfrm_ckpt["args"]
    variant = sfrm_args["variant"]
    base_channels = int(sfrm_args["base_channels"])
    boundary_radius = int(sfrm_args["boundary_radius"])
    uncertainty_quantile = float(sfrm_args["uncertainty_quantile"])

    test_dataset = NucleiPointDataset(
        args.images,
        args.masks,
        confounders_dir=args.confounders,
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    reference_model = SmallUNet(base_channels=base_channels).to(device)
    reference_model.load_state_dict(baseline_ckpt["model"])
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    model = BoundarySFRMUNet(
        in_channels=3,
        base_channels=base_channels,
        risk_channels=risk_channels_for_variant(variant),
    ).to(device)
    model.load_state_dict(sfrm_ckpt["model"])
    model.eval()

    eval_args = argparse.Namespace(
        variant=variant,
        boundary_radius=boundary_radius,
        uncertainty_quantile=uncertainty_quantile,
    )
    summary, rows = evaluate(
        reference_model=reference_model,
        model=model,
        loader=test_loader,
        args=eval_args,
        device=device,
    )

    for row in rows:
        patch_id = str(row["patch_id"])
        meta = metadata_map.get(patch_id)
        if meta is None:
            raise KeyError(f"Missing metadata for patch/test ID: {patch_id}")
        row["source"] = meta.get("source", "")

    source_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        source_groups[str(row["source"])].append(row)

    metric_keys = [
        "baseline_dice",
        "refined_dice",
        "delta_dice",
        "baseline_boundary_dice",
        "refined_boundary_dice",
        "delta_boundary_dice",
        "baseline_aji",
        "refined_aji",
        "delta_aji",
        "baseline_pq",
        "refined_pq",
        "delta_pq",
        "baseline_confounder_fpr",
        "refined_confounder_fpr",
        "delta_confounder_fpr",
    ]
    per_source_rows: list[dict[str, object]] = []
    for source_id, source_rows in sorted(source_groups.items()):
        aggregated = mean_rows(source_rows, metric_keys)
        aggregated["source"] = source_id
        aggregated["n_patches"] = len(source_rows)
        per_source_rows.append({"source": source_id, "n_patches": len(source_rows), **aggregated})

    summary_payload = {
        "split_images": str(args.images),
        "split_metadata": str(args.metadata),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "sfrm_checkpoint": str(args.sfrm_checkpoint),
        "variant": variant,
        "base_channels": base_channels,
        "boundary_radius": boundary_radius,
        "uncertainty_quantile": uncertainty_quantile,
        "n_samples": len(rows),
        "n_sources": len(per_source_rows),
        **summary,
    }

    write_csv(args.output_dir / "test_per_sample.csv", rows)
    write_csv(args.output_dir / "test_per_source.csv", per_source_rows)
    (args.output_dir / "test_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
