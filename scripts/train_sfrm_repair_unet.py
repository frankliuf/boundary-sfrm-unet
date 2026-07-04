from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    workspace_root = ROOT.parent
    parser = argparse.ArgumentParser(description="Train SFRM-guided repair U-Net.")
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument("--train-images", type=Path, default=workspace_root / "data/monuseg_train_split_patches/images")
    parser.add_argument("--train-masks", type=Path, default=workspace_root / "data/monuseg_train_split_patches/masks")
    parser.add_argument("--val-images", type=Path, default=workspace_root / "data/monuseg_val_split_patches/images")
    parser.add_argument("--val-masks", type=Path, default=workspace_root / "data/monuseg_val_split_patches/masks")
    parser.add_argument("--train-confounders", type=Path, default=workspace_root / "data/monuseg_train_split_confounders")
    parser.add_argument("--val-confounders", type=Path, default=workspace_root / "data/monuseg_val_split_confounders")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint from confounder_prompting/scripts/train_point_unet.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["rgb_prob", "entropy_only", "random_map", "sfrm_boundary", "sfrm_full"],
        default="sfrm_full",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--baseline-base-channels", type=int, default=24)
    parser.add_argument("--repair-base-channels", type=int, default=24)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--uncertainty-quantile", type=float, default=0.85)
    parser.add_argument("--risk-beta", type=float, default=2.0)
    parser.add_argument("--lambda-risk", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=0.2)
    parser.add_argument("--lambda-boundary", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_confounder_project_to_path(workspace_root: Path) -> None:
    candidates = (ROOT, workspace_root / "confounder_prompting")
    for candidate in candidates:
        if (candidate / "confounder_mining").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return
    raise FileNotFoundError("Cannot find the bundled or sibling confounder_mining package")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset: Any, limit: int) -> Any:
    if limit > 0:
        dataset.samples = dataset.samples[:limit]
    return dataset


def variant_input_channels(variant: str) -> int:
    if variant == "rgb_prob":
        return 4
    if variant == "entropy_only":
        return 5
    if variant == "random_map":
        return 9
    if variant == "sfrm_boundary":
        return 7
    if variant == "sfrm_full":
        return 10
    raise ValueError(f"Unsupported variant: {variant}")


def build_variant_tensors(
    image: torch.Tensor,
    baseline_prob: torch.Tensor,
    patch_ids: list[str],
    variant: str,
    boundary_radius: int,
    uncertainty_quantile: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from src.failure_regions import build_sfrm_risk_maps

    image_cpu = image.detach().cpu()
    prob_cpu = baseline_prob.detach().cpu().numpy()[:, 0]
    aux_channels: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    random_channels: list[torch.Tensor] = []

    for patch_id, prob in zip(patch_ids, prob_cpu, strict=True):
        risk_maps = build_sfrm_risk_maps(
            prob,
            boundary_radius=boundary_radius,
            uncertainty_quantile=uncertainty_quantile,
        )
        if variant == "rgb_prob":
            aux = [torch.from_numpy(prob[None])]
            gate = torch.ones((1, prob.shape[0], prob.shape[1]), dtype=torch.float32)
        elif variant == "entropy_only":
            aux = [
                torch.from_numpy(prob[None]),
                torch.from_numpy(risk_maps.entropy[None]),
            ]
            gate = torch.from_numpy(risk_maps.entropy[None] / max(float(risk_maps.entropy.max()), 1e-7))
        elif variant == "random_map":
            rng = np.random.default_rng(zlib.crc32(patch_id.encode("utf-8")) & 0xFFFFFFFF)
            random_stack = rng.random((5, prob.shape[0], prob.shape[1]), dtype=np.float32)
            aux = [torch.from_numpy(prob[None]), torch.from_numpy(random_stack)]
            gate = torch.from_numpy(random_stack[:1])
        elif variant == "sfrm_boundary":
            aux = [
                torch.from_numpy(prob[None]),
                torch.from_numpy(risk_maps.entropy[None]),
                torch.from_numpy(risk_maps.boundary_band[None]),
                torch.from_numpy(risk_maps.boundary_risk[None]),
            ]
            gate = torch.from_numpy(np.maximum(risk_maps.boundary_band, risk_maps.boundary_risk)[None])
        elif variant == "sfrm_full":
            aux = [
                torch.from_numpy(prob[None]),
                torch.from_numpy(risk_maps.entropy[None]),
                torch.from_numpy(risk_maps.boundary_band[None]),
                torch.from_numpy(risk_maps.boundary_risk[None]),
                torch.from_numpy(risk_maps.uncertainty_cluster[None]),
                torch.from_numpy(risk_maps.topology_risk[None]),
                torch.from_numpy(risk_maps.morphology_risk[None]),
            ]
            gate = torch.from_numpy(risk_maps.composite_risk[None])
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        aux_channels.append(torch.cat(aux, dim=0))
        gates.append(gate.float())

    aux_tensor = torch.stack(aux_channels, dim=0)
    gate_tensor = torch.stack(gates, dim=0).clamp(0.0, 1.0)
    repair_input = torch.cat([image_cpu, aux_tensor], dim=1).to(device=device, dtype=torch.float32)
    return repair_input, gate_tensor.to(device=device, dtype=torch.float32)


def target_boundary_weight(target: torch.Tensor, radius: int, boundary_weight: float) -> torch.Tensor:
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(target.float(), kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - target.float(), kernel_size=kernel_size, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return 1.0 + boundary_weight * boundary


def train_one_epoch(
    baseline_model: torch.nn.Module,
    repair_model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    from src.metrics import (
        preserve_low_risk_loss,
        soft_dice_loss_from_logits,
        weighted_bce_loss_from_logits,
    )

    repair_model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        patch_ids = list(batch["id"])

        with torch.no_grad():
            baseline_logits = baseline_model(image)
            baseline_prob = torch.sigmoid(baseline_logits)

        repair_input, gate = build_variant_tensors(
            image=image,
            baseline_prob=baseline_prob,
            patch_ids=patch_ids,
            variant=args.variant,
            boundary_radius=args.boundary_radius,
            uncertainty_quantile=args.uncertainty_quantile,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        corrected_logits, _ = repair_model(repair_input, baseline_logits, gate)
        base_bce = F.binary_cross_entropy_with_logits(corrected_logits, target.float())
        dice_loss = soft_dice_loss_from_logits(corrected_logits, target.float())
        risk_loss = weighted_bce_loss_from_logits(
            corrected_logits,
            target.float(),
            1.0 + args.risk_beta * gate,
        )
        boundary_loss = weighted_bce_loss_from_logits(
            corrected_logits,
            target.float(),
            target_boundary_weight(target, args.boundary_radius, args.boundary_weight),
        )
        preserve_loss = preserve_low_risk_loss(corrected_logits, baseline_logits, gate)
        loss = (
            base_bce
            + dice_loss
            + args.lambda_risk * risk_loss
            + args.lambda_boundary * boundary_loss
            + args.lambda_preserve * preserve_loss
        )
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * image.shape[0]

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    baseline_model: torch.nn.Module,
    repair_model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    from confounder_mining.instance_metrics import instance_metrics_from_binary
    from src.metrics import boundary_band_numpy, dice_numpy, masked_dice_numpy, masked_error_rate_numpy

    repair_model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        instances = batch["instance"].cpu().numpy()[:, 0]
        confounders = batch["confounder"].cpu().numpy()[:, 0]
        patch_ids = list(batch["id"])

        baseline_logits = baseline_model(image)
        baseline_prob = torch.sigmoid(baseline_logits)
        repair_input, gate = build_variant_tensors(
            image=image,
            baseline_prob=baseline_prob,
            patch_ids=patch_ids,
            variant=args.variant,
            boundary_radius=args.boundary_radius,
            uncertainty_quantile=args.uncertainty_quantile,
            device=device,
        )
        corrected_logits, residual_logits = repair_model(repair_input, baseline_logits, gate)

        baseline_pred = (torch.sigmoid(baseline_logits).cpu().numpy()[:, 0] >= 0.5)
        corrected_pred = (torch.sigmoid(corrected_logits).cpu().numpy()[:, 0] >= 0.5)
        target_np = target.cpu().numpy()[:, 0] >= 0.5
        gate_np = gate.cpu().numpy()[:, 0]

        for i, patch_id in enumerate(patch_ids):
            gt = target_np[i]
            baseline_mask = baseline_pred[i]
            corrected_mask = corrected_pred[i]
            gt_boundary = boundary_band_numpy(gt, args.boundary_radius)
            baseline_boundary = boundary_band_numpy(baseline_mask, args.boundary_radius)
            corrected_boundary = boundary_band_numpy(corrected_mask, args.boundary_radius)
            high_risk = gate_np[i] >= 0.5
            low_risk = ~high_risk

            base_aji, base_pq = instance_metrics_from_binary(baseline_mask, instances[i])
            repaired_aji, repaired_pq = instance_metrics_from_binary(corrected_mask, instances[i])
            confounder_region = confounders[i] > 0.5
            baseline_confounder_fpr = float(baseline_mask[confounder_region].mean()) if confounder_region.any() else 0.0
            repaired_confounder_fpr = float(corrected_mask[confounder_region].mean()) if confounder_region.any() else 0.0

            row = {
                "patch_id": patch_id,
                "baseline_dice": dice_numpy(baseline_mask, gt),
                "repaired_dice": dice_numpy(corrected_mask, gt),
                "baseline_boundary_dice": dice_numpy(baseline_boundary, gt_boundary),
                "repaired_boundary_dice": dice_numpy(corrected_boundary, gt_boundary),
                "baseline_aji": base_aji,
                "repaired_aji": repaired_aji,
                "baseline_pq": base_pq,
                "repaired_pq": repaired_pq,
                "baseline_high_risk_dice": masked_dice_numpy(baseline_mask, gt, high_risk),
                "repaired_high_risk_dice": masked_dice_numpy(corrected_mask, gt, high_risk),
                "baseline_low_risk_dice": masked_dice_numpy(baseline_mask, gt, low_risk),
                "repaired_low_risk_dice": masked_dice_numpy(corrected_mask, gt, low_risk),
                "baseline_high_risk_error": masked_error_rate_numpy(baseline_mask, gt, high_risk),
                "repaired_high_risk_error": masked_error_rate_numpy(corrected_mask, gt, high_risk),
                "baseline_low_risk_error": masked_error_rate_numpy(baseline_mask, gt, low_risk),
                "repaired_low_risk_error": masked_error_rate_numpy(corrected_mask, gt, low_risk),
                "baseline_confounder_fpr": baseline_confounder_fpr,
                "repaired_confounder_fpr": repaired_confounder_fpr,
                "mean_gate": float(gate_np[i].mean()),
                "high_risk_frac": float(high_risk.mean()),
                "mean_abs_residual_logit": float(residual_logits[i].abs().mean().item()),
            }
            row["delta_dice"] = row["repaired_dice"] - row["baseline_dice"]
            row["delta_boundary_dice"] = row["repaired_boundary_dice"] - row["baseline_boundary_dice"]
            row["delta_aji"] = row["repaired_aji"] - row["baseline_aji"]
            row["delta_pq"] = row["repaired_pq"] - row["baseline_pq"]
            row["delta_high_risk_dice"] = row["repaired_high_risk_dice"] - row["baseline_high_risk_dice"]
            row["delta_low_risk_dice"] = row["repaired_low_risk_dice"] - row["baseline_low_risk_dice"]
            row["delta_high_risk_error"] = row["repaired_high_risk_error"] - row["baseline_high_risk_error"]
            row["delta_low_risk_error"] = row["repaired_low_risk_error"] - row["baseline_low_risk_error"]
            row["delta_confounder_fpr"] = row["repaired_confounder_fpr"] - row["baseline_confounder_fpr"]
            rows.append(row)

    if not rows:
        raise RuntimeError("No evaluation rows were produced.")

    summary_keys = [
        "baseline_dice",
        "repaired_dice",
        "delta_dice",
        "baseline_boundary_dice",
        "repaired_boundary_dice",
        "delta_boundary_dice",
        "baseline_aji",
        "repaired_aji",
        "delta_aji",
        "baseline_pq",
        "repaired_pq",
        "delta_pq",
        "baseline_high_risk_dice",
        "repaired_high_risk_dice",
        "delta_high_risk_dice",
        "baseline_low_risk_dice",
        "repaired_low_risk_dice",
        "delta_low_risk_dice",
        "baseline_high_risk_error",
        "repaired_high_risk_error",
        "delta_high_risk_error",
        "baseline_low_risk_error",
        "repaired_low_risk_error",
        "delta_low_risk_error",
        "baseline_confounder_fpr",
        "repaired_confounder_fpr",
        "delta_confounder_fpr",
        "mean_gate",
        "high_risk_frac",
        "mean_abs_residual_logit",
    ]
    summary = {key: float(np.mean([float(row[key]) for row in rows])) for key in summary_keys}
    return summary, rows


def save_checkpoint(
    path: Path,
    repair_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "repair_model": repair_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    workspace_root = args.workspace_root.resolve()
    add_confounder_project_to_path(workspace_root)

    from confounder_mining.dataset import NucleiPointDataset
    from confounder_mining.unet import SmallUNet
    from src.models import RepairUNet, ResidualRepairWrapper

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = maybe_limit_dataset(
        NucleiPointDataset(
            args.train_images,
            args.train_masks,
            confounders_dir=args.train_confounders,
            augment=True,
        ),
        args.limit_train,
    )
    val_dataset = maybe_limit_dataset(
        NucleiPointDataset(
            args.val_images,
            args.val_masks,
            confounders_dir=args.val_confounders,
            augment=False,
        ),
        args.limit_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    baseline_model = SmallUNet(base_channels=args.baseline_base_channels).to(device)
    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_checkpoint["model"])
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad = False

    repair_net = RepairUNet(
        in_channels=variant_input_channels(args.variant),
        base_channels=args.repair_base_channels,
    )
    repair_model = ResidualRepairWrapper(repair_net, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(repair_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / "history.csv"
    best_boundary = -1.0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "baseline_dice",
            "repaired_dice",
            "delta_dice",
            "baseline_boundary_dice",
            "repaired_boundary_dice",
            "delta_boundary_dice",
            "baseline_aji",
            "repaired_aji",
            "delta_aji",
            "baseline_pq",
            "repaired_pq",
            "delta_pq",
            "delta_high_risk_dice",
            "delta_low_risk_dice",
            "delta_high_risk_error",
            "delta_low_risk_error",
            "delta_confounder_fpr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                baseline_model=baseline_model,
                repair_model=repair_model,
                loader=train_loader,
                optimizer=optimizer,
                args=args,
                device=device,
            )
            summary, rows = evaluate(
                baseline_model=baseline_model,
                repair_model=repair_model,
                loader=val_loader,
                args=args,
                device=device,
            )
            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                **{key: f"{summary[key]:.6f}" for key in fieldnames if key not in {"epoch", "train_loss"}},
            }
            writer.writerow(row)
            handle.flush()

            per_sample_path = args.output_dir / f"val_epoch_{epoch:02d}_per_sample.csv"
            with per_sample_path.open("w", newline="", encoding="utf-8") as sample_handle:
                sample_writer = csv.DictWriter(sample_handle, fieldnames=list(rows[0].keys()))
                sample_writer.writeheader()
                sample_writer.writerows(rows)

            metrics_path = args.output_dir / f"val_epoch_{epoch:02d}_summary.json"
            metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            save_checkpoint(args.output_dir / "last_model.pt", repair_model, optimizer, epoch, summary, args)
            if summary["repaired_boundary_dice"] > best_boundary:
                best_boundary = summary["repaired_boundary_dice"]
                save_checkpoint(args.output_dir / "best_boundary_model.pt", repair_model, optimizer, epoch, summary, args)

            print(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "delta_dice": round(summary["delta_dice"], 6),
                    "delta_boundary_dice": round(summary["delta_boundary_dice"], 6),
                    "delta_aji": round(summary["delta_aji"], 6),
                    "delta_pq": round(summary["delta_pq"], 6),
                    "delta_high_risk_dice": round(summary["delta_high_risk_dice"], 6),
                    "delta_low_risk_dice": round(summary["delta_low_risk_dice"], 6),
                    "delta_high_risk_error": round(summary["delta_high_risk_error"], 6),
                    "delta_low_risk_error": round(summary["delta_low_risk_error"], 6),
                }
            )

    print(f"history={history_path}")
    print(f"best_boundary={best_boundary:.6f}")


if __name__ == "__main__":
    main()
