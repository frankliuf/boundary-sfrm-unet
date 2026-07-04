from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    workspace_root = ROOT.parent
    parser = argparse.ArgumentParser(description="Train fully supervised U-Net baseline.")
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument("--train-images", type=Path, default=workspace_root / "data/monuseg_train_split_patches/images")
    parser.add_argument("--train-masks", type=Path, default=workspace_root / "data/monuseg_train_split_patches/masks")
    parser.add_argument("--val-images", type=Path, default=workspace_root / "data/monuseg_val_split_patches/images")
    parser.add_argument("--val-masks", type=Path, default=workspace_root / "data/monuseg_val_split_patches/masks")
    parser.add_argument("--train-confounders", type=Path, default=workspace_root / "data/monuseg_train_split_confounders")
    parser.add_argument("--val-confounders", type=Path, default=workspace_root / "data/monuseg_val_split_confounders")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_confounder_project_to_path(workspace_root: Path) -> None:
    candidates = (ROOT, workspace_root / "confounder_prompting")
    for candidate in candidates:
        if (candidate / "confounder_mining").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return
    raise FileNotFoundError("Cannot find the bundled or sibling confounder_mining package")


def maybe_limit_dataset(dataset, limit: int):
    if limit > 0:
        dataset.samples = dataset.samples[:limit]
    return dataset


def target_boundary_weight(target: torch.Tensor, radius: int, boundary_weight: float = 2.0) -> torch.Tensor:
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(target.float(), kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - target.float(), kernel_size=kernel_size, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return 1.0 + boundary_weight * boundary


def train_one_epoch(model, loader: DataLoader, optimizer, boundary_radius: int, device: torch.device) -> float:
    from src.metrics import soft_dice_loss_from_logits, weighted_bce_loss_from_logits

    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = F.binary_cross_entropy_with_logits(logits, target.float())
        loss = loss + soft_dice_loss_from_logits(logits, target.float())
        loss = loss + 0.5 * weighted_bce_loss_from_logits(
            logits,
            target.float(),
            target_boundary_weight(target, boundary_radius),
        )
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * image.shape[0]
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader: DataLoader, boundary_radius: int, device: torch.device) -> dict[str, float]:
    from confounder_mining.instance_metrics import instance_metrics_from_binary
    from src.metrics import boundary_band_numpy, dice_numpy

    model.eval()
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        instances = batch["instance"].cpu().numpy()[:, 0]
        confounders = batch["confounder"].cpu().numpy()[:, 0]

        logits = model(image)
        pred = (torch.sigmoid(logits).cpu().numpy()[:, 0] >= 0.5)
        target_np = target.cpu().numpy()[:, 0] >= 0.5

        for i in range(pred.shape[0]):
            pred_mask = pred[i]
            gt = target_np[i]
            pred_boundary = boundary_band_numpy(pred_mask, boundary_radius)
            gt_boundary = boundary_band_numpy(gt, boundary_radius)
            aji, pq = instance_metrics_from_binary(pred_mask, instances[i])
            confounder_region = confounders[i] > 0.5
            confounder_fpr = float(pred_mask[confounder_region].mean()) if confounder_region.any() else 0.0
            rows.append(
                {
                    "dice": dice_numpy(pred_mask, gt),
                    "boundary_dice": dice_numpy(pred_boundary, gt_boundary),
                    "aji": aji,
                    "pq": pq,
                    "confounder_fpr": confounder_fpr,
                }
            )

    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0].keys()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict[str, float], args: argparse.Namespace) -> None:
    torch.save(
        {
            "model": model.state_dict(),
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / "history.csv"
    best_boundary = -1.0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["epoch", "train_loss", "dice", "boundary_dice", "aji", "pq", "confounder_fpr"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, args.boundary_radius, device)
            metrics = evaluate(model, val_loader, args.boundary_radius, device)
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss
            writer.writerow({key: metrics.get(key) for key in fieldnames})
            handle.flush()

            with (args.output_dir / f"val_epoch_{epoch:02d}_summary.json").open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, indent=2)

            save_checkpoint(args.output_dir / "last_model.pt", model, optimizer, epoch, metrics, args)
            if metrics["boundary_dice"] > best_boundary:
                best_boundary = metrics["boundary_dice"]
                save_checkpoint(args.output_dir / "best_boundary_model.pt", model, optimizer, epoch, metrics, args)


if __name__ == "__main__":
    main()
