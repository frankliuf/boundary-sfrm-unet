from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage import measure
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    workspace_root = ROOT.parent
    parser = argparse.ArgumentParser(description="Train Boundary-SFRM-UNet.")
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument("--train-images", type=Path, default=workspace_root / "data/monuseg_train_split_patches/images")
    parser.add_argument("--train-masks", type=Path, default=workspace_root / "data/monuseg_train_split_patches/masks")
    parser.add_argument("--val-images", type=Path, default=workspace_root / "data/monuseg_val_split_patches/images")
    parser.add_argument("--val-masks", type=Path, default=workspace_root / "data/monuseg_val_split_patches/masks")
    parser.add_argument("--train-confounders", type=Path, default=workspace_root / "data/monuseg_train_split_confounders")
    parser.add_argument("--val-confounders", type=Path, default=workspace_root / "data/monuseg_val_split_confounders")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["two_pass_no_risk", "entropy_only", "boundary_sfrm", "boundary_sfrm_v2", "boundary_sfrm_v3", "boundary_sfrm_v4", "boundary_sfrm_v5", "learned_failure_head", "learned_failure_head_calibrated", "full_sfrm"],
        default="boundary_sfrm",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--uncertainty-quantile", type=float, default=0.85)
    parser.add_argument("--freeze-coarse-epochs", type=int, default=2)
    parser.add_argument("--lambda-coarse", type=float, default=0.5)
    parser.add_argument("--lambda-boundary", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=0.2)
    parser.add_argument("--lambda-risk-focus", type=float, default=0.0)
    parser.add_argument("--lambda-failure", type=float, default=0.5)
    parser.add_argument("--failure-teacher-mix", type=float, default=0.0)
    parser.add_argument("--failure-structure-gate-scale", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--risk-focus-weight", type=float, default=2.0)
    parser.add_argument("--failure-positive-weight", type=float, default=4.0)
    parser.add_argument("--failure-contact-restrict", action="store_true")
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


def risk_channels_for_variant(variant: str) -> int:
    if variant == "two_pass_no_risk":
        return 1
    if variant == "entropy_only":
        return 2
    if variant == "learned_failure_head":
        return 4
    if variant == "learned_failure_head_calibrated":
        return 4
    if variant == "boundary_sfrm":
        return 4
    if variant == "boundary_sfrm_v2":
        return 6
    if variant == "boundary_sfrm_v3":
        return 7
    if variant == "boundary_sfrm_v4":
        return 8
    if variant == "boundary_sfrm_v5":
        return 8
    if variant == "full_sfrm":
        return 9
    raise ValueError(f"Unsupported variant: {variant}")


def binary_entropy_numpy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def normalize_numpy(values: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / (hi - lo + eps)


def robust_normalize_tensor(
    tensor: torch.Tensor,
    low_q: float = 0.10,
    high_q: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    flat = tensor.flatten(start_dim=2)
    lo = torch.quantile(flat, low_q, dim=2, keepdim=True)
    hi = torch.quantile(flat, high_q, dim=2, keepdim=True)
    normalized = (flat - lo) / (hi - lo + eps)
    return normalized.clamp(0.0, 1.0).view_as(tensor)


def instance_contact_zone_numpy(inst_map: np.ndarray, radius: int) -> np.ndarray:
    labels = np.asarray(inst_map, dtype=np.int32)
    foreground = labels > 0
    if foreground.sum() == 0:
        return np.zeros(labels.shape, dtype=bool)

    distances, indices = ndi.distance_transform_edt(~foreground, return_indices=True)
    nearest_labels = labels[tuple(indices)]
    neighborhood_max = ndi.maximum_filter(nearest_labels, size=3)
    neighborhood_min = ndi.minimum_filter(nearest_labels, size=3)
    ridge = (
        (distances <= float(radius))
        & (nearest_labels > 0)
        & (neighborhood_max != neighborhood_min)
    )
    ridge = ndi.binary_dilation(ridge, iterations=1)
    return ridge.astype(bool)


def target_boundary_weight(target: torch.Tensor, radius: int, boundary_weight: float) -> torch.Tensor:
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(target.float(), kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - target.float(), kernel_size=kernel_size, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return 1.0 + boundary_weight * boundary


def build_failure_target_batch(
    coarse_prob: torch.Tensor,
    target: torch.Tensor,
    instances: torch.Tensor,
    confounders: torch.Tensor,
    boundary_radius: int,
    contact_restrict: bool,
    device: torch.device,
) -> torch.Tensor:
    from src.metrics import boundary_band_numpy

    coarse_np = coarse_prob.detach().cpu().numpy()[:, 0]
    target_np = target.detach().cpu().numpy()[:, 0] > 0.5
    instance_np = instances.detach().cpu().numpy()[:, 0]
    conf_np = confounders.detach().cpu().numpy()[:, 0] > 0.5

    outputs: list[torch.Tensor] = []
    for prob, gt, inst_map, conf in zip(coarse_np, target_np, instance_np, conf_np, strict=True):
        pred = prob >= 0.5
        error = np.logical_xor(pred, gt)
        boundary_region = np.logical_or(
            boundary_band_numpy(pred, boundary_radius),
            boundary_band_numpy(gt, boundary_radius),
        )
        boundary_fail = np.logical_and(error, boundary_region)
        gt_contact_zone = instance_contact_zone_numpy(inst_map, radius=max(boundary_radius + 1, 3))
        gt_contact_zone = ndi.binary_dilation(gt_contact_zone, iterations=1)
        support_region = np.logical_or(gt_contact_zone, boundary_region)
        false_bridge = np.logical_and(pred, ~gt)
        confounder_leak = np.logical_and(false_bridge, conf)
        if contact_restrict:
            confounder_leak = np.logical_and(confounder_leak, support_region)
        structure_fail = confounder_leak.copy()
        pred_labels = measure.label(pred, connectivity=2)
        for region in measure.regionprops(pred_labels):
            component_mask = pred_labels == region.label
            overlap_labels = [int(v) for v in np.unique(inst_map[component_mask]) if int(v) != 0]
            if len(overlap_labels) < 2:
                continue
            distance = ndi.distance_transform_edt(component_mask).astype(np.float32)
            neckness = 1.0 - normalize_numpy(distance)
            component_bridge = np.logical_and(false_bridge, component_mask)
            merge_focus = np.logical_and(component_bridge, gt_contact_zone)
            if merge_focus.sum() < max(6, int(0.02 * component_mask.sum())):
                merge_focus = np.logical_and(component_bridge, neckness >= 0.6)
            if merge_focus.sum() < max(4, int(0.01 * component_mask.sum())):
                merge_focus = component_bridge
            if contact_restrict:
                merge_focus = np.logical_and(merge_focus, support_region)
            structure_fail = np.logical_or(structure_fail, merge_focus)

        structure_fail = ndi.binary_dilation(structure_fail, iterations=1)
        if contact_restrict:
            structure_fail = np.logical_and(structure_fail, ndi.binary_dilation(support_region, iterations=1))
        structure_fail = np.logical_and(structure_fail, np.logical_or(pred, gt_contact_zone))

        failure_target = np.stack(
            [
                boundary_fail.astype(np.float32),
                structure_fail.astype(np.float32),
            ],
            axis=0,
        )
        outputs.append(torch.from_numpy(failure_target))

    return torch.stack(outputs, dim=0).to(device=device, dtype=torch.float32)


def build_risk_tensor_batch(
    coarse_state: Any,
    coarse_prob: torch.Tensor,
    variant: str,
    boundary_radius: int,
    uncertainty_quantile: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "learned_failure_head" or variant == "learned_failure_head_calibrated":
        entropy = torch.from_numpy(binary_entropy_numpy(coarse_prob.detach().cpu().numpy()[:, 0])).to(
            device=device,
            dtype=torch.float32,
        )[:, None]
        entropy = entropy / entropy.amax(dim=(2, 3), keepdim=True).clamp_min(1e-7)
        failure_prob = torch.sigmoid(coarse_state.failure_logits)
        if variant == "learned_failure_head_calibrated":
            calibrated_failure_prob = robust_normalize_tensor(failure_prob)
            primary_risk = calibrated_failure_prob.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)
            risk_tensor = torch.cat([coarse_prob, entropy, calibrated_failure_prob], dim=1)
        else:
            primary_risk = failure_prob.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)
            risk_tensor = torch.cat([coarse_prob, entropy, failure_prob], dim=1)
        return risk_tensor, primary_risk

    from src.failure_regions import build_sfrm_risk_maps

    prob_cpu = coarse_prob.detach().cpu().numpy()[:, 0]
    feature_cpu = None
    if variant == "boundary_sfrm_v4":
        feature_cpu = coarse_state.e1.detach().cpu().numpy()
    elif variant == "boundary_sfrm_v5":
        feature_cpu = coarse_state.coarse_d1.detach().cpu().numpy()
    risk_channels: list[torch.Tensor] = []
    primary_risks: list[torch.Tensor] = []

    for idx, prob in enumerate(prob_cpu):
        feature_map = feature_cpu[idx] if feature_cpu is not None else None
        risk_maps = build_sfrm_risk_maps(
            prob,
            feature_map=feature_map,
            boundary_radius=boundary_radius,
            uncertainty_quantile=uncertainty_quantile,
        )
        if variant == "two_pass_no_risk":
            risk = np.zeros((1, prob.shape[0], prob.shape[1]), dtype=np.float32)
            primary = np.zeros((1, prob.shape[0], prob.shape[1]), dtype=np.float32)
        elif variant == "entropy_only":
            entropy_norm = risk_maps.entropy / max(float(risk_maps.entropy.max()), 1e-7)
            risk = np.stack([prob, entropy_norm], axis=0).astype(np.float32)
            primary = entropy_norm[None].astype(np.float32)
        elif variant == "boundary_sfrm":
            risk = np.stack(
                [
                    prob,
                    risk_maps.entropy,
                    risk_maps.boundary_band,
                    risk_maps.boundary_risk,
                ],
                axis=0,
            ).astype(np.float32)
            primary = np.maximum(risk_maps.boundary_band, risk_maps.boundary_risk)[None].astype(np.float32)
        elif variant == "boundary_sfrm_v2":
            risk = np.stack(
                [
                    prob,
                    risk_maps.entropy,
                    risk_maps.soft_boundary_weight,
                    risk_maps.local_uncertainty,
                    risk_maps.probability_gradient,
                    risk_maps.soft_boundary_risk,
                ],
                axis=0,
            ).astype(np.float32)
            primary = risk_maps.soft_boundary_risk[None].astype(np.float32)
        elif variant == "boundary_sfrm_v3":
            risk = np.stack(
                [
                    prob,
                    risk_maps.entropy,
                    risk_maps.soft_boundary_weight,
                    risk_maps.local_uncertainty,
                    risk_maps.probability_gradient,
                    risk_maps.soft_boundary_risk,
                    risk_maps.split_risk,
                ],
                axis=0,
            ).astype(np.float32)
            primary = np.maximum(risk_maps.soft_boundary_risk, risk_maps.split_risk)[None].astype(np.float32)
        elif variant == "boundary_sfrm_v4" or variant == "boundary_sfrm_v5":
            risk = np.stack(
                [
                    prob,
                    risk_maps.entropy,
                    risk_maps.soft_boundary_weight,
                    risk_maps.local_uncertainty,
                    risk_maps.probability_gradient,
                    risk_maps.soft_boundary_risk,
                    risk_maps.split_risk,
                    risk_maps.feature_split_risk,
                ],
                axis=0,
            ).astype(np.float32)
            primary = np.maximum(risk_maps.soft_boundary_risk, risk_maps.feature_split_risk)[None].astype(np.float32)
        elif variant == "full_sfrm":
            risk = np.stack(
                [
                    prob,
                    risk_maps.entropy,
                    risk_maps.boundary_band,
                    risk_maps.boundary_risk,
                    risk_maps.split_risk,
                    risk_maps.feature_split_risk,
                    risk_maps.uncertainty_cluster,
                    risk_maps.topology_risk,
                    risk_maps.morphology_risk,
                ],
                axis=0,
            ).astype(np.float32)
            primary = risk_maps.composite_risk[None].astype(np.float32)
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        risk_channels.append(torch.from_numpy(risk))
        primary_risks.append(torch.from_numpy(primary))

    risk_tensor = torch.stack(risk_channels, dim=0).to(device=device, dtype=torch.float32)
    primary_risk_tensor = torch.stack(primary_risks, dim=0).clamp(0.0, 1.0).to(device=device, dtype=torch.float32)
    return risk_tensor, primary_risk_tensor


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    freeze_coarse_active: bool,
) -> float:
    from src.metrics import preserve_low_risk_loss, soft_dice_loss_from_logits, weighted_bce_loss_from_logits

    model.train()
    if freeze_coarse_active:
        for module in model.coarse_path_modules():
            module.eval()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        instances = batch["instance"].to(device)
        confounders = batch["confounder"].to(device)

        optimizer.zero_grad(set_to_none=True)
        coarse_state = model.forward_coarse(image)
        coarse_prob = torch.sigmoid(coarse_state.coarse_logits)
        risk_tensor, primary_risk = build_risk_tensor_batch(
            coarse_state=coarse_state,
            coarse_prob=coarse_prob,
            variant=args.variant,
            boundary_radius=args.boundary_radius,
            uncertainty_quantile=args.uncertainty_quantile,
            device=device,
        )
        refined_logits = model.decode_refine(coarse_state, risk_tensor)
        failure_target = None
        failure_loss = torch.zeros((), device=device)
        focus_gate = primary_risk
        if args.variant == "learned_failure_head" or args.variant == "learned_failure_head_calibrated":
            failure_target = build_failure_target_batch(
                coarse_prob=coarse_prob,
                target=target,
                instances=instances,
                confounders=confounders,
                boundary_radius=args.boundary_radius,
                contact_restrict=args.failure_contact_restrict,
                device=device,
            )
            failure_weight = 1.0 + args.failure_positive_weight * failure_target
            failure_loss = weighted_bce_loss_from_logits(
                coarse_state.failure_logits,
                failure_target,
                failure_weight,
            )
            teacher_boundary = failure_target[:, 0:1]
            teacher_structure = failure_target[:, 1:2] * float(max(args.failure_structure_gate_scale, 0.0))
            teacher_focus = torch.maximum(teacher_boundary, teacher_structure).clamp(0.0, 1.0)
            mix = float(np.clip(args.failure_teacher_mix, 0.0, 1.0))
            if mix > 0.0:
                focus_gate = ((1.0 - mix) * primary_risk + mix * teacher_focus).clamp(0.0, 1.0)

        coarse_loss = F.binary_cross_entropy_with_logits(coarse_state.coarse_logits, target.float())
        coarse_loss = coarse_loss + soft_dice_loss_from_logits(coarse_state.coarse_logits, target.float())
        refine_loss = F.binary_cross_entropy_with_logits(refined_logits, target.float())
        refine_loss = refine_loss + soft_dice_loss_from_logits(refined_logits, target.float())
        boundary_loss = weighted_bce_loss_from_logits(
            refined_logits,
            target.float(),
            target_boundary_weight(target, args.boundary_radius, args.boundary_weight),
        )
        preserve_loss = preserve_low_risk_loss(refined_logits, coarse_state.coarse_logits, focus_gate)
        risk_focus_weight = 1.0 + args.risk_focus_weight * focus_gate
        risk_focus_loss = weighted_bce_loss_from_logits(
            refined_logits,
            target.float(),
            risk_focus_weight,
        )

        loss = (
            args.lambda_coarse * coarse_loss
            + refine_loss
            + args.lambda_boundary * boundary_loss
            + args.lambda_preserve * preserve_loss
            + args.lambda_risk_focus * risk_focus_loss
            + args.lambda_failure * failure_loss
        )
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * image.shape[0]

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    reference_model: torch.nn.Module,
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    from confounder_mining.instance_metrics import instance_metrics_from_binary
    from src.metrics import boundary_band_numpy, dice_numpy, masked_error_rate_numpy, masked_dice_numpy

    reference_model.eval()
    model.eval()
    rows: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        instances = batch["instance"].cpu().numpy()[:, 0]
        confounders = batch["confounder"].cpu().numpy()[:, 0]
        patch_ids = list(batch["id"])

        baseline_logits = reference_model(image)
        baseline_prob = torch.sigmoid(baseline_logits)
        baseline_pred = (baseline_prob.cpu().numpy()[:, 0] >= 0.5)

        coarse_state = model.forward_coarse(image)
        coarse_prob = torch.sigmoid(coarse_state.coarse_logits)
        risk_tensor, primary_risk = build_risk_tensor_batch(
            coarse_state=coarse_state,
            coarse_prob=coarse_prob,
            variant=args.variant,
            boundary_radius=args.boundary_radius,
            uncertainty_quantile=args.uncertainty_quantile,
            device=device,
        )
        refined_logits = model.decode_refine(coarse_state, risk_tensor)
        refined_pred = (torch.sigmoid(refined_logits).cpu().numpy()[:, 0] >= 0.5)
        coarse_pred = (coarse_prob.cpu().numpy()[:, 0] >= 0.5)

        target_np = target.cpu().numpy()[:, 0] >= 0.5
        primary_risk_np = primary_risk.cpu().numpy()[:, 0]

        for i, patch_id in enumerate(patch_ids):
            gt = target_np[i]
            base_mask = baseline_pred[i]
            coarse_mask = coarse_pred[i]
            refined_mask = refined_pred[i]
            gt_boundary = boundary_band_numpy(gt, args.boundary_radius)
            base_boundary = boundary_band_numpy(base_mask, args.boundary_radius)
            coarse_boundary = boundary_band_numpy(coarse_mask, args.boundary_radius)
            refined_boundary = boundary_band_numpy(refined_mask, args.boundary_radius)
            high_risk = primary_risk_np[i] >= 0.5
            low_risk = ~high_risk

            base_aji, base_pq = instance_metrics_from_binary(base_mask, instances[i])
            coarse_aji, coarse_pq = instance_metrics_from_binary(coarse_mask, instances[i])
            refined_aji, refined_pq = instance_metrics_from_binary(refined_mask, instances[i])
            confounder_region = confounders[i] > 0.5
            base_confounder_fpr = float(base_mask[confounder_region].mean()) if confounder_region.any() else 0.0
            coarse_confounder_fpr = float(coarse_mask[confounder_region].mean()) if confounder_region.any() else 0.0
            refined_confounder_fpr = float(refined_mask[confounder_region].mean()) if confounder_region.any() else 0.0

            row = {
                "patch_id": patch_id,
                "baseline_dice": dice_numpy(base_mask, gt),
                "coarse_dice": dice_numpy(coarse_mask, gt),
                "refined_dice": dice_numpy(refined_mask, gt),
                "baseline_boundary_dice": dice_numpy(base_boundary, gt_boundary),
                "coarse_boundary_dice": dice_numpy(coarse_boundary, gt_boundary),
                "refined_boundary_dice": dice_numpy(refined_boundary, gt_boundary),
                "baseline_aji": base_aji,
                "coarse_aji": coarse_aji,
                "refined_aji": refined_aji,
                "baseline_pq": base_pq,
                "coarse_pq": coarse_pq,
                "refined_pq": refined_pq,
                "baseline_high_risk_error": masked_error_rate_numpy(base_mask, gt, high_risk),
                "coarse_high_risk_error": masked_error_rate_numpy(coarse_mask, gt, high_risk),
                "refined_high_risk_error": masked_error_rate_numpy(refined_mask, gt, high_risk),
                "baseline_low_risk_error": masked_error_rate_numpy(base_mask, gt, low_risk),
                "coarse_low_risk_error": masked_error_rate_numpy(coarse_mask, gt, low_risk),
                "refined_low_risk_error": masked_error_rate_numpy(refined_mask, gt, low_risk),
                "baseline_high_risk_dice": masked_dice_numpy(base_mask, gt, high_risk),
                "coarse_high_risk_dice": masked_dice_numpy(coarse_mask, gt, high_risk),
                "refined_high_risk_dice": masked_dice_numpy(refined_mask, gt, high_risk),
                "baseline_confounder_fpr": base_confounder_fpr,
                "coarse_confounder_fpr": coarse_confounder_fpr,
                "refined_confounder_fpr": refined_confounder_fpr,
                "mean_primary_risk": float(primary_risk_np[i].mean()),
                "high_risk_frac": float(high_risk.mean()),
            }
            row["delta_dice"] = row["refined_dice"] - row["baseline_dice"]
            row["delta_boundary_dice"] = row["refined_boundary_dice"] - row["baseline_boundary_dice"]
            row["delta_aji"] = row["refined_aji"] - row["baseline_aji"]
            row["delta_pq"] = row["refined_pq"] - row["baseline_pq"]
            row["delta_high_risk_error"] = row["refined_high_risk_error"] - row["baseline_high_risk_error"]
            row["delta_low_risk_error"] = row["refined_low_risk_error"] - row["baseline_low_risk_error"]
            row["delta_confounder_fpr"] = row["refined_confounder_fpr"] - row["baseline_confounder_fpr"]
            rows.append(row)

    if not rows:
        raise RuntimeError("No evaluation rows were produced.")

    summary_keys = [
        "baseline_dice",
        "coarse_dice",
        "refined_dice",
        "delta_dice",
        "baseline_boundary_dice",
        "coarse_boundary_dice",
        "refined_boundary_dice",
        "delta_boundary_dice",
        "baseline_aji",
        "coarse_aji",
        "refined_aji",
        "delta_aji",
        "baseline_pq",
        "coarse_pq",
        "refined_pq",
        "delta_pq",
        "baseline_high_risk_error",
        "coarse_high_risk_error",
        "refined_high_risk_error",
        "delta_high_risk_error",
        "baseline_low_risk_error",
        "coarse_low_risk_error",
        "refined_low_risk_error",
        "delta_low_risk_error",
        "baseline_confounder_fpr",
        "coarse_confounder_fpr",
        "refined_confounder_fpr",
        "delta_confounder_fpr",
        "mean_primary_risk",
        "high_risk_frac",
    ]
    summary = {key: float(np.mean([float(row[key]) for row in rows])) for key in summary_keys}
    return summary, rows


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
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


def maybe_freeze_coarse_path(model: torch.nn.Module, epoch: int, freeze_coarse_epochs: int) -> bool:
    should_freeze = epoch <= freeze_coarse_epochs
    model.set_coarse_path_requires_grad(not should_freeze)
    return should_freeze


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    workspace_root = args.workspace_root.resolve()
    add_confounder_project_to_path(workspace_root)

    from confounder_mining.dataset import NucleiPointDataset
    from confounder_mining.unet import SmallUNet
    from src.models import BoundarySFRMUNet

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

    checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline_state = checkpoint["model"]

    reference_model = SmallUNet(base_channels=args.base_channels).to(device)
    reference_model.load_state_dict(baseline_state)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    model = BoundarySFRMUNet(
        in_channels=3,
        base_channels=args.base_channels,
        risk_channels=risk_channels_for_variant(args.variant),
    ).to(device)
    model.initialize_from_small_unet(baseline_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / "history.csv"
    best_boundary = -1.0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "baseline_dice",
            "coarse_dice",
            "refined_dice",
            "delta_dice",
            "baseline_boundary_dice",
            "coarse_boundary_dice",
            "refined_boundary_dice",
            "delta_boundary_dice",
            "baseline_aji",
            "coarse_aji",
            "refined_aji",
            "delta_aji",
            "baseline_pq",
            "coarse_pq",
            "refined_pq",
            "delta_pq",
            "delta_high_risk_error",
            "delta_low_risk_error",
            "delta_confounder_fpr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            freeze_coarse_active = maybe_freeze_coarse_path(model, epoch, args.freeze_coarse_epochs)
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                args=args,
                device=device,
                freeze_coarse_active=freeze_coarse_active,
            )
            summary, rows = evaluate(
                reference_model=reference_model,
                model=model,
                loader=val_loader,
                args=args,
                device=device,
            )
            summary["epoch"] = epoch
            summary["train_loss"] = train_loss
            writer.writerow({key: summary.get(key) for key in fieldnames})
            handle.flush()

            with (args.output_dir / f"val_epoch_{epoch:02d}_summary.json").open("w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            with (args.output_dir / f"val_epoch_{epoch:02d}_per_sample.csv").open("w", newline="", encoding="utf-8") as fp:
                per_sample_writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
                per_sample_writer.writeheader()
                per_sample_writer.writerows(rows)

            save_checkpoint(args.output_dir / "last_model.pt", model, optimizer, epoch, summary, args)
            if summary["refined_boundary_dice"] > best_boundary:
                best_boundary = summary["refined_boundary_dice"]
                save_checkpoint(args.output_dir / "best_boundary_model.pt", model, optimizer, epoch, summary, args)


if __name__ == "__main__":
    main()
