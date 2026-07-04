from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def soft_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def weighted_bce_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    weighted = losses * weight_map
    return weighted.mean()


def preserve_low_risk_loss(
    corrected_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    corrected_prob = torch.sigmoid(corrected_logits)
    baseline_prob = torch.sigmoid(baseline_logits)
    low_risk = 1.0 - gate
    return (low_risk * (corrected_prob - baseline_prob).abs()).mean()


def boundary_band_numpy(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask_tensor, kernel_size=kernel_size, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return boundary[0, 0].numpy() > 0.5


def dice_numpy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    inter = float(np.logical_and(pred_bool, gt_bool).sum())
    denom = float(pred_bool.sum() + gt_bool.sum())
    return (2.0 * inter + eps) / (denom + eps)


def masked_dice_numpy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-7) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return 1.0
    pred_valid = np.logical_and(pred.astype(bool), valid)
    gt_valid = np.logical_and(gt.astype(bool), valid)
    inter = float(np.logical_and(pred_valid, gt_valid).sum())
    denom = float(pred_valid.sum() + gt_valid.sum())
    return (2.0 * inter + eps) / (denom + eps)


def masked_error_rate_numpy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return 0.0
    error = np.logical_xor(pred.astype(bool), gt.astype(bool))
    return float(error[valid].mean())
