from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection
    dice = ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return float(dice.item()), float(iou.item())


def boundary_dice_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    radius: int = 2,
    eps: float = 1e-6,
) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    pred_boundary = _mask_boundary(pred, radius=radius)
    target_boundary = _mask_boundary(target, radius=radius)
    intersection = (pred_boundary * target_boundary).sum(dim=(1, 2, 3))
    pred_sum = pred_boundary.sum(dim=(1, 2, 3))
    target_sum = target_boundary.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection
    dice = ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return float(dice.item()), float(iou.item())


def confounder_false_positive_rate(
    logits: torch.Tensor,
    confounder_map: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    valid = confounder_map > 0.5
    if not valid.any():
        return 0.0
    pred = (torch.sigmoid(logits) >= threshold).float()
    rate = (pred[valid].sum() + eps) / (valid.sum() + eps)
    return float(rate.item())


def _mask_boundary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive.")
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=radius)
    return (dilated - eroded).clamp(min=0.0, max=1.0)


def partial_bce_loss(
    logits: torch.Tensor,
    sparse_labels: torch.Tensor,
    balanced: bool = False,
) -> torch.Tensor:
    valid = sparse_labels != 255
    if not valid.any():
        return logits.sum() * 0.0
    targets = sparse_labels.float().unsqueeze(1)
    valid = valid.unsqueeze(1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if balanced:
        foreground = valid & (targets > 0.5)
        background = valid & (targets <= 0.5)
        terms = []
        if foreground.any():
            terms.append(losses[foreground].mean())
        if background.any():
            terms.append(losses[background].mean())
        if terms:
            return torch.stack(terms).mean()
    return losses[valid].mean()


def confounder_probability_loss(
    logits: torch.Tensor,
    confounder_map: torch.Tensor,
    gate_threshold: float = 0.0,
) -> torch.Tensor:
    valid = confounder_map > 0.5
    if not valid.any():
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits).squeeze(1)
    valid = valid.squeeze(1)
    if gate_threshold > 0:
        valid = valid & (probs.detach() >= gate_threshold)
    if not valid.any():
        return logits.sum() * 0.0
    return probs[valid].mean()


def prototype_contrastive_loss(
    features: torch.Tensor,
    sparse_labels: torch.Tensor,
    confounder_map: torch.Tensor,
    temperature: float = 0.2,
    max_samples: int = 2048,
    bidirectional: bool = False,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    features = F.normalize(features, dim=1)
    pos_mask = sparse_labels == 1
    neg_mask = confounder_map.squeeze(1) > 0.5
    losses: list[torch.Tensor] = []

    for batch_index in range(features.shape[0]):
        sample_features = features[batch_index].permute(1, 2, 0)
        positive_features = sample_features[pos_mask[batch_index]]
        negative_features = sample_features[neg_mask[batch_index]]
        if positive_features.numel() == 0 or negative_features.numel() == 0:
            continue

        if positive_features.shape[0] > max_samples:
            positive_features = positive_features[
                torch.randperm(positive_features.shape[0], device=features.device)[:max_samples]
            ]
        if negative_features.shape[0] > max_samples:
            negative_features = negative_features[
                torch.randperm(negative_features.shape[0], device=features.device)[:max_samples]
            ]

        positive_prototype = F.normalize(positive_features.mean(dim=0), dim=0)
        negative_prototype = F.normalize(negative_features.mean(dim=0), dim=0)
        logits = torch.stack(
            [
                positive_features @ positive_prototype,
                positive_features @ negative_prototype,
            ],
            dim=1,
        ) / temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
        losses.append(F.cross_entropy(logits, targets))
        if bidirectional:
            negative_logits = torch.stack(
                [
                    negative_features @ positive_prototype,
                    negative_features @ negative_prototype,
                ],
                dim=1,
            ) / temperature
            negative_targets = torch.ones(
                negative_logits.shape[0],
                dtype=torch.long,
                device=features.device,
            )
            losses.append(F.cross_entropy(negative_logits, negative_targets))

    if not losses:
        return features.sum() * 0.0
    return torch.stack(losses).mean()


def boundary_anchor_contrastive_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    sparse_labels: torch.Tensor,
    confounder_map: torch.Tensor,
    temperature: float = 0.2,
    uncertainty_margin: float = 0.2,
    max_samples: int = 2048,
    anchor_mode: str = "ignore",
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < uncertainty_margin <= 0.5:
        raise ValueError("uncertainty_margin must be in (0, 0.5].")
    allowed_anchor_modes = {"ignore", "ignore_confounder", "confounder"}
    if anchor_mode not in allowed_anchor_modes:
        allowed = ", ".join(sorted(allowed_anchor_modes))
        raise ValueError(f"anchor_mode must be one of: {allowed}.")

    features = F.normalize(features, dim=1)
    probs = torch.sigmoid(logits).squeeze(1)
    pos_mask = sparse_labels == 1
    confounder_mask = confounder_map.squeeze(1) > 0.5
    neg_mask = confounder_mask
    ignore_mask = sparse_labels == 255
    uncertain_mask = (probs.detach() - 0.5).abs() <= uncertainty_margin
    if anchor_mode == "ignore":
        anchor_mask = ignore_mask & uncertain_mask
    elif anchor_mode == "ignore_confounder":
        anchor_mask = ignore_mask & confounder_mask & uncertain_mask
    else:
        anchor_mask = confounder_mask & uncertain_mask
    losses: list[torch.Tensor] = []

    for batch_index in range(features.shape[0]):
        sample_features = features[batch_index].permute(1, 2, 0)
        positive_features = sample_features[pos_mask[batch_index]]
        negative_features = sample_features[neg_mask[batch_index]]
        anchor_features = sample_features[anchor_mask[batch_index]]
        anchor_probs = probs[batch_index][anchor_mask[batch_index]]
        if (
            positive_features.numel() == 0
            or negative_features.numel() == 0
            or anchor_features.numel() == 0
        ):
            continue

        positive_features = _sample_rows(positive_features, max_samples)
        negative_features = _sample_rows(negative_features, max_samples)
        anchor_features, anchor_probs = _sample_feature_probability_pair(
            anchor_features,
            anchor_probs,
            max_samples,
        )

        positive_prototype = F.normalize(positive_features.mean(dim=0), dim=0)
        negative_prototype = F.normalize(negative_features.mean(dim=0), dim=0)
        similarity_logits = torch.stack(
            [
                anchor_features @ positive_prototype,
                anchor_features @ negative_prototype,
            ],
            dim=1,
        ) / temperature
        pseudo_targets = (anchor_probs.detach() < 0.5).long()
        losses.append(F.cross_entropy(similarity_logits, pseudo_targets))

    if not losses:
        return features.sum() * 0.0
    return torch.stack(losses).mean()


def _sample_rows(values: torch.Tensor, max_samples: int) -> torch.Tensor:
    if values.shape[0] <= max_samples:
        return values
    indices = torch.randperm(values.shape[0], device=values.device)[:max_samples]
    return values[indices]


def _sample_feature_probability_pair(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.shape[0] <= max_samples:
        return features, probabilities
    indices = torch.randperm(features.shape[0], device=features.device)[:max_samples]
    return features[indices], probabilities[indices]
