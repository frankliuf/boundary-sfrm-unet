"""Reliability metrics for failure-region analysis."""

from __future__ import annotations

import numpy as np


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Compute binary Dice score."""

    pred = prediction.astype(bool)
    gt = target.astype(bool)
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    return float((2.0 * intersection + eps) / (denom + eps))


def region_fraction(region: np.ndarray) -> float:
    """Fraction of pixels covered by a binary region."""

    mask = region.astype(bool)
    return float(mask.mean())


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean value inside a binary mask. Returns NaN when the mask is empty."""

    region = mask.astype(bool)
    if not np.any(region):
        return float("nan")
    return float(np.asarray(values)[region].mean())

