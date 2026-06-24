"""Spatial failure-region decomposition primitives.

The first implementation will stay model-agnostic: it consumes masks,
probability maps, and uncertainty maps produced by any segmentation model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FailureRegions:
    """Container for binary candidate failure regions."""

    boundary_band: np.ndarray
    high_uncertainty: np.ndarray
    false_positive: np.ndarray | None = None
    false_negative: np.ndarray | None = None


def binary_entropy(probability: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute pixel-wise entropy for a binary foreground probability map."""

    p = np.clip(probability.astype(np.float32), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def high_uncertainty_region(
    uncertainty: np.ndarray,
    quantile: float = 0.85,
) -> np.ndarray:
    """Return the top-quantile uncertainty region as a binary mask."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    threshold = float(np.quantile(uncertainty, quantile))
    return uncertainty >= threshold


def prediction_error_regions(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return false-positive and false-negative regions."""

    pred = prediction.astype(bool)
    gt = target.astype(bool)
    false_positive = pred & ~gt
    false_negative = ~pred & gt
    return false_positive, false_negative

