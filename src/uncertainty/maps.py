"""Uncertainty map helpers."""

from __future__ import annotations

import numpy as np


def binary_margin_uncertainty(probability: np.ndarray) -> np.ndarray:
    """Return uncertainty based on distance from the 0.5 decision boundary."""

    p = np.asarray(probability, dtype=np.float32)
    return 1.0 - 2.0 * np.abs(p - 0.5)


def normalize_unit_interval(values: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Normalize an array to [0, 1]."""

    x = np.asarray(values, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / (hi - lo + eps)

