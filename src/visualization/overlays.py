"""Visualization helpers for failure-region figures."""

from __future__ import annotations

import numpy as np


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB image to uint8 for plotting/export."""

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    arr = arr - float(arr.min())
    denom = float(arr.max()) + 1e-7
    return np.clip(255.0 * arr / denom, 0, 255).astype(np.uint8)

