from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image
from scipy import ndimage


def load_mask(mask_path: str) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return mask > 0


def connected_components(binary_mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected component coordinates as arrays of [y, x]."""
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    components: list[np.ndarray] = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary_mask[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            components.append(np.asarray(coords, dtype=np.int32))
    return components


def simulate_point_annotations(
    mask: np.ndarray,
    min_area: int = 8,
    point_strategy: str = "centroid",
) -> np.ndarray:
    """Simulate point annotations from a semantic or instance mask.

    For instance masks, each positive label is treated as one object. For binary
    masks, connected components are used.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    if point_strategy not in {"centroid", "distance"}:
        raise ValueError("point_strategy must be 'centroid' or 'distance'.")

    points: list[tuple[int, int]] = []
    positive_labels = [label for label in np.unique(mask) if label != 0]
    if len(positive_labels) > 1:
        for label in positive_labels:
            coords = np.argwhere(mask == label)
            if len(coords) < min_area:
                continue
            points.append(_representative_point(mask == label, coords, point_strategy))
    else:
        for coords in connected_components(binarize_mask(mask)):
            if len(coords) < min_area:
                continue
            component_mask = np.zeros(mask.shape, dtype=bool)
            component_mask[coords[:, 0], coords[:, 1]] = True
            points.append(_representative_point(component_mask, coords, point_strategy))

    return np.asarray(points, dtype=np.int32)


def _representative_point(
    object_mask: np.ndarray,
    coords: np.ndarray,
    point_strategy: str,
) -> tuple[int, int]:
    if point_strategy == "centroid":
        yx = np.round(coords.mean(axis=0)).astype(int)
        return int(yx[0]), int(yx[1])

    distance = ndimage.distance_transform_edt(object_mask)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return int(y), int(x)
