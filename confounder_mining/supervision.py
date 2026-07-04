from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


IGNORE_LABEL = 255


def make_point_supervision_labels(
    image_shape: tuple[int, int],
    points_yx: np.ndarray,
    positive_radius: float = 5.0,
    background_inner_radius: float = 18.0,
) -> np.ndarray:
    """Create sparse point-supervised labels for binary segmentation.

    Labels:
    - 1: confident foreground near point annotations.
    - 0: confident background far from every point.
    - 255: ignore region between positive seeds and far background.
    """
    height, width = image_shape
    labels = np.full((height, width), IGNORE_LABEL, dtype=np.uint8)
    if points_yx.size == 0:
        labels[:, :] = 0
        return labels

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    min_distance = np.full((height, width), np.inf, dtype=np.float32)
    for point_y, point_x in points_yx:
        distance = np.sqrt((yy - point_y) ** 2 + (xx - point_x) ** 2)
        min_distance = np.minimum(min_distance, distance)

    labels[min_distance <= positive_radius] = 1
    labels[min_distance >= background_inner_radius] = 0
    return labels


def make_seeded_region_pseudo_labels(
    image_shape: tuple[int, int],
    points_yx: np.ndarray,
    seeded_region_radius: float = 10.0,
    background_inner_radius: float = 24.0,
) -> np.ndarray:
    """Create a conservative seeded-region pseudo-mask from point labels.

    This baseline expands each point into a local foreground disk while keeping
    an ignore band before the far-background labels. It deliberately avoids
    using the full mask boundary, so it remains a point-supervised baseline.
    """
    if seeded_region_radius >= background_inner_radius:
        raise ValueError("seeded_region_radius must be smaller than background_inner_radius.")
    return make_point_supervision_labels(
        image_shape,
        points_yx,
        positive_radius=seeded_region_radius,
        background_inner_radius=background_inner_radius,
    )


def make_voronoi_kmeans_pseudo_labels(
    image_rgb: np.ndarray,
    points_yx: np.ndarray,
    positive_radius: float = 4.0,
) -> np.ndarray:
    """Create a coarse pseudo-mask from points using Voronoi cells and k-means.

    This lightweight baseline follows the core idea used by point-supervised
    nuclei methods such as SC-Net: propagate point annotations to coarse
    pixel-level labels using topology and color clustering. It deliberately
    avoids full masks and foundation features.
    """
    height, width = image_rgb.shape[:2]
    labels = np.zeros((height, width), dtype=np.uint8)
    if points_yx.size == 0:
        return labels

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    distances = np.stack(
        [(yy - point_y) ** 2 + (xx - point_x) ** 2 for point_y, point_x in points_yx],
        axis=0,
    )
    voronoi = distances.argmin(axis=0)

    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    if len(np.unique(pixels, axis=0)) < 2:
        return make_point_supervision_labels(
            (height, width),
            points_yx,
            positive_radius=positive_radius,
            background_inner_radius=max(positive_radius + 1.0, positive_radius * 2.0),
        )

    clusters = KMeans(n_clusters=2, n_init=3, max_iter=50, random_state=0).fit_predict(pixels)
    cluster_map = clusters.reshape(height, width)
    cluster_means = [
        float(pixels[clusters == cluster_id].mean())
        for cluster_id in range(2)
    ]
    nucleus_cluster = int(np.argmin(cluster_means))
    nucleus_like = cluster_map == nucleus_cluster

    for point_index, (point_y, point_x) in enumerate(points_yx):
        cell = voronoi == point_index
        labels[cell & nucleus_like] = 1
        disk = (yy - point_y) ** 2 + (xx - point_x) ** 2 <= positive_radius**2
        labels[disk] = 1
    return labels


def make_affinity_random_walk_pseudo_labels(
    image_rgb: np.ndarray,
    points_yx: np.ndarray,
    positive_radius: float = 4.0,
    background_inner_radius: float = 24.0,
    color_sigma: float = 18.0,
    alpha: float = 0.92,
    iterations: int = 80,
) -> np.ndarray:
    """Propagate point labels through a local color-affinity random walk.

    This baseline tests whether feature-driven label propagation can replace
    explicit confounder mining. It uses only point-derived seeds and image
    affinity, then trains the same U-Net with the generated full pseudo-mask.
    """
    height, width = image_rgb.shape[:2]
    labels = np.zeros((height, width), dtype=np.uint8)
    if points_yx.size == 0:
        return labels
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    min_distance = np.full((height, width), np.inf, dtype=np.float32)
    for point_y, point_x in points_yx:
        distance = np.sqrt((yy - point_y) ** 2 + (xx - point_x) ** 2)
        min_distance = np.minimum(min_distance, distance)

    fg_seed = min_distance <= positive_radius
    bg_seed = min_distance >= background_inner_radius
    seed_mask = fg_seed | bg_seed
    seed_prob = np.full((height, width), 0.5, dtype=np.float32)
    seed_prob[fg_seed] = 1.0
    seed_prob[bg_seed] = 0.0

    image = image_rgb.astype(np.float32)
    sigma2 = max(color_sigma, 1e-3) ** 2
    weights = _neighbor_affinities(image, sigma2)
    prob = seed_prob.copy()
    for _ in range(iterations):
        neighbor_sum = np.zeros_like(prob, dtype=np.float32)
        weight_sum = np.zeros_like(prob, dtype=np.float32)
        for shifted_prob, weight in _shifted_neighbors(prob, weights):
            neighbor_sum += shifted_prob * weight
            weight_sum += weight
        smoothed = neighbor_sum / np.maximum(weight_sum, 1e-6)
        prob = alpha * smoothed + (1.0 - alpha) * seed_prob
        prob[fg_seed] = 1.0
        prob[bg_seed] = 0.0

    labels[prob >= 0.5] = 1
    labels[fg_seed] = 1
    return labels


def _neighbor_affinities(image_rgb: np.ndarray, sigma2: float) -> dict[str, np.ndarray]:
    height, width = image_rgb.shape[:2]
    weights = {
        "up": np.zeros((height, width), dtype=np.float32),
        "down": np.zeros((height, width), dtype=np.float32),
        "left": np.zeros((height, width), dtype=np.float32),
        "right": np.zeros((height, width), dtype=np.float32),
    }
    diff = image_rgb[1:, :, :] - image_rgb[:-1, :, :]
    vertical = np.exp(-(diff * diff).sum(axis=2) / sigma2).astype(np.float32)
    weights["up"][1:, :] = vertical
    weights["down"][:-1, :] = vertical
    diff = image_rgb[:, 1:, :] - image_rgb[:, :-1, :]
    horizontal = np.exp(-(diff * diff).sum(axis=2) / sigma2).astype(np.float32)
    weights["left"][:, 1:] = horizontal
    weights["right"][:, :-1] = horizontal
    return weights


def _shifted_neighbors(prob: np.ndarray, weights: dict[str, np.ndarray]):
    up = np.zeros_like(prob)
    up[1:, :] = prob[:-1, :]
    yield up, weights["up"]
    down = np.zeros_like(prob)
    down[:-1, :] = prob[1:, :]
    yield down, weights["down"]
    left = np.zeros_like(prob)
    left[:, 1:] = prob[:, :-1]
    yield left, weights["left"]
    right = np.zeros_like(prob)
    right[:, :-1] = prob[:, 1:]
    yield right, weights["right"]


def label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "foreground": int((labels == 1).sum()),
        "background": int((labels == 0).sum()),
        "ignore": int((labels == IGNORE_LABEL).sum()),
    }
