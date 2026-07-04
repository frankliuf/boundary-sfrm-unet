from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .features import FeatureMap


@dataclass(frozen=True)
class MiningConfig:
    r_pos: float = 8.0
    r_neg_inner: float = 10.0
    r_neg_outer: float = 34.0
    sim_threshold: float = 0.55
    top_quantile: float = 0.80
    max_points: int = 32
    negative_clusters: int = 3
    random_state: int = 7


@dataclass(frozen=True)
class MiningResult:
    positive_seed_map: np.ndarray
    negative_candidate_map: np.ndarray
    similarity_map: np.ndarray
    prototype_labels: np.ndarray
    positive_prototypes: torch.Tensor
    negative_prototypes: torch.Tensor


def _feature_grid_coordinates(feature_map: FeatureMap) -> tuple[np.ndarray, np.ndarray]:
    _, grid_h, grid_w = feature_map.features.shape
    ys = (np.arange(grid_h) + 0.5) * feature_map.stride_y
    xs = (np.arange(grid_w) + 0.5) * feature_map.stride_x
    return np.meshgrid(ys, xs, indexing="ij")


def _sample_positive_prototype(
    features: torch.Tensor,
    positive_mask: np.ndarray,
) -> torch.Tensor:
    coords = np.argwhere(positive_mask)
    if len(coords) == 0:
        raise ValueError("Positive seed mask is empty.")
    feature_vectors = features[:, coords[:, 0], coords[:, 1]].T
    prototype = feature_vectors.mean(dim=0)
    return F.normalize(prototype, dim=0)


def mine_annular_hard_negatives(
    feature_map: FeatureMap,
    points_yx: np.ndarray,
    config: MiningConfig,
) -> MiningResult:
    """Mine annular high-similarity hard negatives around point labels."""
    features = F.normalize(feature_map.features, dim=0)
    _, grid_h, grid_w = features.shape
    yy, xx = _feature_grid_coordinates(feature_map)

    positive_seed_map = np.zeros((grid_h, grid_w), dtype=bool)
    negative_candidate_map = np.zeros((grid_h, grid_w), dtype=bool)
    similarity_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    positive_prototypes: list[torch.Tensor] = []

    if points_yx.size == 0:
        raise ValueError("No point annotations were provided.")

    selected_points = points_yx
    if len(points_yx) > config.max_points:
        rng = np.random.default_rng(config.random_state)
        indices = rng.choice(len(points_yx), size=config.max_points, replace=False)
        selected_points = points_yx[np.sort(indices)]

    for point_y, point_x in selected_points:
        distance = np.sqrt((yy - point_y) ** 2 + (xx - point_x) ** 2)
        positive_mask = distance <= config.r_pos
        if not positive_mask.any():
            continue

        positive_prototype = _sample_positive_prototype(features, positive_mask)
        positive_prototypes.append(positive_prototype)
        sim = torch.einsum("c,cij->ij", positive_prototype, features).numpy()
        annular_mask = (distance > config.r_neg_inner) & (distance < config.r_neg_outer)
        annular_values = sim[annular_mask]
        if annular_values.size == 0:
            continue
        adaptive_threshold = np.quantile(annular_values, config.top_quantile)
        threshold = max(config.sim_threshold, float(adaptive_threshold))
        hard_negative_mask = annular_mask & (sim > threshold)

        positive_seed_map |= positive_mask
        negative_candidate_map |= hard_negative_mask
        similarity_map = np.maximum(similarity_map, sim.astype(np.float32))

    if not positive_prototypes:
        raise ValueError("No positive prototypes could be initialized.")

    positive_stack = torch.stack(positive_prototypes, dim=0)
    negative_coords = np.argwhere(negative_candidate_map)
    if len(negative_coords) == 0:
        negative_prototypes = torch.empty((0, features.shape[0]), dtype=features.dtype)
        prototype_labels = np.full((grid_h, grid_w), -1, dtype=np.int32)
        return MiningResult(
            positive_seed_map=positive_seed_map,
            negative_candidate_map=negative_candidate_map,
            similarity_map=similarity_map,
            prototype_labels=prototype_labels,
            positive_prototypes=positive_stack,
            negative_prototypes=negative_prototypes,
        )

    negative_features = features[:, negative_coords[:, 0], negative_coords[:, 1]].T
    cluster_count = min(config.negative_clusters, len(negative_coords))
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=config.random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(negative_features.numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).to(dtype=features.dtype)
    centers = F.normalize(centers, dim=1)

    prototype_labels = np.full((grid_h, grid_w), -1, dtype=np.int32)
    prototype_labels[negative_coords[:, 0], negative_coords[:, 1]] = labels

    return MiningResult(
        positive_seed_map=positive_seed_map,
        negative_candidate_map=negative_candidate_map,
        similarity_map=similarity_map,
        prototype_labels=prototype_labels,
        positive_prototypes=positive_stack,
        negative_prototypes=centers,
    )
