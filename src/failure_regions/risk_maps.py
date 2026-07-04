from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from .decompose import binary_entropy
from ..uncertainty.maps import normalize_unit_interval


@dataclass(frozen=True)
class SFRMRiskMaps:
    prob: np.ndarray
    pred: np.ndarray
    entropy: np.ndarray
    boundary_band: np.ndarray
    boundary_risk: np.ndarray
    soft_boundary_weight: np.ndarray
    local_uncertainty: np.ndarray
    probability_gradient: np.ndarray
    soft_boundary_risk: np.ndarray
    split_risk: np.ndarray
    feature_split_risk: np.ndarray
    uncertainty_cluster: np.ndarray
    topology_risk: np.ndarray
    morphology_risk: np.ndarray
    composite_risk: np.ndarray


def square_boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if radius <= 0:
        raise ValueError("radius must be positive")
    kernel_size = radius * 2 + 1
    footprint = np.ones((kernel_size, kernel_size), dtype=bool)
    dilated = morphology.dilation(mask_bool, footprint)
    eroded = morphology.erosion(mask_bool, footprint)
    return np.logical_xor(dilated, eroded)


def build_sfrm_risk_maps(
    probability: np.ndarray,
    feature_map: np.ndarray | None = None,
    threshold: float = 0.5,
    boundary_radius: int = 2,
    uncertainty_quantile: float = 0.85,
    min_component_area: int = 12,
) -> SFRMRiskMaps:
    prob = np.asarray(probability, dtype=np.float32)
    pred = prob >= threshold
    entropy = binary_entropy(prob)
    boundary_band = square_boundary_band(pred, boundary_radius)

    boundary_risk = normalize_unit_interval(entropy * boundary_band.astype(np.float32))
    soft_boundary_weight = _soft_boundary_weight(boundary_band, boundary_radius=boundary_radius)
    local_uncertainty = _local_uncertainty_map(entropy, boundary_radius=boundary_radius)
    probability_gradient = _probability_gradient_map(prob)
    uncertainty_threshold = float(np.quantile(entropy, uncertainty_quantile))
    uncertainty_cluster = (entropy >= uncertainty_threshold).astype(np.float32)

    split_risk = _split_risk_map(pred, prob, min_component_area=min_component_area)
    feature_split_risk = _feature_split_risk_map(
        pred,
        prob,
        feature_map=feature_map,
        min_component_area=min_component_area,
    )
    topology_risk = _topology_risk_map(pred, min_component_area=min_component_area)
    morphology_risk = _morphology_risk_map(pred)
    soft_boundary_risk = _soft_boundary_risk_map(
        soft_boundary_weight=soft_boundary_weight,
        entropy=entropy,
        local_uncertainty=local_uncertainty,
        probability_gradient=probability_gradient,
        morphology_risk=morphology_risk,
    )
    composite_risk = np.maximum.reduce(
        [
            boundary_risk,
            soft_boundary_risk,
            split_risk,
            feature_split_risk,
            uncertainty_cluster,
            topology_risk,
            morphology_risk,
        ]
    ).astype(np.float32)

    return SFRMRiskMaps(
        prob=prob,
        pred=pred.astype(np.uint8),
        entropy=entropy.astype(np.float32),
        boundary_band=boundary_band.astype(np.float32),
        boundary_risk=boundary_risk.astype(np.float32),
        soft_boundary_weight=soft_boundary_weight.astype(np.float32),
        local_uncertainty=local_uncertainty.astype(np.float32),
        probability_gradient=probability_gradient.astype(np.float32),
        soft_boundary_risk=soft_boundary_risk.astype(np.float32),
        split_risk=split_risk.astype(np.float32),
        feature_split_risk=feature_split_risk.astype(np.float32),
        uncertainty_cluster=uncertainty_cluster.astype(np.float32),
        topology_risk=topology_risk.astype(np.float32),
        morphology_risk=morphology_risk.astype(np.float32),
        composite_risk=composite_risk.astype(np.float32),
    )


def _soft_boundary_weight(boundary_band: np.ndarray, boundary_radius: int) -> np.ndarray:
    sigma = max(float(boundary_radius), 1.0)
    distance = ndi.distance_transform_edt(~boundary_band.astype(bool)).astype(np.float32)
    weight = np.exp(-(distance ** 2) / (2.0 * sigma * sigma))
    return normalize_unit_interval(weight)


def _local_uncertainty_map(entropy: np.ndarray, boundary_radius: int) -> np.ndarray:
    window = max(3, boundary_radius * 2 + 1)
    local_mean = ndi.uniform_filter(entropy.astype(np.float32), size=window, mode="nearest")
    return normalize_unit_interval(local_mean)


def _probability_gradient_map(prob: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(prob.astype(np.float32))
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return normalize_unit_interval(grad_mag)


def _soft_boundary_risk_map(
    soft_boundary_weight: np.ndarray,
    entropy: np.ndarray,
    local_uncertainty: np.ndarray,
    probability_gradient: np.ndarray,
    morphology_risk: np.ndarray,
) -> np.ndarray:
    local_boundary_uncertainty = soft_boundary_weight * (
        0.5 * normalize_unit_interval(entropy) + 0.5 * local_uncertainty
    )
    weak_transition = soft_boundary_weight * (1.0 - probability_gradient)
    morphology_boundary = soft_boundary_weight * morphology_risk
    combined = (
        0.15 * soft_boundary_weight
        + 0.4 * local_boundary_uncertainty
        + 0.3 * weak_transition
        + 0.15 * morphology_boundary
    )
    return normalize_unit_interval(combined.astype(np.float32))


def _split_risk_map(
    pred: np.ndarray,
    prob: np.ndarray,
    min_component_area: int,
) -> np.ndarray:
    pred_bool = pred.astype(bool)
    if not pred_bool.any():
        return np.zeros(pred.shape, dtype=np.float32)

    distance = ndi.distance_transform_edt(pred_bool).astype(np.float32)
    distance_smooth = ndi.gaussian_filter(distance, sigma=1.0)
    local_max = morphology.local_maxima(distance_smooth)
    labeled = measure.label(pred_bool, connectivity=2)
    output = np.zeros(pred.shape, dtype=np.float32)

    for region in measure.regionprops(labeled):
        if region.area < min_component_area:
            continue

        component_mask = labeled == region.label
        component_max = measure.label(np.logical_and(local_max, component_mask), connectivity=2)
        if component_max.max() < 2:
            continue

        watershed_labels = segmentation.watershed(
            -distance_smooth,
            markers=component_max,
            mask=component_mask,
        )
        label_max = ndi.maximum_filter(watershed_labels, size=3)
        label_min = ndi.minimum_filter(watershed_labels, size=3)
        split_line = component_mask & (label_max != label_min)
        if not split_line.any():
            continue

        component_distance = distance_smooth * component_mask.astype(np.float32)
        neckness = component_mask.astype(np.float32) * (
            1.0 - normalize_unit_interval(component_distance)
        )
        bridge_uncertainty = component_mask.astype(np.float32) * (
            1.0 - normalize_unit_interval(np.abs(prob - 0.5))
        )
        split_band = ndi.binary_dilation(split_line, structure=np.ones((3, 3), dtype=bool))
        split_score = split_band.astype(np.float32) * (0.65 * neckness + 0.35 * bridge_uncertainty)
        output = np.maximum(output, normalize_unit_interval(split_score))

    return output.astype(np.float32)


def _feature_split_risk_map(
    pred: np.ndarray,
    prob: np.ndarray,
    feature_map: np.ndarray | None,
    min_component_area: int,
    prototype_radius: int = 3,
    max_prototypes: int = 4,
) -> np.ndarray:
    if feature_map is None:
        return np.zeros(pred.shape, dtype=np.float32)

    pred_bool = pred.astype(bool)
    if not pred_bool.any():
        return np.zeros(pred.shape, dtype=np.float32)

    feature = np.asarray(feature_map, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("feature_map must have shape [C, H, W]")

    feat_norm = np.linalg.norm(feature, axis=0, keepdims=True)
    feature = feature / np.clip(feat_norm, 1e-6, None)

    distance = ndi.distance_transform_edt(pred_bool).astype(np.float32)
    distance_smooth = ndi.gaussian_filter(distance, sigma=1.0)
    local_max = morphology.local_maxima(distance_smooth)
    labeled = measure.label(pred_bool, connectivity=2)
    output = np.zeros(pred.shape, dtype=np.float32)

    yy, xx = np.ogrid[: pred.shape[0], : pred.shape[1]]

    for region in measure.regionprops(labeled):
        if region.area < min_component_area:
            continue

        component_mask = labeled == region.label
        component_max = measure.label(np.logical_and(local_max, component_mask), connectivity=2)
        if component_max.max() < 2:
            continue

        peak_regions = []
        for peak in measure.regionprops(component_max, intensity_image=distance_smooth):
            peak_regions.append(
                (
                    float(peak.intensity_max),
                    int(round(peak.centroid[0])),
                    int(round(peak.centroid[1])),
                )
            )
        peak_regions.sort(reverse=True)
        peak_regions = peak_regions[:max_prototypes]
        if len(peak_regions) < 2:
            continue

        prototypes: list[np.ndarray] = []
        marker_image = np.zeros(pred.shape, dtype=np.int32)
        for marker_idx, (_, cy, cx) in enumerate(peak_regions, start=1):
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (prototype_radius ** 2)
            support = np.logical_and(component_mask, disk)
            if support.sum() < 4:
                support = np.logical_and(component_mask, distance_smooth >= max(distance_smooth[cy, cx] * 0.7, 1.0))
            if support.sum() < 4:
                continue
            prototype = feature[:, support].mean(axis=1)
            prototype = prototype / np.clip(np.linalg.norm(prototype), 1e-6, None)
            prototypes.append(prototype.astype(np.float32))
            marker_image[cy, cx] = marker_idx

        if len(prototypes) < 2:
            continue

        watershed_labels = segmentation.watershed(
            -distance_smooth,
            markers=marker_image,
            mask=component_mask,
        )
        label_max = ndi.maximum_filter(watershed_labels, size=3)
        label_min = ndi.minimum_filter(watershed_labels, size=3)
        split_line = component_mask & (label_max != label_min)
        if not split_line.any():
            continue

        pixel_features = feature[:, component_mask].T
        prototype_matrix = np.stack(prototypes, axis=0)
        sim = pixel_features @ prototype_matrix.T
        sim.sort(axis=1)
        top1 = sim[:, -1]
        top2 = sim[:, -2]
        ambiguity = 1.0 - np.clip(top1 - top2, 0.0, 1.0)

        ambiguity_map = np.zeros(pred.shape, dtype=np.float32)
        ambiguity_map[component_mask] = ambiguity.astype(np.float32)
        ambiguity_map = normalize_unit_interval(ambiguity_map)

        component_distance = distance_smooth * component_mask.astype(np.float32)
        neckness = component_mask.astype(np.float32) * (
            1.0 - normalize_unit_interval(component_distance)
        )
        bridge_uncertainty = component_mask.astype(np.float32) * (
            1.0 - normalize_unit_interval(np.abs(prob - 0.5))
        )
        split_band = ndi.binary_dilation(split_line, structure=np.ones((3, 3), dtype=bool)).astype(np.float32)
        split_score = split_band * (
            0.5 * ambiguity_map
            + 0.3 * neckness
            + 0.2 * bridge_uncertainty
        )
        output = np.maximum(output, normalize_unit_interval(split_score))

    return output.astype(np.float32)


def _topology_risk_map(pred: np.ndarray, min_component_area: int) -> np.ndarray:
    pred_bool = pred.astype(bool)
    labeled = measure.label(pred_bool, connectivity=2)
    output = np.zeros(pred.shape, dtype=np.float32)
    if labeled.max() == 0:
        return output

    for region in measure.regionprops(labeled):
        if region.area < min_component_area:
            output[labeled == region.label] = 1.0

    closed = morphology.closing(pred_bool, morphology.disk(1))
    merge_gaps = np.logical_and(closed, ~pred_bool)
    output[merge_gaps] = 1.0
    return output


def _morphology_risk_map(pred: np.ndarray) -> np.ndarray:
    pred_bool = pred.astype(bool)
    opened = morphology.opening(pred_bool, morphology.disk(1))
    closed = morphology.closing(pred_bool, morphology.disk(1))
    residual = np.logical_or(
        np.logical_xor(pred_bool, opened),
        np.logical_xor(pred_bool, closed),
    )
    return residual.astype(np.float32)
