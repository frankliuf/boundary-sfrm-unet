from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as connected_label


def instance_metrics_from_binary(
    pred_binary: np.ndarray,
    target_instance: np.ndarray,
    pq_iou_threshold: float = 0.5,
) -> tuple[float, float]:
    pred_instance = connected_label(pred_binary > 0, connectivity=1).astype(np.int32)
    target_instance = relabel_consecutive(target_instance.astype(np.int32))
    return aggregated_jaccard_index(pred_instance, target_instance), panoptic_quality(
        pred_instance,
        target_instance,
        iou_threshold=pq_iou_threshold,
    )


def aggregated_jaccard_index(pred_instance: np.ndarray, target_instance: np.ndarray) -> float:
    pairwise_iou, pairwise_intersection, pairwise_union, pred_labels, target_labels = pairwise_stats(
        pred_instance,
        target_instance,
    )
    if len(pred_labels) == 0 and len(target_labels) == 0:
        return 1.0
    if len(pred_labels) == 0 or len(target_labels) == 0:
        return 0.0

    target_indices, pred_indices = linear_sum_assignment(-pairwise_iou)
    matched = pairwise_iou[target_indices, pred_indices] > 0
    matched_targets = target_indices[matched]
    matched_preds = pred_indices[matched]

    intersection_sum = float(pairwise_intersection[matched_targets, matched_preds].sum())
    union_sum = float(pairwise_union[matched_targets, matched_preds].sum())
    matched_target_set = set(int(i) for i in matched_targets)
    matched_pred_set = set(int(i) for i in matched_preds)

    for target_index, target_label in enumerate(target_labels):
        if target_index not in matched_target_set:
            union_sum += float((target_instance == target_label).sum())
    for pred_index, pred_label in enumerate(pred_labels):
        if pred_index not in matched_pred_set:
            union_sum += float((pred_instance == pred_label).sum())

    if union_sum == 0:
        return 0.0
    return intersection_sum / union_sum


def panoptic_quality(
    pred_instance: np.ndarray,
    target_instance: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    pairwise_iou, _, _, pred_labels, target_labels = pairwise_stats(pred_instance, target_instance)
    if len(pred_labels) == 0 and len(target_labels) == 0:
        return 1.0
    if len(pred_labels) == 0 or len(target_labels) == 0:
        return 0.0

    target_indices, pred_indices = linear_sum_assignment(-pairwise_iou)
    matched_ious = pairwise_iou[target_indices, pred_indices]
    valid = matched_ious >= iou_threshold
    true_positive = int(valid.sum())
    false_positive = len(pred_labels) - true_positive
    false_negative = len(target_labels) - true_positive
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    if denominator == 0:
        return 0.0
    return float(matched_ious[valid].sum()) / denominator


def pairwise_stats(
    pred_instance: np.ndarray,
    target_instance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    pred_labels = [int(label) for label in np.unique(pred_instance) if label != 0]
    target_labels = [int(label) for label in np.unique(target_instance) if label != 0]
    iou = np.zeros((len(target_labels), len(pred_labels)), dtype=np.float64)
    intersection = np.zeros_like(iou)
    union = np.zeros_like(iou)

    pred_masks = {label: pred_instance == label for label in pred_labels}
    target_masks = {label: target_instance == label for label in target_labels}
    for target_index, target_label in enumerate(target_labels):
        target_mask = target_masks[target_label]
        for pred_index, pred_label in enumerate(pred_labels):
            pred_mask = pred_masks[pred_label]
            inter = float(np.logical_and(target_mask, pred_mask).sum())
            uni = float(np.logical_or(target_mask, pred_mask).sum())
            intersection[target_index, pred_index] = inter
            union[target_index, pred_index] = uni
            if uni > 0:
                iou[target_index, pred_index] = inter / uni
    return iou, intersection, union, pred_labels, target_labels


def relabel_consecutive(mask: np.ndarray) -> np.ndarray:
    output = np.zeros(mask.shape, dtype=np.int32)
    next_label = 1
    for value in sorted(int(label) for label in np.unique(mask) if label != 0):
        output[mask == value] = next_label
        next_label += 1
    return output
