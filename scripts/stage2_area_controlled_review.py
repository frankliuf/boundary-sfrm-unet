from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from stage1_review_budget_simulation import (
    build_labels,
    build_scores,
    numeric_column,
    read_table,
    review_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run foreground-area-controlled review-budget simulation."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
    parser.add_argument("--area-bins", type=int, default=5)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros(values.size, dtype=int)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.nanquantile(values[finite], quantiles)
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.size, dtype=int)
    bins = np.digitize(values, edges[1:-1], right=True)
    bins[~finite] = -1
    return bins.astype(int)


def percentile_within_bins(scores: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    for bin_id in sorted(set(int(v) for v in bins)):
        mask = bins == bin_id
        if not mask.any():
            continue
        vals = scores[mask]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(order.size, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, order.size)
        out[mask] = ranks
    return out


def stratified_review_metrics(
    labels: np.ndarray, scores: np.ndarray, bins: np.ndarray, budget: float
) -> dict[str, float | int]:
    selected = np.zeros(labels.size, dtype=bool)
    for bin_id in sorted(set(int(v) for v in bins)):
        idx = np.flatnonzero(bins == bin_id)
        if idx.size == 0:
            continue
        k = max(1, int(math.ceil(idx.size * budget)))
        local_order = idx[np.argsort(-np.nan_to_num(scores[idx], nan=-np.inf), kind="mergesort")]
        selected[local_order[:k]] = True
    positives = int(labels.sum())
    captured = int(labels[selected].sum())
    review_count = int(selected.sum())
    n = labels.size
    base_rate = positives / n if n else 0.0
    unreviewed = n - review_count
    remaining = positives - captured
    remaining_rate = remaining / unreviewed if unreviewed > 0 else 0.0
    return {
        "n": n,
        "budget": float(budget),
        "review_count": review_count,
        "positives": positives,
        "captured": captured,
        "recall": captured / positives if positives else math.nan,
        "precision": captured / review_count if review_count else math.nan,
        "enrichment_over_random": (captured / positives) / budget
        if positives and budget > 0
        else math.nan,
        "remaining_positive_rate": remaining_rate,
        "accepted_bad_case_reduction": 1.0 - (remaining_rate / base_rate)
        if base_rate > 0
        else math.nan,
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    y = labels[mask].astype(bool)
    x = scores[mask]
    if y.sum() == 0 or y.sum() == y.size:
        return None
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    y = labels[mask].astype(bool)
    x = scores[mask]
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, x))


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    rows = read_table(args.features_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.features_csv}")

    labels = build_labels(rows, args.bad_dice_threshold, args.top_error_quantile)
    raw_scores = build_scores(rows)
    area = numeric_column(rows, "feat__global_uncertainty__foreground_area_frac")
    bins = make_quantile_bins(area, args.area_bins)
    area_controlled_scores = {
        f"{name}__area_ranked": percentile_within_bins(values, bins)
        for name, values in raw_scores.items()
    }

    metric_rows: list[dict[str, Any]] = []
    discrim_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for bin_id in sorted(set(int(v) for v in bins)):
        mask = bins == bin_id
        bin_rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "area_bin": bin_id,
                "n": int(mask.sum()),
                "area_min": float(np.nanmin(area[mask])),
                "area_max": float(np.nanmax(area[mask])),
                "area_mean": float(np.nanmean(area[mask])),
            }
        )

    score_sets = {
        **{f"{name}__raw_global_ranking": values for name, values in raw_scores.items()},
        **area_controlled_scores,
    }
    for label_name, label_values in labels.items():
        for score_name, score_values in score_sets.items():
            discrim_rows.append(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "label": label_name,
                    "score": score_name,
                    "positives": int(label_values.sum()),
                    "negatives": int(label_values.size - label_values.sum()),
                    "auroc_directional": safe_auc(label_values, score_values),
                    "auprc": safe_auprc(label_values, score_values),
                }
            )
            for budget in args.budgets:
                if score_name.endswith("__raw_global_ranking"):
                    metrics = review_metrics(label_values, score_values, budget)
                    mode = "raw_global_ranking"
                else:
                    metrics = stratified_review_metrics(label_values, score_values, bins, budget)
                    mode = "area_stratified_ranking"
                metric_rows.append(
                    {
                        "dataset": args.dataset,
                        "model": args.model,
                        "label": label_name,
                        "score": score_name,
                        "ranking_mode": mode,
                        **metrics,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "area_controlled_review_metrics.csv"
    discrim_path = args.output_dir / "area_controlled_score_discriminability.csv"
    bins_path = args.output_dir / "area_bins.csv"
    summary_path = args.output_dir / "area_controlled_summary.json"
    write_csv(metrics_path, metric_rows)
    write_csv(discrim_path, discrim_rows)
    write_csv(bins_path, bin_rows)

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "features_csv": str(args.features_csv),
        "n_rows": len(rows),
        "area_bins": args.area_bins,
        "bin_counts": {str(int(v)): int((bins == v).sum()) for v in sorted(set(bins))},
        "labels": {
            name: {
                "positives": int(values.sum()),
                "negatives": int(values.size - values.sum()),
            }
            for name, values in labels.items()
        },
        "outputs": {
            "metrics": str(metrics_path),
            "discriminability": str(discrim_path),
            "bins": str(bins_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_safe)

    print(f"rows={len(rows)}")
    print(f"area_bins={summary['bin_counts']}")
    print(f"labels={len(labels)}")
    print(f"scores={len(score_sets)}")
    print(f"wrote={metrics_path}")
    print(f"wrote={discrim_path}")


if __name__ == "__main__":
    main()
