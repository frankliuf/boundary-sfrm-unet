from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate fixed-budget human review using leakage-free global "
            "uncertainty and SFRM risk scores."
        )
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
        help="Fractions of cases reviewed.",
    )
    return parser.parse_args()


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row[column]
        values.append(float(raw) if raw not in ("", "nan", "NaN", "None") else math.nan)
    return np.asarray(values, dtype=np.float64)


def percentile_score(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    scores = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return scores
    order = np.argsort(values[finite], kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, order.size)
    scores[finite] = ranks
    scores[~finite] = 0.0
    return scores


def robust_deviation_score(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    scores = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return scores
    vals = values[finite]
    median = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - median)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.nanstd(vals))
    if scale <= 1e-12:
        return scores
    scores[finite] = np.abs(vals - median) / scale
    return percentile_score(scores)


def mean_available(score_parts: list[np.ndarray], n: int) -> np.ndarray:
    if not score_parts:
        return np.zeros(n, dtype=np.float64)
    stacked = np.vstack(score_parts)
    return np.nanmean(stacked, axis=0)


def max_available(score_parts: list[np.ndarray], n: int) -> np.ndarray:
    if not score_parts:
        return np.zeros(n, dtype=np.float64)
    stacked = np.vstack(score_parts)
    return np.nanmax(stacked, axis=0)


def require_columns(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    available = set(rows[0])
    return [column for column in columns if column in available]


def assert_no_deployable_leakage(rows: list[dict[str, str]]) -> None:
    bad_tokens = ("gt", "dice", "error", "lecr")
    leakage = [
        name
        for name in rows[0]
        if name.startswith("feat__") and any(token in name.lower() for token in bad_tokens)
    ]
    if leakage:
        raise RuntimeError(f"Deployable feature columns contain leakage tokens: {leakage}")


def build_labels(
    rows: list[dict[str, str]], bad_dice_threshold: float, top_error_quantile: float
) -> dict[str, np.ndarray]:
    dice = numeric_column(rows, "eval__dice")
    boundary = numeric_column(rows, "eval__boundary_dice")
    error = numeric_column(rows, "eval__error_area_frac")
    boundary_error = numeric_column(rows, "eval__boundary_error_area_frac")
    lecr_uncertainty = numeric_column(rows, "eval__lecr_uncertainty")
    lecr_boundary = numeric_column(rows, "eval__lecr_boundary_error")
    return {
        f"bad_dice_lt_{bad_dice_threshold:g}": dice < bad_dice_threshold,
        "low_boundary_dice_le_q25": boundary <= float(np.nanquantile(boundary, 0.25)),
        f"high_error_ge_q{top_error_quantile:g}": error
        >= float(np.nanquantile(error, top_error_quantile)),
        f"high_boundary_error_ge_q{top_error_quantile:g}": boundary_error
        >= float(np.nanquantile(boundary_error, top_error_quantile)),
        f"high_lecr_boundary_ge_q{top_error_quantile:g}": lecr_boundary
        >= float(np.nanquantile(lecr_boundary, top_error_quantile)),
        f"high_lecr_uncertainty_ge_q{top_error_quantile:g}": lecr_uncertainty
        >= float(np.nanquantile(lecr_uncertainty, top_error_quantile)),
        "gray_high_dice_high_boundary_error": (dice >= float(np.nanmedian(dice)))
        & (boundary_error >= float(np.nanquantile(boundary_error, top_error_quantile))),
    }


def build_scores(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    n = len(rows)

    def col(name: str) -> np.ndarray:
        return numeric_column(rows, name)

    def available(names: list[str]) -> list[str]:
        return require_columns(rows, names)

    boundary_cols = available(
        [
            "feat__boundary_risk__pred_boundary_area_frac",
            "feat__boundary_risk__boundary_mean_entropy",
            "feat__boundary_risk__boundary_mean_margin_uncertainty",
            "feat__boundary_risk__boundary_high_entropy_frac",
        ]
    )
    boundary_confidence_cols = available(
        [
            "feat__boundary_risk__boundary_mean_entropy",
            "feat__boundary_risk__boundary_mean_margin_uncertainty",
            "feat__boundary_risk__boundary_high_entropy_frac",
        ]
    )
    uncertainty_cluster_cols = available(
        [
            "feat__uncertainty_cluster__high_uncertainty_area_frac",
            "feat__uncertainty_cluster__high_uncertainty_largest_component_frac",
            "feat__uncertainty_cluster__high_uncertainty_small_component_count",
            "feat__uncertainty_cluster__high_uncertainty_threshold",
        ]
    )
    topology_one_sided_cols = available(
        [
            "feat__topology_risk__threshold_area_frac_std",
            "feat__topology_risk__small_component_count",
            "feat__topology_risk__component_area_cv",
            "feat__topology_risk__mean_eccentricity",
            "feat__anatomical_topological_consistency__opening_residual_frac",
            "feat__anatomical_topological_consistency__closing_residual_frac",
        ]
    )
    topology_deviation_cols = available(
        [
            "feat__topology_risk__component_count",
            "feat__topology_risk__mean_component_area_frac",
            "feat__topology_risk__largest_component_area_frac",
        ]
    )

    scores: dict[str, np.ndarray] = {
        "global_mean_entropy": percentile_score(
            col("feat__global_uncertainty__mean_entropy")
        ),
        "global_max_entropy": percentile_score(col("feat__global_uncertainty__max_entropy")),
        "global_mean_margin_uncertainty": percentile_score(
            col("feat__global_uncertainty__mean_margin_uncertainty")
        ),
        "foreground_area_frac": percentile_score(
            col("feat__global_uncertainty__foreground_area_frac")
        ),
    }
    scores["boundary_risk_score"] = mean_available(
        [percentile_score(col(name)) for name in boundary_cols], n
    )
    boundary_high_confidence_parts = [percentile_score(col(name)) for name in boundary_confidence_cols]
    scores["boundary_overconfidence_score"] = mean_available(
        [1.0 - part for part in boundary_high_confidence_parts], n
    )
    scores["boundary_abnormality_score"] = mean_available(
        [robust_deviation_score(col(name)) for name in boundary_confidence_cols], n
    )
    scores["boundary_dual_risk_score"] = max_available(
        [
            scores["boundary_risk_score"],
            scores["boundary_overconfidence_score"],
            scores["boundary_abnormality_score"],
        ],
        n,
    )
    scores["uncertainty_cluster_score"] = mean_available(
        [percentile_score(col(name)) for name in uncertainty_cluster_cols], n
    )
    scores["topology_risk_score"] = mean_available(
        [percentile_score(col(name)) for name in topology_one_sided_cols]
        + [robust_deviation_score(col(name)) for name in topology_deviation_cols],
        n,
    )
    scores["sfrm_composite_score"] = mean_available(
        [
            scores["boundary_risk_score"],
            scores["uncertainty_cluster_score"],
            scores["topology_risk_score"],
        ],
        n,
    )
    scores["sfrm_balanced_score"] = mean_available(
        [
            scores["boundary_dual_risk_score"],
            scores["uncertainty_cluster_score"],
            scores["topology_risk_score"],
        ],
        n,
    )
    return scores


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


def review_metrics(labels: np.ndarray, scores: np.ndarray, budget: float) -> dict[str, float | int]:
    n = labels.size
    k = max(1, int(math.ceil(n * budget)))
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
    selected = order[:k]
    positives = int(labels.sum())
    captured = int(labels[selected].sum())
    base_rate = positives / n if n else 0.0
    unreviewed = n - k
    remaining = positives - captured
    remaining_rate = remaining / unreviewed if unreviewed > 0 else 0.0
    return {
        "n": n,
        "budget": float(budget),
        "review_count": k,
        "positives": positives,
        "captured": captured,
        "recall": captured / positives if positives else math.nan,
        "precision": captured / k if k else math.nan,
        "enrichment_over_random": (captured / positives) / budget
        if positives and budget > 0
        else math.nan,
        "remaining_positive_rate": remaining_rate,
        "accepted_bad_case_reduction": 1.0 - (remaining_rate / base_rate)
        if base_rate > 0
        else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    rows = read_table(args.features_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.features_csv}")
    assert_no_deployable_leakage(rows)

    labels = build_labels(rows, args.bad_dice_threshold, args.top_error_quantile)
    scores = build_scores(rows)

    metric_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for label_name, label_values in labels.items():
        for score_name, score_values in scores.items():
            score_rows.append(
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
                row = review_metrics(label_values, score_values, budget)
                metric_rows.append(
                    {
                        "dataset": args.dataset,
                        "model": args.model,
                        "label": label_name,
                        "score": score_name,
                        **row,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "review_budget_metrics.csv"
    score_path = args.output_dir / "score_discriminability.csv"
    summary_path = args.output_dir / "review_budget_summary.json"
    write_csv(review_path, metric_rows)
    write_csv(score_path, score_rows)

    focus_budget = 0.10
    top_at_10 = sorted(
        [
            row
            for row in metric_rows
            if abs(float(row["budget"]) - focus_budget) < 1e-9
        ],
        key=lambda row: (
            str(row["label"]),
            -float(row["recall"]) if math.isfinite(float(row["recall"])) else 0.0,
        ),
    )
    summary = {
        "features_csv": str(args.features_csv),
        "dataset": args.dataset,
        "model": args.model,
        "n_rows": len(rows),
        "labels": {
            name: {
                "positives": int(values.sum()),
                "negatives": int(values.size - values.sum()),
            }
            for name, values in labels.items()
        },
        "score_names": sorted(scores),
        "outputs": {
            "review_budget_metrics": str(review_path),
            "score_discriminability": str(score_path),
        },
        "top_rows_at_10_percent_budget": top_at_10[:30],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_safe)

    print(f"rows={len(rows)}")
    print(f"labels={len(labels)}")
    print(f"scores={len(scores)}")
    print(f"wrote={review_path}")
    print(f"wrote={score_path}")
    print(f"wrote={summary_path}")


if __name__ == "__main__":
    main()
