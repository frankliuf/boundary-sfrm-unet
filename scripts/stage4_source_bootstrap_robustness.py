from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from stage1_review_budget_simulation import build_labels, build_scores, read_table, review_metrics


PATCH_SOURCE_RE = re.compile(r"^(?P<source>.+?)_y\d+_x\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-level bootstrap robustness for key review-budget comparisons."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
    return parser.parse_args()


def source_id(patch_id: str) -> str:
    match = PATCH_SOURCE_RE.match(patch_id)
    if match:
        return match.group("source")
    parts = patch_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return patch_id


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path: Path) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[(row["label"], row["predictor"], row["patch_id"])] = float(row["prediction"])
    return output


def predictor_scores(
    rows: list[dict[str, str]],
    predefined_scores: dict[str, np.ndarray],
    oof_predictions: dict[tuple[str, str], float],
    label: str,
    predictor: str,
) -> np.ndarray:
    if predictor.startswith("predefined::"):
        name = predictor.split("::", 1)[1]
        if name not in predefined_scores:
            raise KeyError(f"Unknown predefined score: {name}")
        return predefined_scores[name]

    values = []
    for row in rows:
        key = (label, predictor, row["patch_id"])
        if key not in oof_predictions:
            raise KeyError(f"Missing OOF prediction for {key}")
        values.append(oof_predictions[key])
    return np.asarray(values, dtype=np.float64)


def metric_on_indices(labels: np.ndarray, scores: np.ndarray, indices: np.ndarray, budget: float) -> float:
    if indices.size == 0:
        return math.nan
    sub_labels = labels[indices]
    sub_scores = scores[indices]
    return float(review_metrics(sub_labels.astype(bool), sub_scores, budget)["recall"])


def bootstrap_difference(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    sources: np.ndarray,
    budget: float,
    iters: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    unique_sources = np.unique(sources)
    full_indices = np.arange(labels.size)
    full_a = metric_on_indices(labels, score_a, full_indices, budget)
    full_b = metric_on_indices(labels, score_b, full_indices, budget)
    diffs = []
    for _ in range(iters):
        sampled_sources = rng.choice(unique_sources, size=unique_sources.size, replace=True)
        sampled_indices = []
        for sid in sampled_sources:
            sampled_indices.extend(np.flatnonzero(sources == sid).tolist())
        idx = np.asarray(sampled_indices, dtype=int)
        # Duplicated source resampling means duplicated patches. This is intended
        # for a source-level bootstrap of the review ranking statistic.
        diff = metric_on_indices(labels, score_a, idx, budget) - metric_on_indices(
            labels, score_b, idx, budget
        )
        if math.isfinite(diff):
            diffs.append(diff)
    arr = np.asarray(diffs, dtype=np.float64)
    return (
        full_a,
        full_b,
        full_a - full_b,
        float(np.nanpercentile(arr, 2.5)),
        float(np.nanpercentile(arr, 97.5)),
    )


def comparison_plan(dataset: str, model: str) -> list[dict[str, str]]:
    common = [
        {
            "label": "high_lecr_boundary_ge_q0.75",
            "score_a": "lasso_logistic_l1::sfrm_features",
            "score_b": "predefined::global_mean_entropy",
            "hypothesis": "SFRM improves local boundary-error screening over global mean entropy.",
        },
        {
            "label": "low_boundary_dice_le_q25",
            "score_a": "logistic_l2::sfrm_features",
            "score_b": "predefined::global_max_entropy",
            "hypothesis": "SFRM improves low-boundary-quality screening over global max entropy.",
        },
        {
            "label": "low_boundary_dice_le_q25",
            "score_a": "logistic_l2::sfrm_features",
            "score_b": "logistic_l2::global_features",
            "hypothesis": "SFRM is compared against a trained global-feature reliability predictor for low-boundary-quality screening.",
        },
        {
            "label": "high_lecr_boundary_ge_q0.75",
            "score_a": "lasso_logistic_l1::sfrm_features",
            "score_b": "logistic_l2::global_features",
            "hypothesis": "SFRM is compared against a trained global-feature reliability predictor for local boundary-error screening.",
        },
        {
            "label": "bad_dice_lt_0.65",
            "score_a": "logistic_l2::global_features",
            "score_b": "logistic_l2::sfrm_features",
            "hypothesis": "Global features remain competitive for macro-area failures.",
        },
    ]
    if dataset.lower() == "monuseg" and model == "point_seed42":
        common.append(
            {
                "label": "gray_high_dice_high_boundary_error",
                "score_a": "predefined::boundary_overconfidence_score",
                "score_b": "predefined::global_mean_entropy",
                "hypothesis": "Boundary overconfidence captures review-budget global-entropy misses.",
            }
        )
    if dataset.lower() == "consep" and model == "prototype_seed42":
        common = [
            {
                **item,
                "hypothesis": "Core CoNSeP prototype recall-doubling claim.",
            }
            if item["label"] == "low_boundary_dice_le_q25"
            and item["score_a"] == "logistic_l2::sfrm_features"
            and item["score_b"] == "predefined::global_max_entropy"
            else item
            for item in common
        ]
    return common


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    rows = read_table(args.features_csv)
    labels = build_labels(rows, args.bad_dice_threshold, args.top_error_quantile)
    predefined_scores = build_scores(rows)
    oof_predictions = read_predictions(args.predictions_csv)
    sources = np.asarray([source_id(row["patch_id"]) for row in rows])
    rng = np.random.default_rng(args.random_state)

    result_rows: list[dict[str, Any]] = []
    for item in comparison_plan(args.dataset, args.model):
        label_name = item["label"]
        if label_name not in labels:
            raise KeyError(f"Missing label {label_name}")
        y = labels[label_name].astype(int)
        score_a = predictor_scores(rows, predefined_scores, oof_predictions, label_name, item["score_a"])
        score_b = predictor_scores(rows, predefined_scores, oof_predictions, label_name, item["score_b"])
        full_a, full_b, diff, ci_low, ci_high = bootstrap_difference(
            labels=y,
            score_a=score_a,
            score_b=score_b,
            sources=sources,
            budget=args.budget,
            iters=args.bootstrap_iters,
            rng=rng,
        )
        result_rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "label": label_name,
                "score_a": item["score_a"],
                "score_b": item["score_b"],
                "hypothesis": item["hypothesis"],
                "budget": args.budget,
                "n_patches": len(rows),
                "n_sources": int(np.unique(sources).size),
                "positives": int(y.sum()),
                "score_a_recall": full_a,
                "score_b_recall": full_b,
                "recall_diff": diff,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "source_bootstrap_comparisons.csv"
    summary_path = args.output_dir / "source_bootstrap_summary.json"
    write_csv(results_path, result_rows)
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "features_csv": str(args.features_csv),
        "predictions_csv": str(args.predictions_csv),
        "budget": args.budget,
        "bootstrap_iters": args.bootstrap_iters,
        "n_sources": int(np.unique(sources).size),
        "outputs": {"comparisons": str(results_path)},
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_safe)

    print(f"rows={len(rows)}")
    print(f"sources={summary['n_sources']}")
    print(f"comparisons={len(result_rows)}")
    print(f"wrote={results_path}")


if __name__ == "__main__":
    main()
