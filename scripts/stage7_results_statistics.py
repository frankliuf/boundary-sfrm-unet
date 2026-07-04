from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from stage1_review_budget_simulation import build_labels, build_scores, read_table, review_metrics
from stage4_source_bootstrap_robustness import source_id, read_predictions, predictor_scores

RUNS = [
    ("monuseg", "point", 42, "monuseg_point_seed42"),
    ("monuseg", "point", 7, "monuseg_point_seed7"),
    ("monuseg", "point", 123, "monuseg_point_seed123"),
    ("monuseg", "prototype", 42, "monuseg_proto_seed42"),
    ("monuseg", "prototype", 7, "monuseg_proto_seed7"),
    ("monuseg", "prototype", 123, "monuseg_proto_seed123"),
    ("consep", "point", 42, "consep_grid_point_seed42"),
    ("consep", "point", 7, "consep_grid_point_seed7"),
    ("consep", "point", 123, "consep_grid_point_seed123"),
    ("consep", "prototype", 42, "consep_grid_proto_seed42"),
    ("consep", "prototype", 7, "consep_grid_proto_seed7"),
    ("consep", "prototype", 123, "consep_grid_proto_seed123"),
]

COMPARISONS = [
    ("low_boundary_dice_le_q25", "logistic_l2::sfrm_features", "predefined::global_max_entropy", "primary_vs_global_entropy"),
    ("low_boundary_dice_le_q25", "logistic_l2::sfrm_features", "logistic_l2::global_features", "strong_global_predictor"),
    ("high_lecr_boundary_ge_q0.75", "lasso_logistic_l1::sfrm_features", "predefined::global_mean_entropy", "supporting_lecr"),
]


def metric(labels: np.ndarray, scores: np.ndarray, idx: np.ndarray, budget: float) -> float:
    if idx.size == 0:
        return math.nan
    return float(review_metrics(labels[idx].astype(bool), scores[idx], budget)["recall"])


def bootstrap(labels, score_a, score_b, sources, budget=0.10, iters=5000, seed=42):
    rng = np.random.default_rng(seed)
    unique_sources = np.unique(sources)
    full_idx = np.arange(labels.size)
    full_a = metric(labels, score_a, full_idx, budget)
    full_b = metric(labels, score_b, full_idx, budget)
    full_diff = full_a - full_b
    diffs = []
    for _ in range(iters):
        sampled = rng.choice(unique_sources, size=unique_sources.size, replace=True)
        idxs = []
        for sid in sampled:
            idxs.extend(np.flatnonzero(sources == sid).tolist())
        idx = np.asarray(idxs, dtype=int)
        d = metric(labels, score_a, idx, budget) - metric(labels, score_b, idx, budget)
        if math.isfinite(d):
            diffs.append(d)
    arr = np.asarray(diffs, dtype=float)
    ci_low, ci_high = np.nanpercentile(arr, [2.5, 97.5])
    if arr.size == 0 or not math.isfinite(full_diff):
        p = math.nan
    else:
        p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
        p = min(1.0, p)
    return full_a, full_b, full_diff, float(ci_low), float(ci_high), p


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    out_rows = []
    for dataset, model_family, seed, prefix in RUNS:
        feature_path = Path(f"experiments/summaries/stage0_{prefix}_features_full.csv")
        pred_path = Path(f"experiments/summaries/stage3_predictor_{prefix}/oof_predictions.csv")
        rows = read_table(feature_path)
        labels = build_labels(rows, 0.65, 0.75)
        predefined = build_scores(rows)
        oof = read_predictions(pred_path)
        sources = np.asarray([source_id(r["patch_id"]) for r in rows])
        for label_name, score_a_name, score_b_name, comparison in COMPARISONS:
            y = labels[label_name].astype(int)
            score_a = predictor_scores(rows, predefined, oof, label_name, score_a_name)
            score_b = predictor_scores(rows, predefined, oof, label_name, score_b_name)
            a, b, diff, lo, hi, p = bootstrap(y, score_a, score_b, sources, seed=seed)
            out_rows.append({
                "dataset": dataset,
                "model_family": model_family,
                "seed": seed,
                "run_prefix": prefix,
                "comparison": comparison,
                "endpoint": label_name,
                "score_a": score_a_name,
                "score_b": score_b_name,
                "n_patches": len(rows),
                "n_sources": int(np.unique(sources).size),
                "positives": int(y.sum()),
                "score_a_recall_10pct": a,
                "score_b_recall_10pct": b,
                "recall_diff": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "bootstrap_p_two_sided": p,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })
    out = Path("experiments/summaries/stage7_results_statistics/source_bootstrap_pvalues.csv")
    write_csv(out, out_rows)
    print("wrote", out, "rows", len(out_rows))


if __name__ == "__main__":
    main()
