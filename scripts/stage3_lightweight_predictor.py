from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stage1_review_budget_simulation import (
    build_labels,
    build_scores,
    read_table,
    review_metrics,
)


PATCH_SOURCE_RE = re.compile(r"^(?P<source>.+?)_y\d+_x\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate lightweight leakage-free reliability predictors."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def numeric_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        for i, row in enumerate(rows):
            raw = row[column]
            matrix[i, j] = float(raw) if raw not in ("", "nan", "NaN", "None") else math.nan
    return matrix


def source_id(patch_id: str) -> str:
    match = PATCH_SOURCE_RE.match(patch_id)
    if match:
        return match.group("source")
    parts = patch_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return patch_id


def feature_sets(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    all_features = [name for name in rows[0] if name.startswith("feat__")]
    leakage_tokens = ("gt", "dice", "error", "lecr")
    leakage = [
        name for name in all_features if any(token in name.lower() for token in leakage_tokens)
    ]
    if leakage:
        raise RuntimeError(f"Deployable feature columns contain leakage tokens: {leakage}")

    def has_family(name: str, family: str) -> bool:
        return name.startswith(f"feat__{family}__")

    global_cols = [name for name in all_features if has_family(name, "global_uncertainty")]
    sfrm_cols = [
        name
        for name in all_features
        if not has_family(name, "global_uncertainty")
        and (
            has_family(name, "boundary_risk")
            or has_family(name, "uncertainty_cluster")
            or has_family(name, "topology_risk")
            or has_family(name, "anatomical_topological_consistency")
            or has_family(name, "feature_ambiguity")
        )
    ]
    return {
        "global_features": global_cols,
        "sfrm_features": sfrm_cols,
        "all_deployable_features": all_features,
    }


def make_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=0.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "lasso_logistic_l1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def splitter(labels: np.ndarray, groups: np.ndarray, max_folds: int, random_state: int):
    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        n_splits = min(max_folds, unique_groups.size)
        return GroupKFold(n_splits=n_splits).split(np.zeros_like(labels), labels, groups)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    n_splits = min(max_folds, positives, negatives)
    if n_splits < 2:
        return None
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    ).split(np.zeros_like(labels), labels)


def out_of_fold_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: Pipeline,
    max_folds: int,
    random_state: int,
) -> tuple[np.ndarray, int, str]:
    splits = splitter(y, groups, max_folds, random_state)
    if splits is None:
        return np.full(y.size, float(np.mean(y)), dtype=np.float64), 0, "constant_prevalence"

    predictions = np.full(y.size, np.nan, dtype=np.float64)
    used_folds = 0
    mode = "group_kfold" if np.unique(groups).size >= 2 else "stratified_kfold"
    for train_idx, test_idx in splits:
        if np.unique(y[train_idx]).size < 2:
            predictions[test_idx] = float(np.mean(y[train_idx]))
            continue
        estimator = model
        estimator.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = estimator.predict_proba(x[test_idx])[:, 1]
        used_folds += 1
    if np.isnan(predictions).any():
        predictions[np.isnan(predictions)] = float(np.nanmean(predictions))
    return predictions, used_folds, mode


def safe_auc_pair(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    mask = np.isfinite(scores)
    y = labels[mask].astype(bool)
    x = scores[mask]
    if y.sum() == 0 or y.sum() == y.size:
        return None, None
    auc = float(roc_auc_score(y, x))
    return auc, max(auc, 1.0 - auc)


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

    patch_ids = [row["patch_id"] for row in rows]
    groups = np.asarray([source_id(pid) for pid in patch_ids])
    labels = build_labels(rows, args.bad_dice_threshold, args.top_error_quantile)
    predefined_scores = build_scores(rows)
    sets = feature_sets(rows)
    models = make_models(args.random_state)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for label_name, y_bool in labels.items():
        y = y_bool.astype(int)

        for score_name, score_values in predefined_scores.items():
            auroc_raw, auroc_directional = safe_auc_pair(y_bool, score_values)
            score_rows.append(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "label": label_name,
                    "predictor": f"predefined::{score_name}",
                    "feature_set": "predefined_score",
                    "cv_mode": "none",
                    "folds_used": 0,
                    "positives": int(y.sum()),
                    "negatives": int(y.size - y.sum()),
                    "auroc_raw": auroc_raw,
                    "auroc_directional": auroc_directional,
                    "auprc": safe_auprc(y_bool, score_values),
                }
            )
            for budget in args.budgets:
                metric_rows.append(
                    {
                        "dataset": args.dataset,
                        "model": args.model,
                        "label": label_name,
                        "predictor": f"predefined::{score_name}",
                        "feature_set": "predefined_score",
                        "cv_mode": "none",
                        **review_metrics(y_bool, score_values, budget),
                    }
                )

        for set_name, columns in sets.items():
            if not columns:
                continue
            x = numeric_matrix(rows, columns)
            for model_name, estimator in models.items():
                pred, folds_used, cv_mode = out_of_fold_predictions(
                    x=x,
                    y=y,
                    groups=groups,
                    model=estimator,
                    max_folds=args.folds,
                    random_state=args.random_state,
                )
                predictor_name = f"{model_name}::{set_name}"
                auroc_raw, auroc_directional = safe_auc_pair(y_bool, pred)
                score_rows.append(
                    {
                        "dataset": args.dataset,
                        "model": args.model,
                        "label": label_name,
                        "predictor": predictor_name,
                        "feature_set": set_name,
                        "cv_mode": cv_mode,
                        "folds_used": folds_used,
                        "positives": int(y.sum()),
                        "negatives": int(y.size - y.sum()),
                        "auroc_raw": auroc_raw,
                        "auroc_directional": auroc_directional,
                        "auprc": safe_auprc(y_bool, pred),
                    }
                )
                for i, patch_id in enumerate(patch_ids):
                    prediction_rows.append(
                        {
                            "dataset": args.dataset,
                            "model": args.model,
                            "label": label_name,
                            "patch_id": patch_id,
                            "source_id": groups[i],
                            "predictor": predictor_name,
                            "feature_set": set_name,
                            "prediction": float(pred[i]),
                            "target": int(y[i]),
                        }
                    )
                for budget in args.budgets:
                    metric_rows.append(
                        {
                            "dataset": args.dataset,
                            "model": args.model,
                            "label": label_name,
                            "predictor": predictor_name,
                            "feature_set": set_name,
                            "cv_mode": cv_mode,
                            **review_metrics(y_bool, pred, budget),
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "predictor_review_budget_metrics.csv"
    scores_path = args.output_dir / "predictor_discriminability.csv"
    predictions_path = args.output_dir / "oof_predictions.csv"
    summary_path = args.output_dir / "predictor_summary.json"
    write_csv(metrics_path, metric_rows)
    write_csv(scores_path, score_rows)
    write_csv(predictions_path, prediction_rows)

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "features_csv": str(args.features_csv),
        "n_rows": len(rows),
        "n_sources": int(np.unique(groups).size),
        "feature_sets": {name: len(cols) for name, cols in sets.items()},
        "labels": {
            name: {"positives": int(values.sum()), "negatives": int(values.size - values.sum())}
            for name, values in labels.items()
        },
        "outputs": {
            "metrics": str(metrics_path),
            "discriminability": str(scores_path),
            "predictions": str(predictions_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_safe)

    print(f"rows={len(rows)}")
    print(f"sources={summary['n_sources']}")
    print(f"labels={len(labels)}")
    print(f"feature_sets={summary['feature_sets']}")
    print(f"wrote={metrics_path}")
    print(f"wrote={scores_path}")


if __name__ == "__main__":
    main()
