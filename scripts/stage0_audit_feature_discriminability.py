from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit univariate discriminability of leakage-free SFRM features."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bad-dice-threshold", type=float, default=0.65)
    parser.add_argument("--top-error-quantile", type=float, default=0.75)
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


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None, int, int]:
    x, y = finite_pair(scores, labels.astype(np.float64))
    if y.size == 0:
        return None, None, 0, 0
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return None, None, positives, negatives
    auc = float(roc_auc_score(y, x))
    return auc, max(auc, 1.0 - auc), positives, negatives


def safe_spearman(feature: np.ndarray, target: np.ndarray) -> tuple[float | None, float | None, int]:
    x, y = finite_pair(feature, target)
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None, None, int(x.size)
    rho, p_value = spearmanr(x, y)
    return float(rho), float(p_value), int(x.size)


def safe_mannwhitney(labels: np.ndarray, values: np.ndarray) -> tuple[float | None, float | None]:
    x, y = finite_pair(values, labels.astype(np.float64))
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return None, None
    positive = x[y.astype(bool)]
    negative = x[~y.astype(bool)]
    stat, p_value = mannwhitneyu(positive, negative, alternative="two-sided")
    return float(stat), float(p_value)


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
        raise RuntimeError(f"No feature rows found in {args.features_csv}")

    feature_cols = [name for name in rows[0] if name.startswith("feat__")]
    eval_cols = [name for name in rows[0] if name.startswith("eval__")]
    if not feature_cols:
        raise RuntimeError("No deployable feature columns found.")

    dice = numeric_column(rows, "eval__dice")
    boundary = numeric_column(rows, "eval__boundary_dice")
    error = numeric_column(rows, "eval__error_area_frac")

    labels = {
        f"bad_dice_lt_{args.bad_dice_threshold:g}": dice < args.bad_dice_threshold,
        "low_boundary_dice_lt_median": boundary < float(np.nanmedian(boundary)),
        f"high_error_ge_q{args.top_error_quantile:g}": error >= float(np.nanquantile(error, args.top_error_quantile)),
    }
    boundary_error = numeric_column(rows, "eval__boundary_error_area_frac")
    lecr_uncertainty = numeric_column(rows, "eval__lecr_uncertainty")
    lecr_boundary = numeric_column(rows, "eval__lecr_boundary_error")
    local_labels = {
        "low_boundary_dice_le_q25": boundary <= float(np.nanquantile(boundary, 0.25)),
        f"high_boundary_error_ge_q{args.top_error_quantile:g}": boundary_error
        >= float(np.nanquantile(boundary_error, args.top_error_quantile)),
        f"high_lecr_boundary_ge_q{args.top_error_quantile:g}": lecr_boundary
        >= float(np.nanquantile(lecr_boundary, args.top_error_quantile)),
        f"high_lecr_uncertainty_ge_q{args.top_error_quantile:g}": lecr_uncertainty
        >= float(np.nanquantile(lecr_uncertainty, args.top_error_quantile)),
        "gray_high_dice_high_boundary_error": (dice >= float(np.nanmedian(dice)))
        & (boundary_error >= float(np.nanquantile(boundary_error, args.top_error_quantile))),
    }
    labels.update(local_labels)

    auc_rows: list[dict[str, Any]] = []
    for label_name, label_values in labels.items():
        for feature in feature_cols:
            values = numeric_column(rows, feature)
            auc, directional_auc, positives, negatives = safe_auc(label_values, values)
            stat, mw_p = safe_mannwhitney(label_values, values)
            auc_rows.append(
                {
                    "label": label_name,
                    "feature": feature,
                    "auc_raw": auc,
                    "auc_directional": directional_auc,
                    "positives": positives,
                    "negatives": negatives,
                    "mannwhitney_u": stat,
                    "mannwhitney_p": mw_p,
                }
            )

    spearman_rows: list[dict[str, Any]] = []
    targets = {name: numeric_column(rows, name) for name in eval_cols}
    for target_name, target in targets.items():
        for feature in feature_cols:
            rho, p_value, n = safe_spearman(numeric_column(rows, feature), target)
            spearman_rows.append(
                {
                    "target": target_name,
                    "feature": feature,
                    "spearman_rho": rho,
                    "spearman_abs": abs(rho) if rho is not None else None,
                    "spearman_p": p_value,
                    "n": n,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    auc_path = args.output_dir / "univariate_auroc.csv"
    spearman_path = args.output_dir / "spearman_correlations.csv"
    write_csv(auc_path, auc_rows)
    write_csv(spearman_path, spearman_rows)

    top_auc = sorted(
        [row for row in auc_rows if row["auc_directional"] is not None],
        key=lambda row: float(row["auc_directional"]),
        reverse=True,
    )[:20]
    top_spearman = sorted(
        [row for row in spearman_rows if row["spearman_abs"] is not None],
        key=lambda row: float(row["spearman_abs"]),
        reverse=True,
    )[:20]
    summary = {
        "features_csv": str(args.features_csv),
        "n_rows": len(rows),
        "n_features": len(feature_cols),
        "labels": {
            name: {
                "positives": int(values.sum()),
                "negatives": int(values.size - values.sum()),
            }
            for name, values in labels.items()
        },
        "top_auc": top_auc,
        "top_spearman": top_spearman,
        "outputs": {
            "univariate_auroc": str(auc_path),
            "spearman_correlations": str(spearman_path),
        },
    }
    summary_path = args.output_dir / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_safe)

    print(f"rows={len(rows)}")
    print(f"features={len(feature_cols)}")
    print(f"wrote={auc_path}")
    print(f"wrote={spearman_path}")
    print(f"wrote={summary_path}")


if __name__ == "__main__":
    main()
