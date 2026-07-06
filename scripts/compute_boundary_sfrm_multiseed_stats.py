from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PRIMARY_METRICS = {
    "dice": ("baseline_dice", "refined_dice", +1.0),
    "boundary_dice": ("baseline_boundary_dice", "refined_boundary_dice", +1.0),
    "aji": ("baseline_aji", "refined_aji", +1.0),
    "pq": ("baseline_pq", "refined_pq", +1.0),
    "confounder_fpr": ("baseline_confounder_fpr", "refined_confounder_fpr", -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed Boundary-SFRM held-out test statistics.")
    parser.add_argument("--eval-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size == 0:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def bootstrap_improvement(
    seed_rows: list[list[dict[str, str]]],
    baseline_key: str,
    refined_key: str,
    direction: float,
    iters: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iters):
        sampled_seed_sets = [seed_rows[idx] for idx in rng.integers(0, len(seed_rows), size=len(seed_rows))]
        pooled: list[float] = []
        for rows in sampled_seed_sets:
            sampled_rows = [rows[idx] for idx in rng.integers(0, len(rows), size=len(rows))]
            for row in sampled_rows:
                base = float(row[baseline_key])
                refined = float(row[refined_key])
                pooled.append(direction * (refined - base))
        draws.append(float(np.mean(pooled)))
    arr = np.asarray(draws, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_summary_rows: list[dict[str, object]] = []
    per_metric_rows: list[dict[str, object]] = []
    source_tables: list[list[dict[str, str]]] = []
    summary_payloads = []
    for eval_dir in args.eval_dirs:
        summary = json.loads((eval_dir / "test_summary.json").read_text(encoding="utf-8"))
        per_source = read_csv(eval_dir / "test_per_source.csv")
        source_tables.append(per_source)
        summary_payloads.append(summary)
        run_name = eval_dir.name
        seed = int(str(run_name).rsplit("seed", 1)[1])
        seed_summary_rows.append(
            {
                "run": run_name,
                "seed": seed,
                "n_samples": summary["n_samples"],
                "n_sources": summary["n_sources"],
                "baseline_dice": summary["baseline_dice"],
                "refined_dice": summary["refined_dice"],
                "delta_dice": summary["delta_dice"],
                "baseline_boundary_dice": summary["baseline_boundary_dice"],
                "refined_boundary_dice": summary["refined_boundary_dice"],
                "delta_boundary_dice": summary["delta_boundary_dice"],
                "baseline_aji": summary["baseline_aji"],
                "refined_aji": summary["refined_aji"],
                "delta_aji": summary["delta_aji"],
                "baseline_pq": summary["baseline_pq"],
                "refined_pq": summary["refined_pq"],
                "delta_pq": summary["delta_pq"],
                "baseline_confounder_fpr": summary["baseline_confounder_fpr"],
                "refined_confounder_fpr": summary["refined_confounder_fpr"],
                "delta_confounder_fpr": summary["delta_confounder_fpr"],
            }
        )

    for metric_name, (baseline_key, refined_key, direction) in PRIMARY_METRICS.items():
        seed_improvements = []
        wilcoxon_ps = []
        for summary_row, per_source in zip(seed_summary_rows, source_tables):
            baseline = np.asarray([float(row[baseline_key]) for row in per_source], dtype=np.float64)
            refined = np.asarray([float(row[refined_key]) for row in per_source], dtype=np.float64)
            improvement = float(np.mean(direction * (refined - baseline)))
            seed_improvements.append(improvement)
            wilcoxon_ps.append(safe_wilcoxon(baseline, refined))

        ci_low, ci_high, bootstrap_p = bootstrap_improvement(
            source_tables,
            baseline_key=baseline_key,
            refined_key=refined_key,
            direction=direction,
            iters=args.bootstrap_iters,
            seed=args.seed,
        )
        per_metric_rows.append(
            {
                "metric": metric_name,
                "mean_improvement": float(np.mean(seed_improvements)),
                "std_improvement": float(np.std(seed_improvements, ddof=0)),
                "positive_seeds": int(sum(impr > 0.0 for impr in seed_improvements)),
                "total_seeds": len(seed_improvements),
                "wilcoxon_p_seed42": next((p for row, p in zip(seed_summary_rows, wilcoxon_ps) if row["seed"] == 42), float("nan")),
                "wilcoxon_p_seed7": next((p for row, p in zip(seed_summary_rows, wilcoxon_ps) if row["seed"] == 7), float("nan")),
                "wilcoxon_p_seed123": next((p for row, p in zip(seed_summary_rows, wilcoxon_ps) if row["seed"] == 123), float("nan")),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "bootstrap_p_two_sided": bootstrap_p,
            }
        )

    write_csv(args.output_dir / "seed_test_summary.csv", sorted(seed_summary_rows, key=lambda row: int(row["seed"])))
    write_csv(args.output_dir / "paired_test_statistics.csv", per_metric_rows)
    payload = {
        "eval_dirs": [str(path) for path in args.eval_dirs],
        "n_seeds": len(args.eval_dirs),
        "runs": summary_payloads,
        "paired_statistics": per_metric_rows,
    }
    (args.output_dir / "paired_test_statistics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
