from __future__ import annotations

import argparse
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready SFRM result tables.")
    parser.add_argument(
        "--multiseed-bootstrap",
        type=Path,
        default=Path("experiments/summaries/stage4_multiseed_summary/source_bootstrap_comparisons_multiseed.csv"),
    )
    parser.add_argument(
        "--external-bootstrap",
        type=Path,
        default=Path("experiments/summaries/stage4_bootstrap_fets_single_sites18_19_20/source_bootstrap_comparisons.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/summaries/stage5_paper_tables"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f4(value: float) -> str:
    return f"{value:.4f}"


def summarize_multiseed(rows: list[dict[str, str]], score_a: str, score_b: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["score_a"] != score_a or row["score_b"] != score_b:
            continue
        key = (row["dataset_short"], row["model_family"], row["label"])
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, model_family, label), items in sorted(grouped.items()):
        diffs = [float(item["recall_diff"]) for item in items]
        a_vals = [float(item["score_a_recall"]) for item in items]
        b_vals = [float(item["score_b_recall"]) for item in items]
        sig = [item["ci_excludes_zero"].lower() == "true" for item in items]
        output.append(
            {
                "dataset": dataset,
                "model_family": model_family,
                "endpoint": label,
                "n_seeds": len(items),
                "sfrm_recall_mean": f4(stats.mean(a_vals)),
                "comparator_recall_mean": f4(stats.mean(b_vals)),
                "recall_diff_mean": f4(stats.mean(diffs)),
                "recall_diff_min": f4(min(diffs)),
                "recall_diff_max": f4(max(diffs)),
                "positive_seeds": f"{sum(v > 0 for v in diffs)}/{len(diffs)}",
                "source_bootstrap_significant_seeds": f"{sum(sig)}/{len(sig)}",
            }
        )
    return output


def external_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if int(row["positives"]) == 0 or row["recall_diff"].lower() == "nan":
            continue
        output.append(
            {
                "endpoint": row["label"],
                "sfrm_predictor": row["score_a"],
                "comparator": row["score_b"],
                "sfrm_recall": f4(float(row["score_a_recall"])),
                "comparator_recall": f4(float(row["score_b_recall"])),
                "recall_diff": f4(float(row["recall_diff"])),
                "ci95": f"[{f4(float(row['bootstrap_ci95_low']))}, {f4(float(row['bootstrap_ci95_high']))}]",
                "ci_excludes_zero": row["ci_excludes_zero"],
                "positives": row["positives"],
            }
        )
    return output


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multiseed = read_csv(args.multiseed_bootstrap)
    external = read_csv(args.external_bootstrap)

    entropy_table = summarize_multiseed(
        multiseed,
        score_a="logistic_l2::sfrm_features",
        score_b="predefined::global_max_entropy",
    )
    strong_global_table = summarize_multiseed(
        multiseed,
        score_a="logistic_l2::sfrm_features",
        score_b="logistic_l2::global_features",
    )
    lecr_entropy_table = summarize_multiseed(
        multiseed,
        score_a="lasso_logistic_l1::sfrm_features",
        score_b="predefined::global_mean_entropy",
    )
    external_table = external_rows(external)

    write_csv(args.output_dir / "table1_primary_low_boundary_vs_global_entropy.csv", entropy_table)
    write_csv(args.output_dir / "table2_low_boundary_vs_trained_global_predictor.csv", strong_global_table)
    write_csv(args.output_dir / "table3_supporting_lecr_vs_global_mean_entropy.csv", lecr_entropy_table)
    write_csv(args.output_dir / "table4_external_3d_validation.csv", external_table)

    md = [
        "# SFRM Paper Tables",
        "",
        "## Table 1. Primary endpoint: low boundary Dice vs conventional global max entropy",
        markdown_table(entropy_table),
        "",
        "## Table 2. Strong baseline: low boundary Dice vs trained global-feature predictor",
        markdown_table(strong_global_table),
        "",
        "## Table 3. Supporting endpoint: high LECR boundary vs global mean entropy",
        markdown_table(lecr_entropy_table),
        "",
        "## Table 4. External 3D FeTS/BraTS validation",
        markdown_table(external_table),
        "",
        "Interpretation guardrail: Table 1 supports the main conventional-global-uncertainty blind-spot claim. Table 2 is a required strong-baseline caveat and supports complementarity rather than universal superiority.",
    ]
    md_path = args.output_dir / "paper_tables.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote={md_path}")


if __name__ == "__main__":
    main()
