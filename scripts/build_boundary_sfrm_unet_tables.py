from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiments" / "boundary_sfrm_runs"
OUT_DIR = ROOT / "outputs" / "paper_tables"


@dataclass(frozen=True)
class MainDatasetConfig:
    dataset: str
    baseline_run: str
    sfrm_run: str


MAIN_DATASETS = (
    MainDatasetConfig(
        dataset="TNBC",
        baseline_run="tnbc_fullsup_unet_seed42",
        sfrm_run="tnbc_sfrm_unet_learned_failure_head_freeze8_seed42",
    ),
    MainDatasetConfig(
        dataset="CryoNuSeg",
        baseline_run="cryonuseg_fullsup_unet_seed42",
        sfrm_run="cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42",
    ),
)


MONUSEG_FIXED_BUDGET = {
    "Baseline U-Net": ("monuseg_fullsup_unet_seed42", 6, False),
    "Two-pass, no risk": ("monuseg_fullsup_two_pass_no_risk_seed42", 8, True),
    "Entropy-only refinement": ("monuseg_fullsup_entropy_only_seed42", 8, True),
    "Boundary-SFRM v3": ("monuseg_fullsup_boundary_sfrm_v3_seed42", 8, True),
    "Learned failure head v3": ("monuseg_fullsup_learned_failure_head_v3_teacher10_seed42", 8, True),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"val_epoch_{epoch:02d}_summary.json"


def _metric(obj: dict, refined: str, plain: str) -> float:
    if refined in obj:
        return float(obj[refined])
    if plain in obj:
        return float(obj[plain])
    raise KeyError(f"Missing metric {refined}/{plain}")


def _epoch_from_name(path: Path) -> int:
    parts = path.stem.split("_")
    return int(parts[2])


def _select_best_boundary(run_dir: Path, refined: bool) -> tuple[int, dict]:
    best: tuple[float, int, dict] | None = None
    metric_names = ("refined_boundary_dice", "boundary_dice") if refined else ("boundary_dice", "boundary_dice")
    for path in sorted(run_dir.glob("val_epoch_*_summary.json")):
        obj = _load_json(path)
        score = _metric(obj, metric_names[0], metric_names[1])
        epoch = _epoch_from_name(path)
        if best is None or score > best[0]:
            best = (score, epoch, obj)
    if best is None:
        raise FileNotFoundError(f"No validation summaries found in {run_dir}")
    return best[1], best[2]


def _select_peak_aji(run_dir: Path, refined: bool) -> tuple[int, dict]:
    best: tuple[float, int, dict] | None = None
    metric_names = ("refined_aji", "aji") if refined else ("aji", "aji")
    for path in sorted(run_dir.glob("val_epoch_*_summary.json")):
        obj = _load_json(path)
        score = _metric(obj, metric_names[0], metric_names[1])
        epoch = _epoch_from_name(path)
        if best is None or score > best[0]:
            best = (score, epoch, obj)
    if best is None:
        raise FileNotFoundError(f"No validation summaries found in {run_dir}")
    return best[1], best[2]


def build_main_summary() -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    audit_rows: list[dict] = []
    for cfg in MAIN_DATASETS:
        baseline_dir = RUN_DIR / cfg.baseline_run
        sfrm_dir = RUN_DIR / cfg.sfrm_run

        baseline_epoch, baseline_obj = _select_best_boundary(baseline_dir, refined=False)
        sfrm_epoch, sfrm_obj = _select_best_boundary(sfrm_dir, refined=True)
        peak_aji_epoch, peak_aji_obj = _select_peak_aji(sfrm_dir, refined=True)

        summary_rows.append(
            {
                "dataset": cfg.dataset,
                "baseline_epoch": baseline_epoch,
                "baseline_dice": _metric(baseline_obj, "dice", "dice"),
                "baseline_boundary_dice": _metric(baseline_obj, "boundary_dice", "boundary_dice"),
                "baseline_aji": _metric(baseline_obj, "aji", "aji"),
                "baseline_pq": _metric(baseline_obj, "pq", "pq"),
                "baseline_confounder_fpr": _metric(baseline_obj, "confounder_fpr", "confounder_fpr"),
                "sfrm_epoch": sfrm_epoch,
                "sfrm_dice": _metric(sfrm_obj, "refined_dice", "dice"),
                "sfrm_boundary_dice": _metric(sfrm_obj, "refined_boundary_dice", "boundary_dice"),
                "sfrm_aji": _metric(sfrm_obj, "refined_aji", "aji"),
                "sfrm_pq": _metric(sfrm_obj, "refined_pq", "pq"),
                "sfrm_confounder_fpr": _metric(sfrm_obj, "refined_confounder_fpr", "confounder_fpr"),
                "delta_dice": float(sfrm_obj["delta_dice"]),
                "delta_boundary_dice": float(sfrm_obj["delta_boundary_dice"]),
                "delta_aji": float(sfrm_obj["delta_aji"]),
                "delta_pq": float(sfrm_obj["delta_pq"]),
                "delta_confounder_fpr": float(sfrm_obj["delta_confounder_fpr"]),
            }
        )

        audit_rows.append(
            {
                "dataset": cfg.dataset,
                "selection_rule": "best validation boundary Dice",
                "selected_epoch": sfrm_epoch,
                "selected_refined_dice": _metric(sfrm_obj, "refined_dice", "dice"),
                "selected_refined_boundary_dice": _metric(sfrm_obj, "refined_boundary_dice", "boundary_dice"),
                "selected_refined_aji": _metric(sfrm_obj, "refined_aji", "aji"),
                "selected_refined_pq": _metric(sfrm_obj, "refined_pq", "pq"),
                "selected_refined_confounder_fpr": _metric(sfrm_obj, "refined_confounder_fpr", "confounder_fpr"),
                "peak_aji_epoch": peak_aji_epoch,
                "peak_aji_refined_dice": _metric(peak_aji_obj, "refined_dice", "dice"),
                "peak_aji_refined_boundary_dice": _metric(peak_aji_obj, "refined_boundary_dice", "boundary_dice"),
                "peak_aji_refined_aji": _metric(peak_aji_obj, "refined_aji", "aji"),
                "peak_aji_refined_pq": _metric(peak_aji_obj, "refined_pq", "pq"),
                "peak_aji_refined_confounder_fpr": _metric(peak_aji_obj, "refined_confounder_fpr", "confounder_fpr"),
            }
        )
    return summary_rows, audit_rows


def build_monuseg_fixed_budget_summary() -> list[dict]:
    rows: list[dict] = []
    for label, (run_name, epoch, refined) in MONUSEG_FIXED_BUDGET.items():
        obj = _load_json(_summary_path(RUN_DIR / run_name, epoch))
        rows.append(
            {
                "model": label,
                "run": run_name,
                "epoch": epoch,
                "selection_rule": "fixed endpoint for cue comparison" if refined else "baseline best validation boundary Dice checkpoint",
                "dice": _metric(obj, "refined_dice", "dice"),
                "boundary_dice": _metric(obj, "refined_boundary_dice", "boundary_dice"),
                "aji": _metric(obj, "refined_aji", "aji"),
                "pq": _metric(obj, "refined_pq", "pq"),
                "confounder_fpr": _metric(obj, "refined_confounder_fpr", "confounder_fpr"),
            }
        )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_checkpoint_audit(path: Path, audit_rows: list[dict], monuseg_rows: list[dict]) -> None:
    lines = [
        "# Boundary-SFRM-UNet checkpoint selection audit",
        "",
        "## Main evidence datasets",
        "",
        "Selection rule: use the checkpoint with the best validation boundary Dice for both the baseline U-Net and the frozen-coarse Boundary-SFRM-UNet.",
        "Peak-AJI checkpoints are listed only as audit context and are not used as the primary reported model if they violate the boundary-based deployment rule.",
        "",
    ]
    for row in audit_rows:
        lines.extend(
            [
                f"### {row['dataset']}",
                f"- selected epoch: {row['selected_epoch']}",
                f"- selected metrics: Dice={row['selected_refined_dice']:.4f}, Boundary Dice={row['selected_refined_boundary_dice']:.4f}, AJI={row['selected_refined_aji']:.4f}, PQ={row['selected_refined_pq']:.4f}, Confounder FPR={row['selected_refined_confounder_fpr']:.4f}",
                f"- peak AJI epoch: {row['peak_aji_epoch']}",
                f"- peak AJI metrics: Dice={row['peak_aji_refined_dice']:.4f}, Boundary Dice={row['peak_aji_refined_boundary_dice']:.4f}, AJI={row['peak_aji_refined_aji']:.4f}, PQ={row['peak_aji_refined_pq']:.4f}, Confounder FPR={row['peak_aji_refined_confounder_fpr']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## MoNuSeg supporting comparison",
            "",
            "Selection rule: compare refinement variants at a common training-budget endpoint (epoch 8) to isolate cue quality under identical optimization budget. The baseline row uses the best validation boundary Dice checkpoint because it has no refinement branch.",
            "",
            "| Model | Epoch | Dice | Boundary Dice | AJI | PQ | Confounder FPR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in monuseg_rows:
        lines.append(
            f"| {row['model']} | {row['epoch']} | {row['dice']:.4f} | {row['boundary_dice']:.4f} | {row['aji']:.4f} | {row['pq']:.4f} | {row['confounder_fpr']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_summary, audit_rows = build_main_summary()
    monuseg_rows = build_monuseg_fixed_budget_summary()
    _write_json(OUT_DIR / "feedback_refinement_summary.json", main_summary)
    _write_json(OUT_DIR / "checkpoint_selection_audit.json", audit_rows)
    _write_json(OUT_DIR / "monuseg_cue_comparison_fixed_epoch8.json", monuseg_rows)
    _write_checkpoint_audit(OUT_DIR / "checkpoint_selection_audit.md", audit_rows, monuseg_rows)


if __name__ == "__main__":
    main()
