from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "confounder_mining").is_dir():
    sys.path.insert(0, str((ROOT.parent / "confounder_prompting").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from confounder_mining.dataset import NucleiPointDataset  # noqa: E402
from confounder_mining.unet import SmallUNet  # noqa: E402
from src.models import BoundarySFRMUNet  # noqa: E402


FIG_DIR = ROOT / "figures" / "paper1"
RUN_DIR = ROOT / "experiments" / "boundary_sfrm_runs"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    val_images: Path
    val_masks: Path
    val_confounders: Path
    baseline_ckpt: Path
    sfrm_ckpt: Path
    per_sample_csv: Path
    baseline_epoch: int
    sfrm_epoch: int


DATASETS = {
    "TNBC": DatasetConfig(
        name="TNBC",
        val_images=ROOT.parent / "data" / "tnbc_val_split_full" / "images",
        val_masks=ROOT.parent / "data" / "tnbc_val_split_full" / "masks",
        val_confounders=ROOT.parent / "data" / "tnbc_val_split_confounders",
        baseline_ckpt=RUN_DIR / "tnbc_fullsup_unet_seed42" / "best_boundary_model.pt",
        sfrm_ckpt=RUN_DIR / "tnbc_sfrm_unet_learned_failure_head_freeze8_seed42" / "best_boundary_model.pt",
        per_sample_csv=RUN_DIR / "tnbc_sfrm_unet_learned_failure_head_freeze8_seed42" / "val_epoch_05_per_sample.csv",
        baseline_epoch=7,
        sfrm_epoch=5,
    ),
    "CryoNuSeg": DatasetConfig(
        name="CryoNuSeg",
        val_images=ROOT.parent / "data" / "cryonuseg_val_split_full" / "images",
        val_masks=ROOT.parent / "data" / "cryonuseg_val_split_full" / "masks",
        val_confounders=ROOT.parent / "data" / "cryonuseg_val_split_confounders",
        baseline_ckpt=RUN_DIR / "cryonuseg_fullsup_unet_seed42" / "best_boundary_model.pt",
        sfrm_ckpt=RUN_DIR / "cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42" / "best_boundary_model.pt",
        per_sample_csv=RUN_DIR / "cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42" / "val_epoch_08_per_sample.csv",
        baseline_epoch=7,
        sfrm_epoch=8,
    ),
    "CoNSeP": DatasetConfig(
        name="CoNSeP",
        val_images=ROOT.parent / "data" / "consep_val_split_full" / "images",
        val_masks=ROOT.parent / "data" / "consep_val_split_full" / "masks",
        val_confounders=ROOT.parent / "data" / "consep_val_split_confounders",
        baseline_ckpt=RUN_DIR / "consep_fullsup_unet_seed42" / "best_boundary_model.pt",
        sfrm_ckpt=RUN_DIR / "consep_fullsup_learned_failure_head_v3_teacher10_seed42" / "best_boundary_model.pt",
        per_sample_csv=RUN_DIR / "consep_fullsup_learned_failure_head_v3_teacher10_seed42" / "val_epoch_06_per_sample.csv",
        baseline_epoch=8,
        sfrm_epoch=6,
    ),
}


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _binary_entropy(prob: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _normalize_map(arr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    return (arr - lo) / (hi - lo + eps)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (1.0 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_contours(ax, mask: np.ndarray, color: str, lw: float = 1.4) -> None:
    for contour in measure.find_contours(mask.astype(np.uint8), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=lw)


def _pick_case(rows: list[dict[str, str]]) -> dict[str, str]:
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        dd = float(row["delta_dice"])
        db = float(row["delta_boundary_dice"])
        da = float(row["delta_aji"])
        dp = float(row["delta_pq"])
        df = float(row["delta_confounder_fpr"])
        score = db + da + dp - 0.25 * max(0.0, -dd) - df
        if db > 0.0 and da > 0.0 and dp > 0.0 and df < 0.0:
            scored.append((score, row))
    if not scored:
        raise RuntimeError("No suitable qualitative case found.")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _pick_case_relaxed(rows: list[dict[str, str]]) -> dict[str, str]:
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        dd = float(row["delta_dice"])
        db = float(row["delta_boundary_dice"])
        da = float(row["delta_aji"])
        dp = float(row["delta_pq"])
        df = float(row["delta_confounder_fpr"])
        score = db + da + dp - 0.15 * max(0.0, -dd) - 0.25 * max(0.0, df)
        if db > 0.0 and da > 0.0 and dp >= 0.0:
            scored.append((score, row))
    if not scored:
        raise RuntimeError("No suitable relaxed qualitative case found.")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _pick_failure_case(rows: list[dict[str, str]], exclude_ids: set[str] | None = None) -> dict[str, str]:
    exclude_ids = exclude_ids or set()
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        if row["patch_id"] in exclude_ids:
            continue
        dd = float(row["delta_dice"])
        db = float(row["delta_boundary_dice"])
        da = float(row["delta_aji"])
        dp = float(row["delta_pq"])
        df = float(row["delta_confounder_fpr"])
        if db < 0.0 and df > 0.0:
            score = abs(db) + abs(min(0.0, da)) + abs(min(0.0, dp)) + df + 0.25 * abs(min(0.0, dd))
            scored.append((score, row))
    if not scored:
        raise RuntimeError("No suitable failure qualitative case found.")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _pick_leakage_failure_case(rows: list[dict[str, str]], exclude_ids: set[str] | None = None) -> dict[str, str]:
    exclude_ids = exclude_ids or set()
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        if row["patch_id"] in exclude_ids:
            continue
        dd = float(row["delta_dice"])
        db = float(row["delta_boundary_dice"])
        da = float(row["delta_aji"])
        dp = float(row["delta_pq"])
        df = float(row["delta_confounder_fpr"])
        if df > 0.06 and (db < 0.02 or da < 0.0 or dp < 0.0):
            score = df + 0.35 * abs(min(0.0, db)) + 0.25 * abs(min(0.0, da)) + 0.25 * abs(min(0.0, dp)) + 0.15 * abs(min(0.0, dd))
            scored.append((score, row))
    if not scored:
        raise RuntimeError("No suitable leakage-oriented qualitative case found.")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _load_models(cfg: DatasetConfig, device: torch.device) -> tuple[SmallUNet, BoundarySFRMUNet]:
    baseline_ckpt = torch.load(cfg.baseline_ckpt, map_location=device, weights_only=False)
    sfrm_ckpt = torch.load(cfg.sfrm_ckpt, map_location=device, weights_only=False)

    baseline = SmallUNet(base_channels=24).to(device)
    baseline.load_state_dict(baseline_ckpt["model"])
    baseline.eval()

    model = BoundarySFRMUNet(base_channels=24, risk_channels=4).to(device)
    model.load_state_dict(sfrm_ckpt["model"])
    model.eval()
    return baseline, model


def _get_dataset_sample(cfg: DatasetConfig, patch_id: str) -> dict[str, torch.Tensor | str]:
    ds = NucleiPointDataset(
        cfg.val_images,
        cfg.val_masks,
        confounders_dir=cfg.val_confounders,
        augment=False,
    )
    for idx in range(len(ds)):
        sample = ds[idx]
        if sample["id"] == patch_id:
            return sample
    raise KeyError(f"Patch id not found: {patch_id}")


def _predict_case(cfg: DatasetConfig, patch_id: str, device: torch.device) -> dict[str, np.ndarray]:
    sample = _get_dataset_sample(cfg, patch_id)
    image = sample["image"].unsqueeze(0).to(device)
    gt = sample["mask"].numpy()[0].astype(bool)
    conf = sample["confounder"].numpy()[0].astype(bool)

    baseline, model = _load_models(cfg, device)
    with torch.no_grad():
        baseline_logits = baseline(image)
        baseline_prob = torch.sigmoid(baseline_logits)
        baseline_mask = baseline_prob[0, 0].cpu().numpy() >= 0.5

        coarse_state = model.forward_coarse(image)
        coarse_prob = torch.sigmoid(coarse_state.coarse_logits)
        failure_prob = torch.sigmoid(coarse_state.failure_logits)
        entropy = _binary_entropy(coarse_prob)
        entropy = entropy / entropy.amax(dim=(2, 3), keepdim=True).clamp_min(1e-7)
        risk_tensor = torch.cat([coarse_prob, entropy, failure_prob], dim=1)
        refined_logits = model.decode_refine(coarse_state, risk_tensor)
        refined_prob = torch.sigmoid(refined_logits)
        refined_mask = refined_prob[0, 0].cpu().numpy() >= 0.5
        risk = failure_prob.max(dim=1).values[0].cpu().numpy()

    rgb = (sample["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return {
        "rgb": rgb,
        "gt": gt,
        "conf": conf,
        "baseline_mask": baseline_mask,
        "refined_mask": refined_mask,
        "risk": _normalize_map(risk),
    }


def figure_qualitative() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selections: list[tuple[DatasetConfig, str, dict[str, str], dict[str, np.ndarray]]] = []
    used_patch_ids: set[str] = set()
    selection_plan = [
        (DATASETS["TNBC"], "repair"),
        (DATASETS["CryoNuSeg"], "repair"),
        (DATASETS["CoNSeP"], "repair"),
        (DATASETS["CoNSeP"], "failure"),
        (DATASETS["CoNSeP"], "leakage"),
    ]
    for cfg, mode in selection_plan:
        rows = _load_csv_rows(cfg.per_sample_csv)
        if mode == "failure":
            row = _pick_failure_case(rows, exclude_ids=used_patch_ids)
        elif mode == "leakage":
            row = _pick_leakage_failure_case(rows, exclude_ids=used_patch_ids)
        elif cfg.name == "CoNSeP":
            row = _pick_case_relaxed(rows)
        else:
            row = _pick_case(rows)
        used_patch_ids.add(row["patch_id"])
        pred = _predict_case(cfg, row["patch_id"], device)
        selections.append((cfg, mode, row, pred))

    fig, axes = plt.subplots(len(selections), 6, figsize=(18, 15.2))
    fig.patch.set_facecolor("white")
    titles = ["Image + GT", "Baseline", "Raw failure risk", "Refined", "Baseline FP/FN", "Refined FP/FN"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=0.0, vmax=1.0)

    for i, (cfg, mode, row, pred) in enumerate(selections):
        rgb = pred["rgb"]
        gt = pred["gt"]
        base = pred["baseline_mask"]
        ref = pred["refined_mask"]
        risk = pred["risk"]

        fp_b = np.logical_and(base, ~gt)
        fn_b = np.logical_and(~base, gt)
        fp_r = np.logical_and(ref, ~gt)
        fn_r = np.logical_and(~ref, gt)

        # Col 0
        axes[i, 0].imshow(rgb)
        _draw_contours(axes[i, 0], gt, "#16A34A")

        # Col 1
        img1 = _overlay_mask(rgb, base, (187, 134, 252), 0.18)
        axes[i, 1].imshow(img1)
        _draw_contours(axes[i, 1], gt, "#16A34A")
        _draw_contours(axes[i, 1], base, "#7C3AED")

        # Col 2
        risk_rgb = (cmap(norm(risk))[..., :3] * 255.0).astype(np.uint8)
        blend = (0.45 * rgb.astype(np.float32) + 0.55 * risk_rgb.astype(np.float32)).astype(np.uint8)
        axes[i, 2].imshow(blend)
        _draw_contours(axes[i, 2], gt, "#FFFFFF", lw=1.0)

        # Col 3
        img3 = _overlay_mask(rgb, ref, (59, 130, 246), 0.18)
        axes[i, 3].imshow(img3)
        _draw_contours(axes[i, 3], gt, "#16A34A")
        _draw_contours(axes[i, 3], ref, "#2563EB")

        # Col 4
        err_b = _overlay_mask(rgb, fp_b, (220, 38, 38), 0.60)
        err_b = _overlay_mask(err_b, fn_b, (37, 99, 235), 0.60)
        axes[i, 4].imshow(err_b)

        # Col 5
        err_r = _overlay_mask(rgb, fp_r, (220, 38, 38), 0.60)
        err_r = _overlay_mask(err_r, fn_r, (37, 99, 235), 0.60)
        axes[i, 5].imshow(err_r)

        for j in range(6):
            axes[i, j].axis("off")

        summary = (
            f"{cfg.name} ({mode})  {row['patch_id']}   "
            f"$\\Delta$Dice={float(row['delta_dice']):+.3f}, "
            f"$\\Delta$Boundary={float(row['delta_boundary_dice']):+.3f}, "
            f"$\\Delta$AJI={float(row['delta_aji']):+.3f}, "
            f"$\\Delta$PQ={float(row['delta_pq']):+.3f}, "
            f"$\\Delta$FPR={float(row['delta_confounder_fpr']):+.3f}"
        )
        axes[i, 0].text(
            -0.02,
            -0.14,
            summary,
            transform=axes[i, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
        )

    fig.suptitle(
        "Boundary-SFRM-UNet qualitative panel: repair successes and dense-scene failure cases",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, "fig_boundary_sfrm_unet_qualitative_repair")

    output_rows = []
    for cfg, mode, row, _ in selections:
        output_rows.append(
            {
                "dataset": cfg.name,
                "mode": mode,
                "patch_id": row["patch_id"],
                "delta_dice": row["delta_dice"],
                "delta_boundary_dice": row["delta_boundary_dice"],
                "delta_aji": row["delta_aji"],
                "delta_pq": row["delta_pq"],
                "delta_confounder_fpr": row["delta_confounder_fpr"],
            }
        )
    with (FIG_DIR / "fig_boundary_sfrm_unet_qualitative_repair_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)


def figure_ablation() -> None:
    joint_hist = _load_csv_rows(RUN_DIR / "tnbc_sfrm_unet_learned_failure_head_seed42" / "history.csv")
    freeze_tnbc_hist = _load_csv_rows(RUN_DIR / "tnbc_sfrm_unet_learned_failure_head_freeze8_seed42" / "history.csv")
    freeze_cryo_hist = _load_csv_rows(RUN_DIR / "cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42" / "history.csv")
    calibrated = _load_json(RUN_DIR / "smoke_cryonuseg_sfrm_calibrated_seed42" / "val_epoch_01_summary.json")

    epochs_joint = np.array([int(r["epoch"]) for r in joint_hist])
    joint_boundary = np.array([float(r["refined_boundary_dice"]) for r in joint_hist])
    joint_pq = np.array([float(r["refined_pq"]) for r in joint_hist])

    epochs_freeze_t = np.array([int(r["epoch"]) for r in freeze_tnbc_hist])
    freeze_t_boundary = np.array([float(r["refined_boundary_dice"]) for r in freeze_tnbc_hist])
    freeze_t_pq = np.array([float(r["refined_pq"]) for r in freeze_tnbc_hist])

    epochs_freeze_c = np.array([int(r["epoch"]) for r in freeze_cryo_hist])
    freeze_c_boundary = np.array([float(r["refined_boundary_dice"]) for r in freeze_cryo_hist])
    freeze_c_pq = np.array([float(r["refined_pq"]) for r in freeze_cryo_hist])

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))
    fig.patch.set_facecolor("white")

    # TNBC stability: boundary dice
    axes[0].plot(epochs_joint, joint_boundary, marker="o", color="#DC2626", linewidth=2.0, label="joint")
    axes[0].plot(epochs_freeze_t, freeze_t_boundary, marker="o", color="#2563EB", linewidth=2.0, label="frozen coarse")
    axes[0].set_title("TNBC boundary Dice", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Refined boundary Dice")
    axes[0].grid(axis="y", color="#D1D5DB", alpha=0.8)
    axes[0].legend(frameon=False, fontsize=10)

    # TNBC stability: PQ
    axes[1].plot(epochs_joint, joint_pq, marker="o", color="#DC2626", linewidth=2.0, label="joint")
    axes[1].plot(epochs_freeze_t, freeze_t_pq, marker="o", color="#2563EB", linewidth=2.0, label="frozen coarse")
    axes[1].set_title("TNBC PQ", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Refined PQ")
    axes[1].grid(axis="y", color="#D1D5DB", alpha=0.8)

    # Cryo: raw vs calibrated refinement
    labels = ["freeze8 final", "calibrated smoke"]
    vals = [
        [float(freeze_cryo_hist[-1]["delta_boundary_dice"]), float(freeze_cryo_hist[-1]["delta_aji"]), float(freeze_cryo_hist[-1]["delta_pq"])],
        [float(calibrated["delta_boundary_dice"]), float(calibrated["delta_aji"]), float(calibrated["delta_pq"])],
    ]
    x = np.arange(len(labels))
    width = 0.22
    metric_names = ["ΔBoundary", "ΔAJI", "ΔPQ"]
    metric_colors = ["#2563EB", "#7C3AED", "#0891B2"]
    for idx, (mname, color) in enumerate(zip(metric_names, metric_colors, strict=True)):
        axes[2].bar(x + (idx - 1) * width, [vals[0][idx], vals[1][idx]], width=width, color=color, edgecolor="black", linewidth=0.8, label=mname)
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=10)
    axes[2].set_title("CryoNuSeg calibration ablation", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Improvement over baseline")
    axes[2].grid(axis="y", color="#D1D5DB", alpha=0.8)
    axes[2].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.set_facecolor("white")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Ablation: frozen-coarse stabilization is necessary, while aggressive calibration harms refinement",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_boundary_sfrm_unet_ablation")


def main() -> None:
    figure_qualitative()
    figure_ablation()


if __name__ == "__main__":
    main()
