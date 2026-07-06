from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "paper1" / "table_visuals"
WORD_DRAFT = ROOT / "manuscript" / "word_draft" / "build_bspc_word.py"


def _load_word_tables():
    spec = importlib.util.spec_from_file_location("build_bspc_word", WORD_DRAFT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TABLES, module.CHANNEL_MASKING


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _to_float(value: str) -> float:
    return float(str(value).replace(",", "").replace("GFLOPs", "").replace("ms", "").strip())


def figure_table1(table: dict) -> None:
    labels = [row[0] for row in table["rows"]]
    metrics = ["Dice", "Boundary Dice", "AJI", "PQ", "Confounder FPR"]
    cols = [2, 3, 4, 5, 6]
    data = np.array([[float(row[i]) for i in cols] for row in table["rows"]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), gridspec_kw={"width_ratios": [4.3, 1.2]})
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.14
    colors = ["#1d4ed8", "#7c3aed", "#0891b2", "#16a34a", "#dc2626"]
    ax = axes[0]
    for j, metric in enumerate(metrics):
        ax.bar(x + (j - 2) * width, data[:, j], width=width, label=metric, color=colors[j], edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Two-pass", "Entropy", "Failure head", "Calibrated"], rotation=15, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Table 1 -> CryoNuSeg main results", fontweight="bold")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax2 = axes[1]
    deltas = data[:, 1] - data[0, 1]
    bars = ax2.bar(np.arange(len(labels)), deltas, color="#7c3aed", edgecolor="black", linewidth=0.7)
    bars[0].set_color("#94a3b8")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(["B", "TP", "E", "FH", "C"])
    ax2.set_ylabel("Delta boundary Dice")
    ax2.set_title("Boundary focus", fontsize=11)
    ax2.grid(axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.8)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    _save(fig, "table1_cryonuseg_main")


def figure_table2(table: dict) -> None:
    metrics = [row[0] for row in table["rows"]]
    deltas = np.array([float(row[3]) for row in table["rows"]], dtype=float)
    ci_text = [row[4] for row in table["rows"]]
    lower = np.array([abs(float(txt.split(",")[0].replace("[", "")) - delta) for txt, delta in zip(ci_text, deltas)], dtype=float)
    upper = np.array([abs(float(txt.split(",")[1].replace("]", "")) - delta) for txt, delta in zip(ci_text, deltas)], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.patch.set_facecolor("white")
    y = np.arange(len(metrics))
    colors = ["#2563eb", "#7c3aed", "#0891b2", "#16a34a", "#dc2626"]
    ax.barh(y, deltas, xerr=np.vstack([lower, upper]), color=colors, edgecolor="black", linewidth=0.8, capsize=4)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Refined - baseline")
    ax.set_title("Table 2 -> Independent CryoNuSeg delta with 95% CI", fontweight="bold")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save(fig, "table2_holdout_delta_ci")


def figure_table3(table: dict) -> None:
    labels = [row[0] for row in table["rows"]]
    metrics = ["Dice", "Boundary Dice", "AJI", "PQ", "Confounder FPR"]
    cols = [1, 2, 3, 4, 5]
    data = np.array([[float(row[i]) for i in cols] for row in table["rows"]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), gridspec_kw={"width_ratios": [4.3, 1.2]})
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    width = 0.16
    colors = ["#1d4ed8", "#7c3aed", "#0891b2", "#16a34a", "#dc2626"]
    ax = axes[0]
    for j, metric in enumerate(metrics):
        ax.bar(x + (j - 2) * width, data[:, j], width=width, label=metric, color=colors[j], edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Two-pass", "Entropy", "Failure head"], rotation=12, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Table 3 -> CoNSeP stress test", fontweight="bold")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax2 = axes[1]
    fpr_delta = data[:, 4] - data[0, 4]
    ax2.bar(np.arange(len(labels)), fpr_delta, color="#dc2626", edgecolor="black", linewidth=0.7)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(["B", "TP", "E", "FH"])
    ax2.set_ylabel("Delta FPR")
    ax2.set_title("Leakage cost", fontsize=11)
    ax2.grid(axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.8)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    _save(fig, "table3_consep_stress")


def figure_table4(table: dict) -> None:
    labels = [row[0] for row in table["rows"]]
    metrics = ["Dice", "Boundary Dice", "AJI", "PQ", "Confounder FPR"]
    cols = [1, 2, 3, 4, 5]
    data = np.array([[float(row[i]) for i in cols] for row in table["rows"]], dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.4))
    fig.patch.set_facecolor("white")
    metric_map = {
        "Boundary Dice": 1,
        "AJI": 2,
        "PQ": 3,
        "Confounder FPR": 4,
    }
    colors = {
        "Boundary Dice": "#2563eb",
        "AJI": "#7c3aed",
        "PQ": "#0891b2",
        "Confounder FPR": "#dc2626",
    }
    x = np.arange(len(labels))
    short_labels = ["Base", "2-pass", "Entropy", "B-SFRM", "L-FH"]
    for ax, (metric, idx) in zip(axes, metric_map.items(), strict=True):
        vals = data[:, idx]
        bars = ax.bar(x, vals, color=colors[metric], edgecolor="black", linewidth=0.7)
        best_idx = int(np.argmin(vals) if metric == "Confounder FPR" else np.argmax(vals))
        bars[best_idx].set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Table 4 -> MoNuSeg cue comparison", fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "table4_monuseg_cues")


def figure_table5(channel_masking: dict) -> None:
    rows = channel_masking["rows"]
    labels = [f"{row[0]}\n{row[1]}" for row in rows]
    metrics = ["Delta Boundary Dice", "Delta AJI", "Delta PQ", "Delta Confounder FPR"]
    data = np.array([[float(row[i]) for i in range(2, 6)] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", vmin=-0.1, vmax=0.1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Table 5 -> Channel masking ablation heatmap", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta vs unmasked checkpoint")
    fig.tight_layout()
    _save(fig, "table5_channel_masking")


def figure_table6(table: dict) -> None:
    labels = [row[0] for row in table["rows"]]
    metrics = ["Delta Dice", "Delta Boundary", "Delta AJI", "Delta PQ", "Delta Conf. FPR"]
    data = np.array([[float(row[i]) for i in range(1, 6)] for row in table["rows"]], dtype=float)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", vmin=-0.35, vmax=0.2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Table 6 -> Stability and calibration ablation", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta over baseline")
    fig.tight_layout()
    _save(fig, "table6_ablation_heatmap")


def figure_table7(table: dict) -> None:
    labels = [row[0] for row in table["rows"]]
    params = np.array([_to_float(row[1]) / 1e6 for row in table["rows"]], dtype=float)
    flops = np.array([_to_float(row[2]) for row in table["rows"]], dtype=float)
    latency = np.array([_to_float(row[3]) for row in table["rows"]], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2))
    fig.patch.set_facecolor("white")
    series = [
        ("Parameters (M)", params, "#2563eb"),
        ("FLOPs", flops, "#7c3aed"),
        ("Inference time (ms)", latency, "#16a34a"),
    ]
    short_labels = ["Baseline", "Boundary-SFRM"]
    for ax, (title, vals, color) in zip(axes, series, strict=True):
        bars = ax.bar(np.arange(len(labels)), vals, color=color, edgecolor="black", linewidth=0.8, width=0.62)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(short_labels, rotation=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for idx, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{vals[idx]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Table 7 -> Computational overhead", fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "table7_compute_overhead")


def main() -> None:
    tables, channel_masking = _load_word_tables()
    figure_table1(tables["Table 1 shows"])
    figure_table2(tables["Table 2 shows"])
    figure_table3(tables["CoNSeP defines the current operating boundary."])
    figure_table4(tables["Table 4 provides"])
    figure_table5(channel_masking)
    figure_table6(tables["Table 6 consolidates"])
    figure_table7(tables["Table 7 compares"])


if __name__ == "__main__":
    main()
