from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .mining import MiningResult


def _resize_map(array: np.ndarray, image: Image.Image) -> np.ndarray:
    pil = Image.fromarray(array)
    pil = pil.resize(image.size, Image.NEAREST)
    return np.asarray(pil)


def render_mining_diagnostics(
    image: Image.Image,
    mask: np.ndarray,
    points_yx: np.ndarray,
    result: MiningResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image.convert("RGB"))
    positive_map = _resize_map(result.positive_seed_map.astype(np.uint8) * 255, image)
    negative_map = _resize_map(
        result.negative_candidate_map.astype(np.uint8) * 255,
        image,
    )
    similarity = result.similarity_map
    sim_min = float(similarity.min())
    sim_max = float(similarity.max())
    sim_norm = (similarity - sim_min) / max(sim_max - sim_min, 1e-6)
    sim_resized = _resize_map((sim_norm * 255).astype(np.uint8), image)
    labels = result.prototype_labels.astype(np.int32)
    label_vis = labels.copy()
    label_vis[label_vis < 0] = 0
    label_vis = _resize_map(label_vis.astype(np.uint8), image)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].imshow(rgb)
    axes[0].scatter(points_yx[:, 1], points_yx[:, 0], c="yellow", s=18)
    axes[0].set_title("Image + simulated points")

    axes[1].imshow(mask > 0, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(rgb)
    axes[2].imshow(positive_map, cmap="Greens", alpha=0.45)
    axes[2].set_title("Positive seed regions")

    axes[3].imshow(rgb)
    axes[3].imshow(negative_map, cmap="Reds", alpha=0.55)
    axes[3].set_title("Annular hard negatives")

    axes[4].imshow(sim_resized, cmap="magma")
    axes[4].set_title("Max positive similarity")

    axes[5].imshow(label_vis, cmap="tab10", vmin=0, vmax=10)
    axes[5].set_title("Negative prototype clusters")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

