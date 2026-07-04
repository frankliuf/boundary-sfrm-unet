from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_synthetic_nuclei_scene(
    size: int = 256,
    seed: int = 13,
) -> tuple[Image.Image, np.ndarray]:
    """Create a synthetic pathology-like dense nuclei scene.

    The mask contains only target nuclei. Extra stain-similar distractors are
    drawn into the image but not included in the mask, creating confounders.
    """
    rng = np.random.default_rng(seed)
    base = Image.new("RGB", (size, size), (226, 196, 218))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(base, "RGBA")
    mask_draw = ImageDraw.Draw(mask)

    centers = [
        (80, 72),
        (107, 78),
        (132, 105),
        (86, 126),
        (160, 150),
        (184, 154),
        (146, 178),
        (68, 178),
    ]
    for idx, (cx, cy) in enumerate(centers, start=1):
        rx = int(rng.integers(10, 16))
        ry = int(rng.integers(8, 14))
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        color = (
            int(rng.integers(82, 112)),
            int(rng.integers(42, 68)),
            int(rng.integers(120, 164)),
            225,
        )
        draw.ellipse(bbox, fill=color)
        mask_draw.ellipse(bbox, fill=idx)

    distractors = [(118, 128), (103, 151), (172, 122), (64, 93), (201, 180)]
    for cx, cy in distractors:
        rx = int(rng.integers(8, 13))
        ry = int(rng.integers(7, 12))
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        draw.ellipse(bbox, fill=(112, 65, 142, 180))

    noise = rng.normal(0, 9, (size, size, 3)).astype(np.int16)
    image_array = np.asarray(base).astype(np.int16)
    image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array).filter(ImageFilter.GaussianBlur(0.4))
    return image, np.asarray(mask, dtype=np.uint8)


def save_synthetic_scene(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image, mask = make_synthetic_nuclei_scene()
    image_path = output_dir / "synthetic_nuclei.png"
    mask_path = output_dir / "synthetic_nuclei_mask.png"
    image.save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path

