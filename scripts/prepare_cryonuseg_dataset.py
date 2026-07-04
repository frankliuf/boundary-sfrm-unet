from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CryoNuSeg into a flat nuclei dataset layout.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to extracted CryoNuSeg root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--annotator-dirname",
        type=str,
        default="Annotator 1 (biologist second round of manual marks up)",
        help="Annotator folder to use as ground truth.",
    )
    return parser.parse_args()


def infer_source(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    return "_".join(parts[:-1])


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    image_dir = input_dir / "tissue images"
    annotator_root = input_dir / args.annotator_dirname
    if not annotator_root.exists():
        raise FileNotFoundError(f"Missing annotator directory: {annotator_root}")
    label_dir = annotator_root / "label masks"
    if not label_dir.exists():
        nested_root = annotator_root / args.annotator_dirname
        nested_label_dir = nested_root / "label masks"
        if nested_label_dir.exists():
            annotator_root = nested_root
            label_dir = nested_label_dir

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing tissue image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label mask directory: {label_dir}")

    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for image_path in sorted(image_dir.glob("*.tif")):
        stem = image_path.stem
        mask_path = label_dir / f"{stem}.tif"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing label mask for {stem}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        mask_np = np.array(Image.open(mask_path))
        if mask_np.ndim != 2:
            raise ValueError(f"Expected 2D instance mask for {mask_path}, got shape {mask_np.shape}")
        if mask_np.dtype != np.uint16:
            mask_np = mask_np.astype(np.uint16)

        out_image_path = out_images / f"{stem}.png"
        out_mask_path = out_masks / f"{stem}.png"
        Image.fromarray(image_np).save(out_image_path)
        Image.fromarray(mask_np).save(out_mask_path)

        positive = mask_np > 0
        rows.append(
            {
                "id": stem,
                "source": infer_source(stem),
                "original_image": image_path.name,
                "height": int(mask_np.shape[0]),
                "width": int(mask_np.shape[1]),
                "instances": int(len(np.unique(mask_np[positive]))),
                "positive_fraction": float(positive.mean()),
            }
        )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "source", "original_image", "height", "width", "instances", "positive_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} CryoNuSeg samples at {output_dir}")


if __name__ == "__main__":
    main()
