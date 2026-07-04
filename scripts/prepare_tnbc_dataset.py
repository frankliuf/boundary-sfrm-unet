from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TNBC nuclei dataset into flat image/mask pairs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_images = args.output_dir / "images"
    output_masks = args.output_dir / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    slide_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir() and path.name.startswith("Slide_"))
    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        gt_dir = args.input_dir / slide_id.replace("Slide_", "GT_")
        if not gt_dir.exists():
            raise FileNotFoundError(f"Missing GT directory for {slide_id}: {gt_dir}")

        for image_path in sorted(slide_dir.glob("*.png")):
            mask_path = gt_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

            stem = f"{slide_id}_{image_path.stem}"
            image = Image.open(image_path).convert("RGB")
            binary_mask = np.asarray(Image.open(mask_path))
            if binary_mask.ndim == 3:
                binary_mask = binary_mask[..., 0]
            instance_mask = connected_instance_mask(binary_mask > 0)

            image.save(output_images / f"{stem}.png")
            Image.fromarray(instance_mask.astype(np.uint16)).save(output_masks / f"{stem}.png")
            rows.append(
                {
                    "id": stem,
                    "source": slide_id,
                    "original_image": image_path.name,
                    "height": str(image.height),
                    "width": str(image.width),
                    "instances": str(int(instance_mask.max())),
                    "positive_fraction": f"{float((instance_mask > 0).mean()):.6f}",
                }
            )

    metadata_path = args.output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "source", "original_image", "height", "width", "instances", "positive_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} TNBC image/mask pairs to {args.output_dir}")
    print(f"metadata: {metadata_path}")


def connected_instance_mask(binary_mask: np.ndarray) -> np.ndarray:
    labeled, count = ndi.label(binary_mask.astype(bool))
    if count == 0:
        return np.zeros(binary_mask.shape, dtype=np.uint16)
    if count > np.iinfo(np.uint16).max:
        raise ValueError(f"Too many connected components for uint16 mask: {count}")
    return labeled.astype(np.uint16)


if __name__ == "__main__":
    main()
