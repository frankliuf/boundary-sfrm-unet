from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a standardized CoNSeP layout for Boundary-SFRM-UNet."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Project data root containing existing CoNSeP patch folders.",
    )
    return parser.parse_args()


SPLITS = {
    "train": {
        "full": "consep_train_grid_patches",
        "confounders": "consep_train_grid_confounders_distance",
    },
    "val": {
        "full": "consep_val_grid_patches",
        "confounders": "consep_val_grid_confounders_distance",
    },
    "test": {
        "full": "consep_test_grid_patches",
        "confounders": "consep_test_grid_confounders_distance",
    },
}


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def load_patch_ids(metadata_path: Path) -> set[str]:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"] for row in csv.DictReader(handle)}


def verify_split(full_dir: Path, conf_dir: Path) -> dict[str, int]:
    image_dir = full_dir / "images"
    mask_dir = full_dir / "masks"
    metadata_path = full_dir / "metadata.csv"
    conf_metadata_path = conf_dir / "metadata.csv"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing image/mask directories under {full_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv under {full_dir}")
    if not conf_metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv under {conf_dir}")

    image_count = sum(1 for _ in image_dir.glob("*.png"))
    mask_count = sum(1 for _ in mask_dir.glob("*.png"))
    conf_count = sum(1 for _ in conf_dir.glob("*.png"))

    if image_count != mask_count:
        raise ValueError(f"Image/mask count mismatch in {full_dir}: {image_count} vs {mask_count}")
    if image_count != conf_count:
        raise ValueError(f"Image/confounder count mismatch in {full_dir}: {image_count} vs {conf_count}")

    patch_ids = load_patch_ids(metadata_path)
    conf_ids = load_patch_ids(conf_metadata_path)
    if patch_ids != conf_ids:
        raise ValueError(f"Patch ID mismatch between {metadata_path} and {conf_metadata_path}")

    return {
        "images": image_count,
        "masks": mask_count,
        "confounders": conf_count,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    rows: list[tuple[str, dict[str, int]]] = []
    for split, cfg in SPLITS.items():
        src_full = data_root / cfg["full"]
        src_conf = data_root / cfg["confounders"]
        dst_full = data_root / f"consep_{split}_split_full"
        dst_conf = data_root / f"consep_{split}_split_confounders"

        copy_tree_contents(src_full, dst_full)
        copy_tree_contents(src_conf, dst_conf)
        rows.append((split, verify_split(dst_full, dst_conf)))

    print("Prepared CoNSeP standardized splits:")
    for split, counts in rows:
        print(
            f"- {split}: images={counts['images']}, masks={counts['masks']}, confounders={counts['confounders']}"
        )


if __name__ == "__main__":
    main()
