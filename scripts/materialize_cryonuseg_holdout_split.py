from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_SPLITS = {
    "train": [
        "Human_LymphNodes",
        "Human_Mediastinum",
        "Human_Pleura",
        "Human_Skin",
        "Human_Thymus",
    ],
    "val": [
        "Human_AdrenalGland",
        "Human_Larynx",
        "Human_Pancreas",
    ],
    "test": [
        "Human_Testes",
        "Human_ThyroidGland",
    ],
}


def parse_source_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty source list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a source-level CryoNuSeg train/val/test split.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Workspace data root containing cryonuseg_full and existing confounder maps.",
    )
    parser.add_argument("--prefix", type=str, default="cryonuseg_holdout")
    parser.add_argument("--train-sources", type=parse_source_list, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val-sources", type=parse_source_list, default=DEFAULT_SPLITS["val"])
    parser.add_argument("--test-sources", type=parse_source_list, default=DEFAULT_SPLITS["test"])
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_confounder_lookup(data_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for split in ("train", "val"):
        conf_dir = data_root / f"cryonuseg_{split}_split_confounders"
        for path in conf_dir.glob("*.png"):
            lookup[path.stem] = path
    if not lookup:
        raise FileNotFoundError("No existing CryoNuSeg confounder maps were found.")
    return lookup


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(
    rows: list[dict[str, str]],
    full_root: Path,
    conf_lookup: dict[str, Path],
    dst_full: Path,
    dst_conf: Path,
) -> None:
    images_dir = dst_full / "images"
    masks_dir = dst_full / "masks"
    ensure_clean_dir(images_dir)
    ensure_clean_dir(masks_dir)
    ensure_clean_dir(dst_conf)

    conf_rows: list[dict[str, str]] = []
    for row in rows:
        sample_id = row["id"]
        image_src = full_root / "images" / f"{sample_id}.png"
        mask_src = full_root / "masks" / f"{sample_id}.png"
        conf_src = conf_lookup.get(sample_id)
        if not image_src.exists() or not mask_src.exists():
            raise FileNotFoundError(f"Missing CryoNuSeg image/mask for {sample_id}")
        if conf_src is None or not conf_src.exists():
            raise FileNotFoundError(f"Missing CryoNuSeg confounder map for {sample_id}")

        shutil.copy2(image_src, images_dir / image_src.name)
        shutil.copy2(mask_src, masks_dir / mask_src.name)
        shutil.copy2(conf_src, dst_conf / conf_src.name)
        conf_rows.append(
            {
                "id": sample_id,
                "source": row["source"],
                "map": str((dst_conf / conf_src.name).relative_to(dst_conf.parent.parent)),
            }
        )

    write_metadata(dst_full / "metadata.csv", rows)
    write_metadata(dst_conf / "metadata.csv", conf_rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    full_root = data_root / "cryonuseg_full"
    metadata_path = full_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing CryoNuSeg metadata: {metadata_path}")

    rows = load_metadata(metadata_path)
    selected_splits = {
        "train": args.train_sources,
        "val": args.val_sources,
        "test": args.test_sources,
    }

    all_sources = {row["source"] for row in rows}
    declared_sources = set().union(*selected_splits.values())
    if all_sources != declared_sources:
        missing = sorted(all_sources - declared_sources)
        extra = sorted(declared_sources - all_sources)
        raise ValueError(f"Split declaration mismatch. Missing={missing}, extra={extra}")

    conf_lookup = build_confounder_lookup(data_root)
    for split_name, sources in selected_splits.items():
        split_rows = [row for row in rows if row["source"] in sources]
        split_ids = {row["id"] for row in split_rows}
        if len(split_ids) != len(split_rows):
            raise ValueError(f"Duplicate IDs found in {split_name} split.")
        dst_full = data_root / f"{args.prefix}_{split_name}_split_full"
        dst_conf = data_root / f"{args.prefix}_{split_name}_split_confounders"
        copy_split(split_rows, full_root, conf_lookup, dst_full, dst_conf)
        print(
            f"{split_name}: sources={sources}, images={len(split_rows)}, "
            f"dst_full={dst_full.name}, dst_conf={dst_conf.name}"
        )


if __name__ == "__main__":
    main()
