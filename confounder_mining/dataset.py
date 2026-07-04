from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .points import load_mask, simulate_point_annotations
from .supervision import (
    make_affinity_random_walk_pseudo_labels,
    make_point_supervision_labels,
    make_seeded_region_pseudo_labels,
    make_voronoi_kmeans_pseudo_labels,
)


@dataclass(frozen=True)
class PatchSample:
    image_path: Path
    mask_path: Path


def list_patch_samples(images_dir: Path, masks_dir: Path) -> list[PatchSample]:
    samples: list[PatchSample] = []
    for image_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / image_path.name
        if mask_path.exists():
            samples.append(PatchSample(image_path=image_path, mask_path=mask_path))
    if not samples:
        raise ValueError(f"No matching image/mask PNG pairs found in {images_dir} and {masks_dir}.")
    return samples


def image_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class NucleiPointDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        confounders_dir: Path | None = None,
        positive_radius: float = 5.0,
        background_inner_radius: float = 18.0,
        point_strategy: str = "centroid",
        supervision_mode: str = "point",
        seeded_region_radius: float = 10.0,
        augment: bool = False,
    ) -> None:
        if supervision_mode not in {"point", "seeded_region", "voronoi_kmeans", "affinity_rw"}:
            raise ValueError("supervision_mode must be 'point', 'seeded_region', 'voronoi_kmeans', or 'affinity_rw'.")
        self.samples = list_patch_samples(images_dir, masks_dir)
        self.confounders_dir = confounders_dir
        self.positive_radius = positive_radius
        self.background_inner_radius = background_inner_radius
        self.point_strategy = point_strategy
        self.supervision_mode = supervision_mode
        self.seeded_region_radius = seeded_region_radius
        self.augment = augment
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8)
        instance_mask, semantic_mask, sparse_labels, confounder_map = self._load_cached_arrays(index, image_array)

        image_tensor = image_to_float_tensor(image)
        instance_tensor = torch.from_numpy(instance_mask.astype(np.int32)).unsqueeze(0)
        mask_tensor = torch.from_numpy(semantic_mask).unsqueeze(0)
        sparse_tensor = torch.from_numpy(sparse_labels)
        confounder_tensor = torch.from_numpy(confounder_map).unsqueeze(0)

        if self.augment:
            (
                image_tensor,
                instance_tensor,
                mask_tensor,
                sparse_tensor,
                confounder_tensor,
            ) = _apply_deterministic_augment(
                image_tensor,
                instance_tensor,
                mask_tensor,
                sparse_tensor,
                confounder_tensor,
                index,
            )

        return {
            "image": image_tensor,
            "instance": instance_tensor,
            "mask": mask_tensor,
            "sparse": sparse_tensor,
            "confounder": confounder_tensor,
            "id": sample.image_path.stem,
        }

    def _load_cached_arrays(
        self,
        index: int,
        image_rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if index in self._cache:
            return self._cache[index]

        sample = self.samples[index]
        instance_mask = load_mask(str(sample.mask_path))
        semantic_mask = (instance_mask > 0).astype(np.float32)
        points_yx = simulate_point_annotations(instance_mask, point_strategy=self.point_strategy)
        if self.supervision_mode == "point":
            sparse_labels = make_point_supervision_labels(
                semantic_mask.shape,
                points_yx,
                positive_radius=self.positive_radius,
                background_inner_radius=self.background_inner_radius,
            ).astype(np.int64)
        elif self.supervision_mode == "seeded_region":
            sparse_labels = make_seeded_region_pseudo_labels(
                semantic_mask.shape,
                points_yx,
                seeded_region_radius=self.seeded_region_radius,
                background_inner_radius=self.background_inner_radius,
            ).astype(np.int64)
        elif self.supervision_mode == "voronoi_kmeans":
            sparse_labels = make_voronoi_kmeans_pseudo_labels(
                image_rgb,
                points_yx,
                positive_radius=self.positive_radius,
            ).astype(np.int64)
        else:
            sparse_labels = make_affinity_random_walk_pseudo_labels(
                image_rgb,
                points_yx,
                positive_radius=self.positive_radius,
                background_inner_radius=self.background_inner_radius,
            ).astype(np.int64)
        confounder_map = np.zeros_like(semantic_mask, dtype=np.float32)
        if self.confounders_dir is not None:
            confounder_path = self.confounders_dir / sample.image_path.name
            if confounder_path.exists():
                confounder_map = (np.asarray(Image.open(confounder_path)) > 0).astype(np.float32)

        self._cache[index] = (instance_mask, semantic_mask, sparse_labels, confounder_map)
        return self._cache[index]


def _apply_deterministic_augment(
    image: torch.Tensor,
    instance: torch.Tensor,
    mask: torch.Tensor,
    sparse: torch.Tensor,
    confounder: torch.Tensor,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if index % 2 == 0:
        image = torch.flip(image, dims=[2])
        instance = torch.flip(instance, dims=[2])
        mask = torch.flip(mask, dims=[2])
        sparse = torch.flip(sparse, dims=[1])
        confounder = torch.flip(confounder, dims=[2])
    if index % 3 == 0:
        image = torch.flip(image, dims=[1])
        instance = torch.flip(instance, dims=[1])
        mask = torch.flip(mask, dims=[1])
        sparse = torch.flip(sparse, dims=[0])
        confounder = torch.flip(confounder, dims=[1])
    return image, instance, mask, sparse, confounder
