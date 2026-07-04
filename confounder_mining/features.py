from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


FeatureBackend = Literal["handcrafted", "dinov2"]


@dataclass(frozen=True)
class FeatureMap:
    features: torch.Tensor
    stride_y: float
    stride_x: float


@lru_cache(maxsize=4)
def load_dinov2_model(model_name: str, device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def image_to_tensor(image: Image.Image, size: int | None = None) -> torch.Tensor:
    """Convert a PIL image to a normalized BCHW tensor in [0, 1]."""
    rgb_image = image.convert("RGB")
    if size is not None:
        rgb_image = rgb_image.resize((size, size), Image.BILINEAR)
    array = np.asarray(rgb_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def extract_handcrafted_features(
    image: Image.Image,
    grid_size: int = 32,
) -> FeatureMap:
    """Extract deterministic patch-level color and gradient features.

    This fallback keeps the prototype runnable without model downloads. It is
    not intended as the final paper feature extractor.
    """
    tensor = image_to_tensor(image)
    _, _, height, width = tensor.shape
    pooled = F.adaptive_avg_pool2d(tensor, (grid_size, grid_size))
    gray = tensor.mean(dim=1, keepdim=True)
    grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    pooled_grad = F.adaptive_avg_pool2d(grad_mag, (grid_size, grid_size))
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, grid_size),
        torch.linspace(0, 1, grid_size),
        indexing="ij",
    )
    coord = torch.stack([yy, xx], dim=0).unsqueeze(0)
    features = torch.cat([pooled, pooled_grad, coord], dim=1).squeeze(0)
    features = F.normalize(features, dim=0)
    return FeatureMap(
        features=features,
        stride_y=height / grid_size,
        stride_x=width / grid_size,
    )


def extract_dinov2_features(
    image: Image.Image,
    model_name: str = "dinov2_vits14",
    image_size: int = 448,
    device: str = "cpu",
) -> FeatureMap:
    """Extract DINOv2 patch-token features via torch.hub.

    This requires network access on first run unless the model is already in the
    local torch hub cache.
    """
    model = load_dinov2_model(model_name, device)

    tensor = image_to_tensor(image, size=image_size).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        output = model.forward_features(tensor)
        patch_tokens = output["x_norm_patchtokens"].squeeze(0)

    grid_size = image_size // 14
    features = patch_tokens.reshape(grid_size, grid_size, -1).permute(2, 0, 1)
    features = F.normalize(features.cpu(), dim=0)
    width, height = image.size
    return FeatureMap(
        features=features,
        stride_y=height / grid_size,
        stride_x=width / grid_size,
    )


def extract_features(
    image: Image.Image,
    backend: FeatureBackend = "handcrafted",
    device: str = "cpu",
    grid_size: int = 32,
) -> FeatureMap:
    if backend == "handcrafted":
        return extract_handcrafted_features(image, grid_size=grid_size)
    if backend == "dinov2":
        return extract_dinov2_features(image, device=device)
    raise ValueError(f"Unsupported feature backend: {backend}")
