from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class CoarseForwardState:
    e1: torch.Tensor
    e2: torch.Tensor
    e3: torch.Tensor
    bottleneck: torch.Tensor
    coarse_d1: torch.Tensor
    failure_logits: torch.Tensor
    coarse_logits: torch.Tensor


class RiskModulationBlock(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        risk_channels: int,
        hidden_channels: int = 16,
        gate_scale: float = 1.0,
        bias_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.gate_scale = gate_scale
        self.bias_scale = bias_scale
        self.stem = nn.Sequential(
            nn.Conv2d(risk_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.gate_proj = nn.Conv2d(hidden_channels, feat_channels, kernel_size=1)
        self.bias_proj = nn.Conv2d(hidden_channels, feat_channels, kernel_size=1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, feat: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        risk_feat = self.stem(risk)
        gate_delta = torch.tanh(self.gate_proj(risk_feat))
        bias_delta = self.bias_proj(risk_feat)
        return feat * (1.0 + self.gate_scale * gate_delta) + self.bias_scale * bias_delta


class RefineDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        risk_channels: int,
        gate_scale: float = 1.0,
        bias_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        hidden_channels = max(16, out_channels // 2)
        self.modulation = RiskModulationBlock(
            feat_channels=out_channels,
            risk_channels=risk_channels,
            hidden_channels=hidden_channels,
            gate_scale=gate_scale,
            bias_scale=bias_scale,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(torch.cat([x, skip], dim=1))
        return self.modulation(x, risk)


class BoundarySFRMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        risk_channels: int = 4,
        gate_scale: float = 1.0,
        bias_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.risk_channels = risk_channels

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.coarse_dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.coarse_dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.coarse_dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.failure_head = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.coarse_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.failure_head.weight)
        nn.init.zeros_(self.failure_head.bias)

        self.refine_dec3 = RefineDecoderBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            risk_channels=risk_channels,
            gate_scale=gate_scale,
            bias_scale=bias_scale,
        )
        self.refine_dec2 = RefineDecoderBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            risk_channels=risk_channels,
            gate_scale=gate_scale,
            bias_scale=bias_scale,
        )
        self.refine_dec1 = RefineDecoderBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            risk_channels=risk_channels,
            gate_scale=gate_scale,
            bias_scale=bias_scale,
        )
        self.refine_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(e3, 2))
        return e1, e2, e3, bottleneck

    def decode_coarse_features(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d3 = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.coarse_dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.coarse_dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.coarse_dec1(torch.cat([d1, e1], dim=1))
        return d3, d2, d1

    def decode_coarse(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        _, _, d1 = self.decode_coarse_features(e1, e2, e3, bottleneck)
        return self.coarse_head(d1)

    def forward_coarse(self, x: torch.Tensor) -> CoarseForwardState:
        e1, e2, e3, bottleneck = self.encode(x)
        _, _, coarse_d1 = self.decode_coarse_features(e1, e2, e3, bottleneck)
        failure_logits = self.failure_head(coarse_d1)
        coarse_logits = self.coarse_head(coarse_d1)
        return CoarseForwardState(
            e1=e1,
            e2=e2,
            e3=e3,
            bottleneck=bottleneck,
            coarse_d1=coarse_d1,
            failure_logits=failure_logits,
            coarse_logits=coarse_logits,
        )

    def _risk_pyramid(self, risk_tensor: torch.Tensor, state: CoarseForwardState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        risk_e3 = F.interpolate(risk_tensor, size=state.e3.shape[-2:], mode="bilinear", align_corners=False)
        risk_e2 = F.interpolate(risk_tensor, size=state.e2.shape[-2:], mode="bilinear", align_corners=False)
        risk_e1 = F.interpolate(risk_tensor, size=state.e1.shape[-2:], mode="bilinear", align_corners=False)
        return risk_e3, risk_e2, risk_e1

    def decode_refine(self, state: CoarseForwardState, risk_tensor: torch.Tensor) -> torch.Tensor:
        risk_e3, risk_e2, risk_e1 = self._risk_pyramid(risk_tensor, state)
        d3 = self.refine_dec3(state.bottleneck, state.e3, risk_e3)
        d2 = self.refine_dec2(d3, state.e2, risk_e2)
        d1 = self.refine_dec1(d2, state.e1, risk_e1)
        return self.refine_head(d1)

    def forward(self, x: torch.Tensor, risk_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.forward_coarse(x)
        refined_logits = self.decode_refine(state, risk_tensor)
        return state.coarse_logits, refined_logits

    def initialize_from_small_unet(self, baseline_state_dict: dict[str, torch.Tensor]) -> None:
        self.enc1.load_state_dict(_slice_state_dict(baseline_state_dict, "enc1"))
        self.enc2.load_state_dict(_slice_state_dict(baseline_state_dict, "enc2"))
        self.enc3.load_state_dict(_slice_state_dict(baseline_state_dict, "enc3"))
        self.bottleneck.load_state_dict(_slice_state_dict(baseline_state_dict, "bottleneck"))
        self.coarse_dec3.load_state_dict(_slice_state_dict(baseline_state_dict, "dec3"))
        self.coarse_dec2.load_state_dict(_slice_state_dict(baseline_state_dict, "dec2"))
        self.coarse_dec1.load_state_dict(_slice_state_dict(baseline_state_dict, "dec1"))
        self.coarse_head.load_state_dict(_slice_state_dict(baseline_state_dict, "head"))

        self.refine_dec3.conv.load_state_dict(self.coarse_dec3.state_dict())
        self.refine_dec2.conv.load_state_dict(self.coarse_dec2.state_dict())
        self.refine_dec1.conv.load_state_dict(self.coarse_dec1.state_dict())
        self.refine_head.load_state_dict(self.coarse_head.state_dict())

    def set_coarse_path_requires_grad(self, requires_grad: bool) -> None:
        modules = self.coarse_path_modules()
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def coarse_path_modules(self) -> list[nn.Module]:
        return [
            self.enc1,
            self.enc2,
            self.enc3,
            self.bottleneck,
            self.coarse_dec3,
            self.coarse_dec2,
            self.coarse_dec1,
            self.coarse_head,
        ]


def _slice_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    full_prefix = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(full_prefix):
            output[key[len(full_prefix) :]] = value
    return output
