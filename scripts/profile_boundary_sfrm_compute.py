from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((ROOT / "confounder_mining").resolve()))

from confounder_mining.unet import SmallUNet  # noqa: E402
from src.models.boundary_sfrm_unet import BoundarySFRMUNet  # noqa: E402


class BoundarySFRMEndToEnd(torch.nn.Module):
    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        self.model = BoundarySFRMUNet(base_channels=base_channels, risk_channels=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.model.forward_coarse(x)
        coarse_prob = torch.sigmoid(state.coarse_logits)
        failure_prob = torch.sigmoid(state.failure_logits)
        eps = 1e-7
        p = coarse_prob.clamp(eps, 1.0 - eps)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        entropy = entropy / entropy.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        risk_tensor = torch.cat([coarse_prob, entropy, failure_prob], dim=1)
        return self.model.decode_refine(state, risk_tensor)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_inference_ms(model: torch.nn.Module, x: torch.Tensor, warmup: int = 20, repeats: int = 50) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / repeats


def measure_flops(model: torch.nn.Module, x: torch.Tensor) -> int:
    model.eval()
    activities = [ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(ProfilerActivity.CUDA)
    with torch.no_grad():
        with profile(activities=activities, with_flops=True, record_shapes=False, profile_memory=False) as prof:
            _ = model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
    total = 0
    for event in prof.key_averages():
        if event.flops is not None:
            total += int(event.flops)
    return total


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 3, 512, 512)
    x = torch.randn(*input_shape, device=device)

    baseline = SmallUNet(in_channels=3, base_channels=24).to(device)
    boundary = BoundarySFRMEndToEnd(base_channels=24).to(device)

    results = {
        "device": str(device),
        "input_shape": list(input_shape),
        "baseline": {
            "parameters": count_parameters(baseline),
            "flops": measure_flops(baseline, x),
            "inference_ms": measure_inference_ms(baseline, x),
        },
        "boundary_sfrm_unet": {
            "parameters": count_parameters(boundary),
            "flops": measure_flops(boundary, x),
            "inference_ms": measure_inference_ms(boundary, x),
        },
    }

    out_path = ROOT / "docs" / "compute_profile_boundary_sfrm_unet.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
