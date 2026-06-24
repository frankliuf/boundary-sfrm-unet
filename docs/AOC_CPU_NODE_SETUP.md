# AOC CPU Auxiliary Node

## Status

Configured on 2026-06-24 as a CPU auxiliary node for the
`failure_region_reliability` project.

## Connection

```text
host: AOC
ip: 192.168.3.27
ssh user: admin
```

SSH key authentication is configured from the main workstation.

## Hardware and OS

```text
OS: Windows 11 Home
CPU: 13th Gen Intel Core i5-13500H
RAM: 16 GB
GPU: Intel UHD Graphics only
Disk:
  C: ~128 GB total
  D: ~329 GB total, ~327 GB free at setup
```

No NVIDIA GPU was detected from the SSH session. This node should not be used
for GPU training or foundation-model feature extraction that requires CUDA.

## Remote Paths

```text
project: D:\research\failure_region_reliability
venv:    D:\research\venvs\failure_region_cpu
```

## Installed Python Stack

The virtual environment was created with Python 3.12 and contains:

- numpy
- pandas
- scipy
- scikit-image
- scikit-learn
- matplotlib
- seaborn
- pyyaml
- tqdm
- pillow

PyTorch was intentionally not installed during the initial setup. Install a CPU
build only if this node needs to load checkpoints directly. Prefer transferring
precomputed probability maps or feature tables from the main workstation.

## Intended Use

Use AOC for:

- feature-table postprocessing;
- statistical tests;
- UMAP/t-SNE and other visualization jobs;
- review-budget simulation;
- figure generation;
- storing intermediate CSV/JSONL caches.

Do not use AOC for:

- CUDA training;
- DINOv2/SAM/UNI heavy feature extraction;
- long GPU-dependent experiments.

## Smoke Test

Remote project import test passed:

```text
remote_project_import_ok 0.8
```

