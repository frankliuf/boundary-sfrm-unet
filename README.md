# Boundary-SFRM-UNet

Boundary-SFRM-UNet is a failure-aware refinement framework for pathology nuclei segmentation. A frozen coarse U-Net produces a probability map and decoder features. A two-channel failure head predicts boundary and structure failure fields. These fields are concatenated with coarse probability and normalized entropy to form a structured risk tensor that modulates a second decoder at three spatial scales.

The repository contains the implementation, dataset preparation scripts, training and evaluation pipelines, paper-level result tables, ablations, and selected figures. Raw images and model checkpoints are intentionally excluded.

## Method

The main pipeline is:

1. train a fully supervised coarse U-Net;
2. initialize and freeze the coarse path;
3. learn boundary- and structure-failure maps from the final coarse decoder feature;
4. build `R = concat(P, H(P), F_b, F_s)`;
5. inject resized copies of `R` into three refinement-decoder stages through multiplicative and additive risk modulation;
6. optimize the failure head and refinement decoder while preserving low-risk coarse predictions.

The current evidence supports a bounded claim: structured failure feedback improves boundary- and object-sensitive segmentation on CryoNuSeg and remains competitive on the denser CoNSeP stress test, where confounder leakage is still challenging.

## Main Results

### CryoNuSeg

| Model | Dice | Boundary Dice | AJI | PQ | Confounder FPR |
|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7429 | 0.5992 | 0.3555 | 0.2248 | 0.4317 |
| Two-pass, no risk | 0.7418 | 0.6097 | 0.3627 | 0.2232 | 0.4060 |
| Entropy only | 0.7433 | 0.5945 | 0.3634 | 0.2163 | 0.4534 |
| Learned failure head | **0.7558** | **0.6236** | **0.3839** | **0.2410** | **0.4041** |

### CoNSeP stress test

| Model | Dice | Boundary Dice | AJI | PQ | Confounder FPR |
|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7315 | 0.6236 | 0.3762 | 0.3000 | 0.4461 |
| Two-pass, no risk | **0.7593** | **0.6439** | 0.3568 | 0.2954 | 0.4904 |
| Entropy only | 0.7502 | 0.6399 | 0.3816 | **0.3187** | 0.4706 |
| Learned failure head | 0.7530 | 0.6411 | **0.3871** | 0.3136 | 0.4875 |

Lower Confounder FPR is better. Full result traces, per-sample CSV files, and ablations are retained under `experiments/`; see [RESULTS.md](RESULTS.md).

## Repository Layout

```text
confounder_mining/             bundled dataset, U-Net, mining, and metric utilities
src/models/                    Boundary-SFRM-UNet and repair models
src/failure_regions/           structured risk-map construction
src/metrics/                   segmentation and reliability metrics
scripts/                       preparation, training, audit, statistics, and figure scripts
experiments/boundary_sfrm_runs training summaries and per-sample result CSV files
experiments/summaries/         reliability-audit feature tables and aggregate outputs
experiments/ablation_channel_masking/
figures/paper1/                selected PNG figures
docs/                          protocols, experiment logs, and research decisions
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch appropriate for the local driver when GPU training is required.

## Data

The experiments use public pathology datasets including CryoNuSeg, CoNSeP, MoNuSeg, and TNBC. Dataset images and annotations are not redistributed. Prepare local folders containing images, semantic masks, instance maps, and confounder maps, then pass their paths explicitly to the scripts.

Example baseline training:

```bash
python scripts/train_full_supervised_unet.py \
  --train-images /path/to/train/images \
  --train-masks /path/to/train/masks \
  --val-images /path/to/val/images \
  --val-masks /path/to/val/masks \
  --train-confounders /path/to/train/confounders \
  --val-confounders /path/to/val/confounders \
  --output-dir experiments/boundary_sfrm_runs/baseline
```

Example Boundary-SFRM-UNet refinement:

```bash
python scripts/train_boundary_sfrm_unet.py \
  --variant learned_failure_head \
  --baseline-checkpoint /path/to/best_boundary_model.pt \
  --train-images /path/to/train/images \
  --train-masks /path/to/train/masks \
  --val-images /path/to/val/images \
  --val-masks /path/to/val/masks \
  --train-confounders /path/to/train/confounders \
  --val-confounders /path/to/val/confounders \
  --failure-teacher-mix 1.0 \
  --failure-structure-gate-scale 0.5 \
  --failure-contact-restrict \
  --freeze-coarse-epochs 8 \
  --epochs 8 \
  --output-dir experiments/boundary_sfrm_runs/learned_failure_head
```

## Reproducibility Notes

- Coarse checkpoints are excluded because of repository size and must be regenerated locally.
- Experiment CSV and JSON outputs are versioned; raw images, cached risk maps, and weights are ignored.
- Main model selection uses validation boundary Dice within each model family.
- CoNSeP is reported as a crowded-scene stress test rather than a universal-win benchmark.

## Status

This is a research repository accompanying an active manuscript. Interfaces and result organization may change before formal release.
