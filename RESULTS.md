# Experiment Results Index

This repository includes numeric experiment outputs but excludes raw datasets, dense cached risk maps, and model checkpoints.

## Boundary-SFRM-UNet Runs

`experiments/boundary_sfrm_runs/` contains epoch-level JSON summaries and per-sample CSV files for:

- CryoNuSeg baseline, two-pass control, entropy-only control, learned failure head, and calibrated failure head;
- CoNSeP baseline, two-pass control, entropy-only control, learned failure head, and contact-restricted variants;
- MoNuSeg baseline and structured-cue ablations;
- TNBC pilot runs used to compare joint training with frozen-coarse refinement.

Checkpoint files (`*.pt`, `*.pth`, and `*.ckpt`) are excluded.

## Channel Ablation

`experiments/ablation_channel_masking/` contains inference-time masking results for the four risk channels:

- coarse probability;
- normalized entropy;
- boundary-failure probability;
- structure-failure probability.

Both per-case CSV files and aggregate JSON summaries are included for CryoNuSeg and CoNSeP.

## Reliability Audit Outputs

`experiments/summaries/` contains Stage 0--7 feature tables, GroupKFold predictions, review-budget simulation outputs, bootstrap statistics, and supporting figures. These files belong to the earlier SFRM reliability-audit analysis and are retained as mechanism evidence.

## Paper-Level Values

The principal tables are reproduced in the repository README. Values should be traced to the corresponding epoch summary and per-sample CSV before reuse in another manuscript or benchmark.

## Excluded Artifacts

- raw public-dataset images and annotations;
- generated patch datasets;
- model checkpoints;
- dense risk-map caches;
- temporary smoke-test outputs that contain no paper evidence;
- manuscript build artifacts.
