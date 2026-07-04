# Experiment Brief (2026-07-04): CryoNuSeg Boundary-SFRM-UNet control closure

## Purpose

This brief closes the missing control chain for the current `Boundary-SFRM-UNet` paper line on `CryoNuSeg`.

The specific question was:

> Are the observed gains from structured failure feedback, or can they be explained by a generic refinement branch or a plain uncertainty cue?

To answer that, the following runs were consolidated under the same `freeze-coarse-for-all-epochs` regime.

## Code-check rule compliance

Before launching the new runs, the training script was checked twice:

1. static read of `D:\paper_MedIA Vol. 107–113\failure_region_reliability\scripts\train_boundary_sfrm_unet.py`
2. `py_compile` and `--help` execution on the same script

## Dataset used

- train images: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_train_split_full\images`
- train masks: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_train_split_full\masks`
- val images: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_val_split_full\images`
- val masks: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_val_split_full\masks`
- train confounders: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_train_split_confounders`
- val confounders: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_val_split_confounders`

Baseline checkpoint for all refinement runs:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_fullsup_unet_seed42\best_boundary_model.pt`

Common training settings for the new control runs:

- `epochs=8`
- `batch_size=4`
- `freeze_coarse_epochs=8`
- `boundary_radius=2`
- `seed=42`

## Runs closed

### Baseline

- run: `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_fullsup_unet_seed42`
- best boundary epoch: `7`

### Generic refinement controls

1. `two_pass_no_risk`
   - run: `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_sfrm_unet_two_pass_no_risk_freeze8_seed42`
2. `entropy_only`
   - run: `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_sfrm_unet_entropy_only_freeze8_seed42`

### Structured failure variants

1. `learned_failure_head`
   - run: `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42`
2. `learned_failure_head_calibrated`
   - run: `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_sfrm_unet_learned_failure_head_calibrated_freeze8_seed42`

## Best-result comparison

Reference baseline anchor:

- Dice: `0.7429`
- Boundary Dice: `0.5992`
- AJI: `0.3555`
- PQ: `0.2248`
- Confounder FPR: `0.4317`

### Table 1. CryoNuSeg control closure summary

| Variant | Best epoch | Dice | Delta Dice | Boundary Dice | Delta Boundary | AJI | Delta AJI | PQ | Delta PQ | Delta confounder FPR | Mean primary risk | High-risk frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 7 | 0.7429 | - | 0.5992 | - | 0.3555 | - | 0.2248 | - | - | - | - |
| Two-pass no risk | 8 | 0.7418 | -0.0011 | 0.6097 | +0.0105 | 0.3627 | +0.0072 | 0.2232 | -0.0016 | -0.0257 | 0.0000 | 0.0000 |
| Entropy only | 8 | 0.7433 | +0.0004 | 0.5945 | -0.0047 | 0.3634 | +0.0079 | 0.2163 | -0.0085 | +0.0217 | 0.6435 | 0.7763 |
| Learned failure head | 8 | 0.7558 | +0.0129 | 0.6236 | +0.0244 | 0.3839 | +0.0284 | 0.2410 | +0.0161 | -0.0276 | 0.3717 | 0.0000 |
| Learned failure head calibrated | 8 | 0.7554 | +0.0124 | 0.6172 | +0.0180 | 0.3823 | +0.0267 | 0.2349 | +0.0101 | -0.0134 | 0.3515 | 0.3409 |

## Main findings

### 1. The gain is not explained by a generic second pass

`two_pass_no_risk` only gave:

- `+0.0105` Boundary Dice
- `+0.0072` AJI
- `-0.0016` PQ
- `-0.0011` Dice

This means a refinement decoder alone is not enough to reproduce the main effect.

### 2. The gain is not explained by plain uncertainty injection

`entropy_only` failed to improve the main structural endpoint:

- Boundary Dice dropped from `0.5992` to `0.5945`
- PQ dropped from `0.2248` to `0.2163`
- confounder leakage worsened by `+0.0217`

At the same time, it produced:

- `mean_primary_risk = 0.6435`
- `high_risk_frac = 0.7763`

Interpretation:

- entropy produces a broad and overactive risk field
- it does not isolate the anatomically useful failure zones that the refinement branch needs

### 3. Structured failure feedback is the strongest segmentation-improving cue on CryoNuSeg

`learned_failure_head` achieved the strongest overall segmentation result:

- Dice: `0.7558` (`+0.0129`)
- Boundary Dice: `0.6236` (`+0.0244`)
- AJI: `0.3839` (`+0.0284`)
- PQ: `0.2410` (`+0.0161`)
- confounder FPR: `0.4041` (`-0.0276`)

This is the cleanest current evidence that feeding structured failure information back into the decoder improves:

- boundary quality
- object separation
- confounder suppression

### 4. Calibration fixes the deployability problem of the risk map

The uncalibrated `learned_failure_head` improved segmentation strongly, but still had:

- `high_risk_frac = 0.0000`

This means the branch was useful for training-time refinement, but not yet suitable as a thresholded deployment-time audit layer.

The calibrated variant changed that:

- `high_risk_frac = 0.3409`
- `delta_high_risk_error = -0.0149`
- boundary/object metrics remained clearly above baseline

Interpretation:

- calibration converts the risk head from a latent training cue into a usable spatial audit signal
- this costs a small amount of segmentation peak performance relative to the uncalibrated best run
- but it materially improves the paper's claim that SFRM is both a refinement signal and an interpretable risk map

## Recommended manuscript use

### Core architectural claim

Use the following wording direction in the manuscript:

> The performance gain does not come from adding a generic refinement decoder or injecting global uncertainty. It comes from feeding structured failure cues back into decoder feature learning.

### Main-text result hierarchy on CryoNuSeg

1. **Best performance variant:** `learned_failure_head`
   - use this to support the segmentation-improvement claim
2. **Audit-ready variant:** `learned_failure_head_calibrated`
   - use this to support the interpretable-risk-map claim
3. **Negative controls:** `two_pass_no_risk` and `entropy_only`
   - use these to show why structure matters

### Paper-level implication

For the current single-paper framing, CryoNuSeg now supports the desired story much more convincingly:

- **output-derived failure signals can be extracted from U-Net**
- **those signals can be fed back into decoder refinement**
- **structured feedback is more useful than generic uncertainty cues**
- **a calibrated variant can expose a non-trivial high-risk region for audit**

## Immediate next recommendation

The CryoNuSeg control chain is now strong enough for manuscript writing.

The next experimental priority should be:

1. keep `CryoNuSeg` as a main-text positive evidence dataset
2. keep `CoNSeP` as a hard stress-test / limitation dataset
3. use `MoNuSeg` for mechanism visualization and architecture explanation
4. update the manuscript result tables and discussion around:
   - structured feedback vs generic uncertainty
   - training-effective vs audit-ready SFRM variants

