# Boundary-SFRM-UNet dataset readiness

Updated: 2026-07-04

## Recommended main evidence scope

For the tightened paper scope of pathology nuclei segmentation, the current recommended dataset set is:

1. `CoNSeP` - main evidence for crowded/contact-zone/clustered-boundary failures
2. `CryoNuSeg` - main evidence for artifact/confounder-heavy boundaries
3. `MoNuSeg` - supporting mechanism and cue-comparison benchmark

`TNBC` remains available locally, but should be treated as optional supplementary evidence rather than a primary main-text dataset.

## Ready local layouts

### CoNSeP

Standardized for the Boundary-SFRM-UNet training line:

- `data/consep_train_split_full`
- `data/consep_val_split_full`
- `data/consep_test_split_full`
- `data/consep_train_split_confounders`
- `data/consep_val_split_confounders`
- `data/consep_test_split_confounders`

Verified counts:

- train: 792 images / 792 masks / 792 confounders
- val: 180 images / 180 masks / 180 confounders
- test: 504 images / 504 masks / 504 confounders

Source material retained:

- raw extracted package: `data/consep/CoNSeP`
- earlier audit/grid patch source: `data/consep_*_grid_*`

### CryoNuSeg

Already in standardized split layout:

- `data/cryonuseg_train_split_full`
- `data/cryonuseg_val_split_full`
- `data/cryonuseg_train_split_confounders`
- `data/cryonuseg_val_split_confounders`

### MoNuSeg

Already in standardized split / support-study layout:

- `data/monuseg_train_split_patches`
- `data/monuseg_val_split_patches`
- `data/monuseg_train_split_confounders`
- `data/monuseg_val_split_confounders`
- additional support-study variants under `data/monuseg_*`

## Immediate experimental implication

The next Boundary-SFRM-UNet experiments should switch the main evidence pair from:

- `TNBC + CryoNuSeg`

to:

- `CoNSeP + CryoNuSeg`

while retaining:

- `MoNuSeg` for cue-comparison / mechanism analysis.
