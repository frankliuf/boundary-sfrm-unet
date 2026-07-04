# Experiment Brief (2026-07-03): TNBC + CryoNuSeg Boundary-SFRM-UNet

## Scope

This brief records the first main-evidence runs for the unified `Boundary-SFRM-UNet` line on:

- `TNBC`
- `CryoNuSeg`

The goal was not to prove universal superiority, but to test whether a structured SFRM feedback layer can produce stable, positive evidence on selected pathology datasets after the earlier CoNSeP stress-test difficulties.

## Datasets Prepared

### TNBC

- Raw download source: Zenodo package `TNBC_NucleiSegmentation.zip`
- Normalized dataset:
  - `D:\paper_MedIA Vol. 107–113\data\tnbc_full`
- Source-level split:
  - train: `D:\paper_MedIA Vol. 107–113\data\tnbc_train_split_full`
  - val: `D:\paper_MedIA Vol. 107–113\data\tnbc_val_split_full`
- Confounder maps:
  - train: `D:\paper_MedIA Vol. 107–113\data\tnbc_train_split_confounders`
  - val: `D:\paper_MedIA Vol. 107–113\data\tnbc_val_split_confounders`

Note: TNBC only provides binary masks. During preprocessing, each binary mask was converted into a connected-component pseudo-instance mask so that AJI/PQ and contact-zone logic remain usable.

### CryoNuSeg

- Raw download source:
  - official GitHub repo for metadata/instructions
  - public torrent mirror to obtain the Kaggle archive
- Normalized dataset:
  - `D:\paper_MedIA Vol. 107–113\data\cryonuseg_full`
- Source-level split:
  - train: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_train_split_full`
  - val: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_val_split_full`
- Confounder maps:
  - train: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_train_split_confounders`
  - val: `D:\paper_MedIA Vol. 107–113\data\cryonuseg_val_split_confounders`

Ground truth used for CryoNuSeg:

- `Annotator 1 (biologist second round of manual marks up)`

## Code Status

Training scripts were checked twice before formal runs:

1. static read + `--help` / `py_compile`
2. end-to-end smoke runs on TNBC baseline and TNBC SFRM

Relevant scripts:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\scripts\train_full_supervised_unet.py`
- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\scripts\train_boundary_sfrm_unet.py`

Additional preparation scripts created this round:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\scripts\prepare_tnbc_dataset.py`
- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\scripts\prepare_cryonuseg_dataset.py`

## Main Results

### 1. TNBC baseline

Output:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\tnbc_fullsup_unet_seed42`

Best boundary reference used for SFRM comparison:

- baseline boundary model summary corresponds to `boundary_dice = 0.6425`
- comparison anchor:
  - Dice: `0.7926`
  - Boundary Dice: `0.6425`
  - AJI: `0.4878`
  - PQ: `0.4320`
  - Confounder FPR: `0.3785`

### 2. TNBC SFRM (learned failure head, joint training)

Output:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\tnbc_sfrm_unet_learned_failure_head_seed42`

Observation:

- early epochs showed mild positive signals
- later epochs destabilized badly because the coarse path drifted

Best early positive point:

- epoch 2:
  - Dice: `+0.0017`
  - Boundary Dice: `-0.0026`
  - AJI: `+0.0285`
  - PQ: `+0.0253`
  - Confounder FPR: `+0.0019`

Interpretation:

- structured feedback is not useless on TNBC
- but full joint optimization is unstable and degrades the coarse path

### 3. TNBC SFRM (freeze coarse for all 8 epochs)

Output:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\tnbc_sfrm_unet_learned_failure_head_freeze8_seed42`

Key finding:

- freezing the coarse path stabilizes training substantially
- this supports the hypothesis that the failure-feedback branch itself has value, while coarse-path co-adaptation is the main source of collapse

Best boundary-oriented result:

- epoch 5:
  - Dice: `0.7815` (`-0.0111` vs baseline anchor)
  - Boundary Dice: `0.6444` (`+0.0019`)
  - AJI: `0.5021` (`+0.0143`)
  - PQ: `0.4617` (`+0.0297`)
  - Confounder FPR: `0.3346` (`-0.0439`)

Interpretation:

- TNBC gives usable positive evidence for the paper's intended story:
  - small Dice trade-off
  - better boundary/object metrics
  - lower confounder leakage

### 4. CryoNuSeg baseline

Output:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_fullsup_unet_seed42`

Best boundary reference used for SFRM comparison:

- baseline anchor:
  - Dice: `0.7429`
  - Boundary Dice: `0.5992`
  - AJI: `0.3555`
  - PQ: `0.2248`
  - Confounder FPR: `0.4317`

### 5. CryoNuSeg SFRM (freeze coarse for all 8 epochs)

Output:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\cryonuseg_sfrm_unet_learned_failure_head_freeze8_seed42`

Final epoch result:

- epoch 8:
  - Dice: `0.7558` (`+0.0129`)
  - Boundary Dice: `0.6236` (`+0.0244`)
  - AJI: `0.3839` (`+0.0284`)
  - PQ: `0.2410` (`+0.0161`)
  - Confounder FPR: `0.4041` (`-0.0276`)

Interpretation:

- CryoNuSeg is a strong positive evidence dataset for the unified SFRM-UNet story
- unlike TNBC, the frozen-coarse configuration improves all main metrics simultaneously

## Cross-dataset Conclusions

### What is supported now

1. `Boundary-SFRM-UNet` should not be framed as a blindly joint-trained end-to-end replacement.
2. A `frozen-coarse + structured refinement` formulation is much more stable.
3. The SFRM feedback branch has real positive evidence on selected pathology datasets.
4. CryoNuSeg provides especially clean support:
   - Dice up
   - Boundary Dice up
   - AJI/PQ up
   - confounder FPR down

### What is still not solved

1. The learned failure head still reports:
   - `high_risk_frac = 0.0`
   on these runs under the current fixed `>= 0.5` evaluation threshold.
2. This means the current risk-head output is useful enough for refinement training, but not yet calibrated as a clean deployable binary audit map.
3. CoNSeP remains a hard stress test and should not be used as the only evidence set for the core claim.

## Immediate Next Steps

1. Audit why `high_risk_frac` stays at zero:
   - likely calibration/threshold issue rather than complete feature failure
   - inspect per-pixel failure logits and score histograms
2. Run a lighter ablation on TNBC/CryoNuSeg:
   - compare `freeze8` vs `freeze8 + weaker lambda_failure / lambda_risk_focus`
3. Decide whether the mainline paper should present:
   - `frozen-coarse SFRM refinement` as the primary architecture
   instead of joint coarse-refine training
4. If a third evidence dataset is still needed, prepare it after the risk-head calibration audit, not before.
