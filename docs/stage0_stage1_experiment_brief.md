# SFRM Stage 0-1 Experiment Brief

Date: 2026-06-24

## Current Question

Paper 1 should not claim that SFRM is a universal Dice predictor. The current
experiments test a narrower and stronger claim:

> Spatial failure-region descriptors reveal boundary, topology, and local
> overconfidence failure modes that are not fully captured by a single global
> uncertainty score.

## Completed Experiments

### Stage 0: Feature-Discrimination Audit

Completed for four prediction sets:

- MoNuSeg point-only U-Net, 168 patches.
- MoNuSeg prototype contrastive model, 168 patches.
- CoNSeP fixed-grid point-only U-Net, 504 patches.
- CoNSeP fixed-grid prototype contrastive model, 504 patches.

Quality checks passed:

- Python compile checks passed.
- 5-patch smoke tests passed.
- Full extraction completed.
- Dice and boundary Dice reproduce previous evaluation CSVs with maximum
  differences around `5e-7`.
- Leakage feature count is 0.
- Per-patch probability, entropy, prediction, ground truth, and error maps were
  cached as `.npz` files for qualitative inspection.

### Stage 1: Fixed Review-Budget Simulation

Completed for the same four prediction sets.

Compared scores:

- Global uncertainty baselines:
  - `global_mean_entropy`;
  - `global_max_entropy`;
  - `global_mean_margin_uncertainty`;
  - `foreground_area_frac`.
- SFRM scores:
  - `boundary_risk_score`;
  - `boundary_overconfidence_score`;
  - `boundary_abnormality_score`;
  - `boundary_dual_risk_score`;
  - `topology_risk_score`;
  - `uncertainty_cluster_score`;
  - `sfrm_composite_score`;
  - `sfrm_balanced_score`.

Budgets:

- 5%, 10%, 20%, and 30% reviewed patches.

## Main Findings

### Finding 1: SFRM is strong on MoNuSeg, especially after model improvement

On MoNuSeg prototype predictions, SFRM-style scores strongly outperformed
global mean entropy for several endpoints at 10% review budget:

| Endpoint | Best SFRM recall | Global mean entropy recall |
|---|---:|---:|
| `bad_dice_lt_0.65` | 0.5484 | 0.0323 |
| `low_boundary_dice_le_q25` | 0.4048 | 0.0238 |
| `gray_high_dice_high_boundary_error` | 0.7000 | 0.4500 |

This supports the argument that, after a stronger segmentation model suppresses
coarse uncertainty, residual failures become more spatial and boundary-local.

### Finding 2: Boundary overconfidence is a real failure mode

The first Stage 1 implementation assumed that higher uncertainty always means
higher review priority. This failed on boundary-local errors. Adding
`boundary_overconfidence_score` revealed a useful pattern:

- MoNuSeg point-only, `gray_high_dice_high_boundary_error`, 10% budget:
  - `boundary_overconfidence_score`: 0.6250 recall;
  - global mean entropy: 0.0000 recall.

This is important for the paper. The framework should not be framed as another
uncertainty method. It should be framed as spatial failure-region modeling,
including both uncertain and overconfident local failures.

### Finding 3: CoNSeP supports SFRM, but not as a universal winner

CoNSeP results are mixed:

- For `high_lecr_boundary_ge_q0.75`, `boundary_overconfidence_score` is modestly
  better than global mean entropy in both point-only and prototype settings.
- For `low_boundary_dice_le_q25` in prototype predictions, `boundary_risk_score`
  beats global max entropy at 10% review budget.
- For `gray_high_dice_high_boundary_error`, global mean entropy remains stronger
  than the current SFRM composite on CoNSeP.

This means CoNSeP should be used as a stress test and limitation-aware
validation, not as an overclaimed victory table.

## Current Scientific Interpretation

The results are strong enough to continue, but the claim must be precise:

- Supported:
  - local boundary-risk descriptors identify failure modes missed by global
    mean entropy;
  - topology and boundary descriptors are useful for bad-case and low-boundary
    quality detection;
  - overconfident boundary failure is a distinct and publishable observation;
  - review-budget simulation is a suitable clinical story.
- Not supported:
  - a single SFRM composite universally beats all global uncertainty baselines;
  - CoNSeP gray-zone results alone prove SFRM superiority;
  - high boundary-error area is a clean endpoint, because it can be confounded
    by object/foreground area.

## Recommended Next Experiments

1. **Figure preparation**
   - Figure A: feature separability and feature-family AUROC.
   - Figure B: review-budget curves.
   - Figure C: qualitative gray-zone cases.
   - Figure D: overconfidence boundary failure examples.

2. **Manuscript table consolidation**
   - Keep a small main table with endpoint-specific results.
   - Move broad predictor sweeps and all weak/mixed endpoints to supplement.
   - Do not present SFRM as a universal ranking function.

## Stage 2 Updates

### Qualitative Mechanism Audit

Diagnostic figures were generated for MoNuSeg gray-zone cases.

Point-only:

- 8 gray-zone positives.
- 5 were captured by `boundary_overconfidence_score` at 10% review budget and
  missed by `global_mean_entropy`.
- 0 were captured by global mean entropy and missed by SFRM.

Prototype:

- 20 gray-zone positives.
- 8 were captured by `sfrm_balanced_score` and missed by global mean entropy.
- 3 were captured by global mean entropy and missed by SFRM.
- 6 were captured by both.

The visual cases support structured local boundary failure, but the term
"low-global-entropy failure" is too strong. The safer wording is:

> global mean entropy can miss spatially structured boundary failures under a
> fixed review budget.

### CoNSeP Area-Controlled Analysis

CoNSeP was re-evaluated with five predicted-foreground-area strata. The review
budget was allocated inside each area bin.

Key 10% budget results:

| Model | Endpoint | Best SFRM-style recall | Global mean entropy recall after area control |
|---|---|---:|---:|
| point | `high_lecr_boundary_ge_q0.75` | 0.1905 | 0.0397 |
| point | `low_boundary_dice_le_q25` | 0.1349 | 0.0794 |
| point | `gray_high_dice_high_boundary_error` | 0.1905 | 0.2024 |
| prototype | `high_lecr_boundary_ge_q0.75` | 0.1349 | 0.0317 |
| prototype | `low_boundary_dice_le_q25` | 0.2698 | 0.2540 |
| prototype | `gray_high_dice_high_boundary_error` | 0.1630 | 0.1522 |

Interpretation:

- Area confounding is real in CoNSeP.
- After controlling for area, boundary overconfidence becomes useful for
  `high_lecr_boundary`.
- Gray-zone detection remains mixed and should be framed cautiously.

### Lightweight Predictor Analysis

GroupKFold lightweight predictors were trained with 14 source-image groups and
5 folds.

Feature sets:

- global-only features;
- SFRM-only features;
- all deployable features.

Models:

- L2 logistic regression;
- L1 logistic regression;
- shallow random forest.

Key results:

| Dataset | Model | Endpoint | Best SFRM-style 10% recall | Key global 10% recall | Conclusion |
|---|---|---:|---:|---:|---|
| MoNuSeg | point | `gray_high_dice_high_boundary_error` | 0.6250 | 0.0000 | boundary overconfidence provides complementary signal |
| MoNuSeg | prototype | `bad_dice_lt_0.65` | 0.5484 | 0.0323 | boundary-risk remains strong after model improvement |
| MoNuSeg | prototype | `low_boundary_dice_le_q25` | 0.3810 | 0.0238 | SFRM improves boundary-quality screening |
| CoNSeP | point | `high_lecr_boundary_ge_q0.75` | 0.2222 | 0.1587 | SFRM improves local boundary-error screening |
| CoNSeP | prototype | `high_lecr_boundary_ge_q0.75` | 0.2222 | 0.1032 | SFRM improves local boundary-error screening |
| CoNSeP | prototype | `low_boundary_dice_le_q25` | 0.3175 | 0.1587 | SFRM improves low-boundary-quality screening |
| CoNSeP | point/prototype | `bad_dice_lt_0.65` | mixed | global better | global features remain useful for coarse bad-case detection |
| CoNSeP | point/prototype | `gray_high_dice_high_boundary_error` | mixed | global/area better or tied | gray-zone is area/foreground-confounded |

Interpretation:

- The predictor experiment validates SFRM as a useful representation for
  endpoint-specific boundary and local-error screening.
- It also confirms that global uncertainty remains useful for coarse and
  area-driven failures.
- This reinforces the narrowed thesis rather than weakening it.

## Stop/Proceed Decision

Proceed, but with a narrowed thesis.

The current data do not justify a broad claim that SFRM always dominates global
uncertainty. They do justify a more defensible and more interesting claim:

> Global uncertainty is an incomplete reliability signal. Spatial failure-region
> modeling decomposes failure into boundary uncertainty, topology abnormality,
> and local overconfidence, enabling more interpretable and endpoint-specific
> review prioritization.
