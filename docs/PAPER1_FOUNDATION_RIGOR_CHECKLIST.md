# Paper 1 Foundation-Grade Rigor Checklist

Date: 2026-06-24

This paper is intended to become the foundation for a follow-up research
program. Therefore, the standard is higher than "a publishable set of
experiments". The current evidence is promising but not yet foundation-grade.

## Current Status

Supported but not final:

- SFRM is useful for boundary-local and micro-structural failure endpoints.
- Global uncertainty remains strong for macro-area failure endpoints.
- The overall story should be scale heterogeneity, not universal superiority.

## Stage 4 Source-Level Robustness Update

Source-level bootstrap over 14 source images was run for the key 10% review
budget comparisons.

Output:

- `experiments/summaries/stage4_bootstrap_all_runs/source_bootstrap_comparisons_all.csv`

Comparisons with 95% bootstrap CI excluding zero:

| Dataset | Model | Endpoint | Comparison | Recall difference | 95% CI |
|---|---|---|---|---:|---|
| MoNuSeg | point | `high_lecr_boundary_ge_q0.75` | SFRM L1 logistic vs global mean entropy | +0.3095 | [0.1818, 0.4546] |
| MoNuSeg | prototype | `low_boundary_dice_le_q25` | SFRM L2 logistic vs global max entropy | +0.3810 | [0.1220, 0.5770] |
| CoNSeP | prototype | `low_boundary_dice_le_q25` | SFRM L2 logistic vs global max entropy | +0.1587 | [0.0261, 0.2921] |

Comparisons that remain suggestive but not source-robust at 95% CI:

- CoNSeP point, `high_lecr_boundary_ge_q0.75`: +0.0556, CI crosses zero.
- CoNSeP prototype, `high_lecr_boundary_ge_q0.75`: +0.1190, CI crosses zero.
- MoNuSeg point gray-zone overconfidence: +0.6250, CI crosses zero because
  positives are few and source concentration is high.
- Macro-area `bad_dice_lt_0.65`: global features are competitive, but source
  bootstrap does not support a strong directional claim.

Implication:

- The manuscript's central quantitative claim should be `low_boundary_dice`,
  especially CoNSeP prototype and MoNuSeg prototype.
- `high_lecr_boundary` can remain an important mechanistic endpoint, but the
  CoNSeP evidence should be described as trend-level unless additional runs
  strengthen it.
- Gray-zone qualitative evidence is useful for mechanism illustration, not for
  a primary statistical claim.

## Stage 4 Qualitative Diversity Update

Existing diagnostic cases are source-concentrated:

- MoNuSeg point SFRM-hit/global-miss gray-zone cases:
  - 5 patches from 2 source images.
  - 4 of 5 patches are from the same source.
- MoNuSeg prototype SFRM-hit/global-miss gray-zone cases:
  - 8 patches from 2 source images.
  - 5 of 8 patches are from the same source.

Implication:

- Current qualitative figures should be framed as regional mechanism examples.
- They are not sufficient as source-diverse qualitative proof.
- Before final figures, select at most one patch per source or explicitly state
  that adjacent patches demonstrate a recurring regional failure pattern.

Not yet fully locked:

- Whether the strongest SFRM results remain significant under source-level
  resampling.
- Whether qualitative cases are sufficiently diverse across source images.
- Whether the feature dilution claim is robust rather than cherry-picked.
- Whether single-seed results are enough for a foundation paper.

## Minimum Evidence Required Before Manuscript Drafting

### 1. Source-level robustness

Requirement:

- Run bootstrap or paired source-level analysis over source images, not only
  patches.
- Report confidence intervals for the key 10% review-budget differences.

Core comparisons:

- CoNSeP prototype, `low_boundary_dice_le_q25`:
  - SFRM logistic vs global max entropy.
- CoNSeP point/prototype, `high_lecr_boundary_ge_q0.75`:
  - SFRM logistic/L1 vs global mean entropy.
- MoNuSeg prototype, `low_boundary_dice_le_q25`:
  - SFRM logistic vs global mean entropy.
- Macro-area sanity check:
  - global-only vs SFRM-only on `bad_dice_lt_0.65`.

Pass condition:

- Key micro-structural comparisons should have positive bootstrap median
  difference and a confidence interval that is not dominated by zero.
- If the 95% CI crosses zero, the result can still be discussed, but not as a
  central claim.

### 2. Source-diverse qualitative evidence

Requirement:

- Qualitative examples must come from multiple source images.
- Adjacent patches from the same source can show a regional mechanism, but they
  cannot be counted as independent visual evidence.

Pass condition:

- At least 3 source-diverse examples for the main boundary-overconfidence
  figure or a clear statement that the figure shows a regional failure pattern.

### 3. Feature dilution audit

Requirement:

- Compare global-only, SFRM-only, and all-deployable features by endpoint.
- Do not claim universal dilution.

Pass condition:

- Feature dilution can be claimed only for endpoints where SFRM-only is equal to
  or better than all-deployable under the same model family and validation
  protocol.

### 4. Seed and split sensitivity

Requirement:

- Check whether existing outputs contain multiple seeds.
- If not, decide whether the paper needs at least one additional seed for the
  reliability audit layer.

Pass condition:

- Either:
  - multiple seeds support the main trend; or
  - the manuscript explicitly frames current results as source-group robustness
    over fixed trained models and lists multi-seed model training as a
    limitation.

Status after 2026-06-24 extension:

- Passed for the primary `low_boundary_dice_le_q25` endpoint.
- MoNuSeg and CoNSeP point/prototype models were audited across seeds
  7/42/123.
- SFRM vs conventional global max entropy is positive in 11/12 runs, with mean
  10% recall difference +0.2123.
- SFRM vs trained global-feature logistic is positive in 8/12 runs, with mean
  10% recall difference +0.0470. This supports complementarity, not universal
  dominance.

### 5. Endpoint hierarchy

Main endpoints:

- `low_boundary_dice_le_q25`

Supporting endpoints:

- `high_lecr_boundary_ge_q0.75`

Secondary endpoints:

- `bad_dice_lt_0.65`
- `gray_high_dice_high_boundary_error`

Do not use `gray_high_dice_high_boundary_error` as the central proof because it
is area/foreground-confounded on CoNSeP.

Do not use `high_lecr_boundary_ge_q0.75` as the central proof unless additional
data make the confidence intervals consistently exclude zero. It currently
supports mechanism interpretation but not the main statistical claim.

### 6. External data sanity validation

Requirement:

- If external cached predictions are available, test whether the main
  micro-structural endpoint survives outside pathology patches.

Status:

- Completed on 44 cached FeTS/BraTS 3D segmentation artifacts from three sites.
- SFRM vs global max entropy on `low_boundary_dice_le_q25`:
  - 10% recall 0.3636 vs 0.0000.
  - source-bootstrap difference +0.3636, 95% CI [+0.1818, +0.6667].
- SFRM vs trained global features:
  - 10% recall 0.3636 vs 0.1818.
  - directionally positive but CI crosses zero.

Interpretation:

- Use as external mechanism validation.
- Do not claim full external benchmark superiority because the set is small and
  has no `bad_dice_lt_0.65` positives.

## Decision Rule

Proceed to formal manuscript writing only if:

- Source-level robustness supports the core micro-structural claims.
- Qualitative evidence is source-diverse or clearly labeled as regional.
- Feature dilution is described endpoint-specifically.
- The limitations around single-seed segmentation models are explicitly handled.

Current decision:

- Proceed to final result-table construction and source-diverse qualitative case
  selection.
- Do not strengthen the claim beyond boundary-local failure screening unless new
  evidence is added.
