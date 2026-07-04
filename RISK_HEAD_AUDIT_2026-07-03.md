# Risk Head Audit (2026-07-03)

## Purpose

After the first `TNBC` and `CryoNuSeg` main-evidence runs, the learned failure-head branch showed a repeated anomaly:

- `mean_primary_risk` was non-trivial (`~0.33-0.37`)
- but `high_risk_frac` stayed at `0.0` under the fixed evaluation rule `primary_risk >= 0.5`

This audit was run to determine whether the failure head:

1. genuinely failed to learn, or
2. learned a useful ranking but remained under-calibrated.

## Main Findings

### 1. The failure head is not dead

For the stable `freeze8` checkpoints:

- `TNBC`:
  - positive failure pixels mean score: `0.4396`
  - negative pixels mean score: `0.3302`
  - gap: `+0.1094`
- `CryoNuSeg`:
  - positive failure pixels mean score: `0.4409`
  - negative pixels mean score: `0.3652`
  - gap: `+0.0757`

Interpretation:

- the head does separate failure-related pixels from non-failure pixels
- therefore the issue is calibration, not complete feature collapse

### 2. The raw score range is compressed below 0.5

Raw primary-risk summary:

- `TNBC`:
  - mean: `0.3352`
  - q99: `0.4701`
  - max: `0.4763`
- `CryoNuSeg`:
  - mean: `0.3717`
  - q99: `0.4868`
  - max: `0.4895`

Interpretation:

- the fixed threshold `0.5` is too high for this branch in the current training regime
- this alone explains why `high_risk_frac` becomes zero

### 3. Offline calibration restores meaningful risk regions

Audited with:

- raw thresholds: `0.45`, `0.47`
- robust normalized thresholds: `0.80`, `0.90`

#### TNBC

- `raw@0.45`:
  - recall: `0.5478`
  - precision: `0.2677`
  - high-risk frac: `0.0978`
- `norm@0.90`:
  - recall: `0.5701`
  - precision: `0.2548`
  - high-risk frac: `0.0937`

#### CryoNuSeg

- `raw@0.45`:
  - recall: `0.5867`
  - precision: `0.3909`
  - high-risk frac: `0.1262`
- `norm@0.90`:
  - recall: `0.3908`
  - precision: `0.4731`
  - high-risk frac: `0.0694`

Interpretation:

- once the threshold is brought into the correct operating range, the learned failure head produces meaningful spatial audit regions
- the current failure head is better described as **rank-informative but under-calibrated**

## Calibrated Modulation Smoke Test

A new experimental variant was added:

- `learned_failure_head_calibrated`

Design:

- robustly normalize failure-head maps before feeding them to the refinement decoder

Smoke run:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\experiments\boundary_sfrm_runs\smoke_cryonuseg_sfrm_calibrated_seed42`

Observed behavior:

- `high_risk_frac` became non-zero (`0.2349`)
- but segmentation metrics dropped sharply in the smoke run

Interpretation:

- calibration is useful for audit/readout
- but directly injecting the calibrated map into refinement is too aggressive in its current form

## Practical Conclusion

The current evidence supports the following decomposition:

1. `Raw failure-head maps` are acceptable for gentle refinement modulation.
2. `Calibrated failure-head maps` are better for explicit audit/readout.
3. These two uses should be decoupled rather than forced into a single shared representation.

## Outputs

Reusable audit script:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\scripts\audit_failure_head_risk.py`

Audit outputs:

- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\outputs\risk_audit_tnbc_freeze8`
- `D:\paper_MedIA Vol. 107–113\failure_region_reliability\outputs\risk_audit_cryonuseg_freeze8`

## Recommended Next Step

Do **not** replace the refinement input with aggressively calibrated risk maps in the mainline architecture.

Instead, the next clean experiment should be:

- keep the successful `freeze8` segmentation branch unchanged
- add a separate audit/readout calibration rule for the failure head
- report both:
  - segmentation improvement
  - calibrated failure localization quality
