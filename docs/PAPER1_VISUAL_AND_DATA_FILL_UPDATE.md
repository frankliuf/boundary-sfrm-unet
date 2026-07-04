# Paper 1 Visual Storytelling and Data-Filling Update

Date: 2026-06-25

## Figure contract

Core conclusion:

> SFRM restores spatial boundary-failure information that global uncertainty discards when dense predictions are compressed into a scalar reliability score.

Evidence hierarchy:

1. Figure 1 establishes the conceptual contrast between global scalar compression and SFRM spatial decomposition.
2. Figure 2 quantifies the primary low-boundary-quality recall advantage over conventional global max entropy and the weaker but positive comparison against trained global features.
3. Figure 3 shows fixed-budget review behavior across review fractions.
4. Figure 4 provides source-diverse visual evidence that SFRM-hit/global-miss cases contain localized boundary failures.
5. Supplementary Figure S1 provides a small external 3D mechanism check, not a main benchmark.

## Updated figures

Main figures:

- `figures/paper1/fig1_sfrm_framework.pdf`
- `figures/paper1/fig2_primary_low_boundary_results.pdf`
- `figures/paper1/fig3_review_budget_curves.pdf`
- `figures/paper1/fig4_source_diverse_qualitative.pdf`

Supplementary figure:

- `figures/paper1/fig4_external_3d_validation.pdf`

All main quantitative figures were exported as PDF, SVG, TIFF, and PNG.

## Figure 4 case-selection correction

The original qualitative selection prioritized SFRM/global score gap and boundary loss, which allowed some very low-Dice cases into CoNSeP. This weakened the overconfident/local-failure story.

Updated logic:

- CoNSeP prototype: require Dice >= 0.45.
- MoNuSeg prototype: require Dice >= 0.50.
- Keep only SFRM-hit/global-miss cases for `low_boundary_dice_le_q25`.
- Enforce source diversity: one selected patch per source where possible.

Current qualitative evidence:

- CoNSeP: 4 cases from 4 distinct source images.
- MoNuSeg: 4 cases from 4 distinct source images.
- Combined main Figure 4 uses two CoNSeP and two MoNuSeg rows.

## Data filling

New statistics table:

- `experiments/summaries/stage7_results_statistics/source_bootstrap_pvalues.csv`

Contents:

- 36 key comparisons.
- 12 pathology runs.
- Endpoints:
  - `low_boundary_dice_le_q25`
  - `high_lecr_boundary_ge_q0.75`
- Comparators:
  - conventional global max entropy,
  - conventional global mean entropy,
  - trained global-feature predictor.
- Statistics:
  - recall at 10% review budget,
  - recall difference,
  - 95% source-bootstrap CI,
  - two-sided bootstrap p value.

## Results text updated

The manuscript Results section now includes:

- macro-area failure caveat,
- primary low-boundary-Dice results with CI and p values,
- strong global-predictor caveat,
- LECR-boundary support with CI and p values,
- source-diverse qualitative interpretation,
- supplementary external 3D mechanism check.

Updated manuscript:

- `manuscript/paper1_sfrm_audit_draft.md`

## Remaining visual risks

- Final visual inspection in the desktop image viewer was not available through the sandbox image helper, so the next manual QA step should be opening the exported PDFs directly.
- Figure 4 TIFF is large because it contains raster histology panels; use PDF/SVG for manuscript assembly and TIFF only for submission upload if required.
