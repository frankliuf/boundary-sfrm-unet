# Manuscript Structure: Scale Heterogeneity in Segmentation Failure Detection

Date: 2026-06-24

This document fixes the scientific narrative for Paper 1. It is intended to
prevent the manuscript from drifting back into a generic "SFRM beats uncertainty"
story. The current experimental evidence supports a stronger and more honest
claim:

> Medical segmentation failures are scale-heterogeneous. Global uncertainty
> captures macro-area failures, whereas spatial failure-region descriptors are
> needed to audit micro-structural boundary and topology failures.

SFRM should therefore be presented as a spatial auditing representation, not as
a universal reliability predictor.

## 1. Title Candidates

Preferred title:

**Beyond Global Uncertainty: Scale Heterogeneity in Medical Image Segmentation Failure Detection**

Rationale:

- Directly foregrounds the scientific finding, not only the method name.
- Sets reviewer expectations that global uncertainty remains part of the story.
- Avoids implying that SFRM is a universally superior predictor.

Alternative titles:

1. **Decomposing Medical Segmentation Failures: Scale Heterogeneity and Spatial Auditing of Overconfident Boundaries**
2. **Spatial Failure-Region Modeling: Auditing Micro-Structural Failures and Feature Dilution in Black-Box Segmentation**
3. **From Macro-Area Collapse to Micro-Structural Failure: Spatial Reliability Auditing for Medical Image Segmentation**

## 2. Core Hypothesis

Existing failure detection methods often reduce a prediction to a single global
uncertainty scalar. This implicitly assumes that segmentation failures are
scale-homogeneous.

This paper challenges that assumption.

Working hypothesis:

> Global uncertainty remains informative for macro-area collapses, where large
> foreground shifts and probability-mass changes dominate the error. However,
> global uncertainty becomes insufficient for micro-structural failures, such as
> boundary leakage, object merging, topology errors, and locally overconfident
> boundary displacement.

## 3. Abstract Logic

The abstract should not follow the conventional "uncertainty is weak, our method
is better" template. It should introduce the scale-heterogeneity hypothesis
before presenting SFRM.

### Abstract Skeleton

Background:

> Reliable deployment of medical image segmentation models requires identifying
> when and where a prediction should be reviewed. Existing failure detection
> strategies commonly summarize predictive uncertainty into a global scalar.

Gap / hypothesis:

> This reduction implicitly assumes that segmentation failures are
> scale-homogeneous. We challenge this assumption by investigating whether
> macro-area failures and micro-structural failures are expressed by different
> reliability signals.

Method:

> We introduce Spatial Failure-Region Modeling (SFRM), a leakage-free spatial
> auditing representation that decomposes segmentation predictions into
> boundary-risk, topology-risk, uncertainty-cluster, and local overconfidence
> descriptors without requiring ground-truth annotations at deployment.

Experiments:

> Using MoNuSeg and CoNSeP nuclei segmentation benchmarks, source-group
> cross-validation, area-controlled review simulations, and qualitative
> mechanism audits, we compare global uncertainty, SFRM descriptors, and
> lightweight reliability predictors under fixed review budgets.

Main finding:

> Global uncertainty remains effective for coarse macro-area failures, whereas
> SFRM provides complementary advantages for boundary-local and
> micro-structural failures, including high boundary-error contribution and
> overconfident boundary displacement.

Final abstract sentence:

> These findings suggest that clinically deployable segmentation auditing should
> move beyond a single confidence score toward scale-aware reliability profiles
> that preserve both macro-level uncertainty and micro-structural failure cues.

Alternative final sentence, more conservative:

> These findings support scale-aware reliability auditing as a practical design
> principle for clinical segmentation deployment, where global uncertainty and
> spatial failure-region descriptors serve complementary roles.

Recommended final sentence:

> These findings support scale-aware reliability auditing as a practical design
> principle for clinical segmentation deployment, where global uncertainty and
> spatial failure-region descriptors serve complementary roles.

## 4. Introduction Ending

The final Introduction paragraph should set the correct expectation before the
reader reaches Results.

Draft:

> Rather than treating segmentation failure detection as a single ranking
> problem, this work investigates whether different failure modes are expressed
> at different spatial scales. We hypothesize that global uncertainty remains
> informative for macro-area failures, where large foreground or probability-mass
> shifts dominate the error, but becomes insufficient for micro-structural
> failures such as boundary leakage, object merging, topology disruption, and
> locally overconfident boundary displacement. To test this hypothesis, we
> introduce Spatial Failure-Region Modeling (SFRM), a leakage-free reliability
> auditing framework that decomposes model failures into boundary-risk,
> topology-risk, uncertainty-cluster, and local overconfidence descriptors. Our
> goal is not to replace global uncertainty with a universal predictor, but to
> expose and quantify the complementary failure signals that emerge when
> segmentation errors become spatially localized.

## 5. Contributions

Contribution 1: Conceptual paradigm

> We propose the concept of scale heterogeneity in medical segmentation failure
> detection, establishing a taxonomy that distinguishes macro-area collapses
> from micro-structural failures.

Contribution 2: Representation framework

> We develop SFRM, a leakage-free spatial auditing representation that
> explicitly parameterizes boundary risk, topology risk, uncertainty clustering,
> and local overconfidence without using ground-truth annotations at deployment.

Contribution 3: Empirical insight

> Through source-group cross-validation, area-controlled review simulations, and
> qualitative audits on MoNuSeg and CoNSeP, we demonstrate that global and
> spatial reliability signals are complementary and reveal a feature dilution
> effect in which global indicators can obscure localized structural error cues.

## 6. Method Opening

The Method section must prevent the reviewer from interpreting SFRM as another
segmentation backbone or a universal confidence regressor.

Draft:

> SFRM is designed as a failure-auditing representation, not as an alternative
> segmentation backbone or a universal confidence regressor. Its objective is to
> preserve spatial topology and local geometric configurations that are
> inevitably lost when a dense prediction is compressed into a single global
> uncertainty scalar. By isolating localized failure regions into explicit
> structural descriptors, SFRM provides a deployable parameter space for auditing
> failures that are clinically meaningful but statistically subtle at the
> whole-patch level.

## 7. Results Structure

### 4.3 Macro-area failures are well captured by global uncertainty

Purpose:

- Show intellectual honesty.
- Establish that global uncertainty remains useful in its proper regime.

Main endpoints:

- `bad_dice_lt_0.65`
- high total error area
- foreground-area dominated CoNSeP gray-zone behavior

Main evidence:

- CoNSeP point, `bad_dice_lt_0.65`, 10% review budget:
  - global logistic features: recall 0.2000
  - SFRM logistic features: recall 0.1765
- CoNSeP prototype, `bad_dice_lt_0.65`, 10% review budget:
  - global logistic features: recall 0.2563
  - best SFRM model: recall 0.2462
- CoNSeP gray-zone labels remain strongly associated with global/area features.

Interpretation:

> Coarse Dice collapse is fundamentally related to large-scale probability-mass
> and foreground-area shifts. A global scalar can be highly effective for this
> macro-area failure regime.

### 4.4 Micro-structural failures require spatial failure-region modeling

Purpose:

- Present the core SFRM win.
- Show that the method's value emerges in boundary-local and structural failure
  endpoints.

Main endpoints:

- `low_boundary_dice_le_q25`
- `high_lecr_boundary_ge_q0.75`
- selected high-Dice/high-boundary-error gray-zone cases

Main evidence:

- CoNSeP prototype, `low_boundary_dice_le_q25`, 10% review budget:
  - SFRM logistic features: recall 0.3175
  - global max entropy: recall 0.1587
  - source-level bootstrap difference: +0.1587, 95% CI [0.0261, 0.2921]
- CoNSeP point, `high_lecr_boundary_ge_q0.75`, 10% review budget:
  - SFRM logistic features: recall 0.2222
  - global mean entropy: recall 0.1587
  - source-level bootstrap CI crosses zero; use as supporting evidence only
- CoNSeP prototype, `high_lecr_boundary_ge_q0.75`, 10% review budget:
  - SFRM L1 logistic features: recall 0.2222
  - global mean entropy: recall 0.1032
  - source-level bootstrap CI crosses zero; use as supporting evidence only
- MoNuSeg prototype, `low_boundary_dice_le_q25`, 10% review budget:
  - SFRM logistic features: recall 0.3810
  - global mean entropy: recall 0.0238
  - source-level bootstrap difference versus global max entropy: +0.3810,
    95% CI [0.1220, 0.5770]
- MoNuSeg point, `gray_high_dice_high_boundary_error`, 10% review budget:
  - boundary overconfidence: recall 0.6250
  - global mean entropy: recall 0.0000
  - source-level bootstrap CI crosses zero because positives are few and source
    concentration is high; use for mechanism illustration only

Interpretation:

> Micro-structural failures are not reliably summarized by global uncertainty.
> They require preserving boundary, topology, and local overconfidence cues.

Primary statistical endpoint:

- `low_boundary_dice_le_q25`

Supporting mechanistic endpoints:

- `high_lecr_boundary_ge_q0.75`
- `gray_high_dice_high_boundary_error`

### 4.5 Feature dilution: global features can obscure local failure cues

Purpose:

- Turn "all deployable is not always best" into a scientific observation.
- Avoid the simplistic assumption that adding more features always improves
  reliability detection.

Main comparisons:

- SFRM-only vs all deployable features for `low_boundary_dice` and
  `high_lecr_boundary`.
- Global-only vs SFRM-only for macro-area endpoints.

Evidence pattern:

- In several micro-structural endpoints, SFRM-only equals or exceeds
  all-deployable features.
- In coarse `bad_dice` endpoints, global-only features often remain stronger.

Interpretation:

> Reliability signals are not only multi-scale but also endpoint-specific.
> Global features can carry high-variance, area-driven signals that dominate
> lightweight predictors and dilute local structural cues.

Caution:

- Do not overstate feature dilution as universal.
- Phrase it as an observed pattern in micro-structural endpoints.

### 4.6 Area-controlled and source-group validation

Purpose:

- Address leakage and area confounding.
- Provide the methodological defense for pathology patch experiments.

Main evidence:

- All lightweight predictors use 14 source-image groups and 5-fold GroupKFold.
- Each source contributes multiple patches, so source grouping is essential.
- CoNSeP area-controlled review simulation uses five predicted foreground-area
  strata with balanced bin sizes.

Area-controlled CoNSeP evidence:

- CoNSeP point, `high_lecr_boundary_ge_q0.75`, 10% review budget:
  - boundary overconfidence area-ranked: recall 0.1905
  - global mean entropy area-ranked: recall 0.0397
- CoNSeP prototype, `high_lecr_boundary_ge_q0.75`, 10% review budget:
  - boundary overconfidence area-ranked: recall 0.1349
  - global mean entropy area-ranked: recall 0.0317
- CoNSeP gray-zone remains mixed after area control.

Interpretation:

> Area confounding is real. After controlling for foreground area, boundary-local
> SFRM signals remain useful for LECR-style failures, while gray-zone detection
> remains a mixed endpoint and should not be used as the primary claim.

### 4.7 Qualitative evidence of boundary overconfidence

Purpose:

- Close the loop from statistics to mechanism.
- Show that SFRM-selected cases correspond to visible boundary and topology
  errors.

Main evidence:

- MoNuSeg point gray-zone:
  - 8 positives.
  - 5 captured by boundary overconfidence at 10% review.
  - 0 captured by global mean entropy.
  - SFRM-hit/global-miss cases come from 2 source images, with 4 of 5 from the
    same source.
- MoNuSeg prototype gray-zone:
  - 20 positives.
  - 8 captured by SFRM balanced score and missed by global mean entropy.
  - 3 captured by global mean entropy and missed by SFRM.
  - 6 captured by both.
  - SFRM-hit/global-miss cases come from 2 source images, with 5 of 8 from the
    same source.

Required visual components:

- H&E patch with GT and prediction contours.
- Entropy map.
- Deployable low-entropy or overconfidence boundary highlight.
- FP/FN evaluation overlay.
- Optional boundary-normal probability profile curve.

Important wording:

- Do not claim these are universally "low global entropy" cases.
- Do not count adjacent patches from the same source as independent qualitative
  evidence.
- Safer claim:

> Under a fixed review budget, global mean entropy can miss structured local
> boundary failures that are prioritized by spatial overconfidence descriptors.

## 8. Main Tables and Figures

### Main Table 1: Failure-scale taxonomy and representative endpoints

Columns:

- Failure scale
- Endpoint
- Expected dominant signal
- Clinical interpretation
- Primary evidence

### Main Table 2: Fixed 10% review-budget recall

Rows:

- MoNuSeg point / prototype
- CoNSeP point / prototype
- macro-area endpoints
- micro-structural endpoints

Columns:

- Global mean entropy
- Global max entropy
- best global predictor
- best SFRM score / predictor
- all-deployable predictor

### Main Figure 1: Conceptual framework

White background, no icons.

Content:

- Dense prediction map.
- Global scalar compression path.
- SFRM spatial decomposition path.
- Macro-area vs micro-structural failure branches.

### Main Figure 2: Review-budget curves

Show selected endpoints only:

- `bad_dice_lt_0.65`
- `low_boundary_dice_le_q25`
- `high_lecr_boundary_ge_q0.75`

Avoid overcrowding all scores.

### Main Figure 3: Feature dilution / feature-family comparison

Show:

- global-only
- SFRM-only
- all-deployable

Use endpoint-specific panels.

### Main Figure 4: Boundary overconfidence qualitative audit

White background.

No decorative icons.

Panels:

- image + contours
- entropy map
- SFRM boundary overconfidence highlight
- FP/FN overlay
- probability profile curve if available

## 9. Supplementary Materials

Move to supplement:

- Full univariate AUROC tables.
- Full predictor sweeps across all endpoints.
- Area-controlled results for every score.
- All diagnostic candidate images.
- Sensitivity to review budget beyond 10%.

## 10. Banned Claims

Do not write:

> SFRM is a universally superior failure detection framework for medical image
> segmentation.

Use:

> SFRM provides a complementary spatial auditing paradigm for micro-structural
> failures that are poorly summarized by global uncertainty.

Do not write:

> SFRM comprehensively improves prediction of coarse Dice failure.

Use:

> Coarse Dice failure remains largely governed by global probability-mass and
> foreground-area signals, whereas SFRM contributes primarily to boundary,
> topology, and local structural failure detection.

Do not write:

> Fusing all deployable and global features yields the strongest reliability
> predictor.

Use:

> We evaluate global, spatial, and combined feature families and observe that
> naive feature fusion can dilute local structural cues in micro-failure
> endpoints.

Do not write:

> Boundary overconfidence always corresponds to low global entropy.

Use:

> Boundary overconfidence can prioritize structured local failures that are
> missed by global mean entropy under a fixed review budget.

Do not write:

> SFRM substantially outperforms all global reliability predictors.

Use:

> SFRM consistently exposes micro-structural boundary failures missed by
> conventional global entropy, while comparisons against trained global-feature
> predictors show a smaller but still directionally useful complementary signal.

## 11. Updated Evidence Hierarchy After Multi-seed and External Validation

Primary result:

- `low_boundary_dice_le_q25`.
- This endpoint is now the most stable across:
  - MoNuSeg point/prototype, seeds 7/42/123.
  - CoNSeP point/prototype, seeds 7/42/123.
  - External 3D FeTS/BraTS cached artifacts.

Primary comparator:

- Conventional global uncertainty:
  - `global_max_entropy` for low-boundary-quality screening.
  - `global_mean_entropy` for LECR-style screening.

Required strong-baseline caveat:

- Trained global-feature predictors are stronger than raw entropy.
- Against trained global predictors, SFRM is positive on average but not
  uniformly significant.
- Therefore the manuscript should argue for a complementary spatial failure
  representation, not a universal replacement for global reliability models.

Supporting result:

- `high_lecr_boundary_ge_q0.75`.
- Multi-seed trends are mostly positive, but confidence intervals often cross
  zero.
- Use this endpoint to explain local critical errors, not as the main proof.

External validation:

- Use the 44-case FeTS/BraTS NPZ audit as an external mechanism check.
- It supports `low_boundary_dice_le_q25`:
  - SFRM recall at 10% budget: 0.3636.
  - global max entropy: 0.0000.
  - source-bootstrap difference: +0.3636, CI [+0.1818, +0.6667].
- Do not use this as a full benchmark because the external set is small and has
  no `bad_dice_lt_0.65` positives.

## 12. Current Stop/Proceed Decision

Proceed to result-table construction and qualitative evidence selection.

Do not run additional broad experiments unless the manuscript attempts a claim
stronger than the evidence above.

Immediate next steps:

1. Build final result tables from existing CSVs.
2. Select main qualitative cases with source-level diversity.
3. Generate white-background, no-icon figures after qualitative cases are fixed.
4. Draft Abstract, Introduction, Method opening, and Results sections according
   to this structure.
