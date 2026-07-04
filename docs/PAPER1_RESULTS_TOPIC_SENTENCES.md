# Paper 1 Results Topic Sentences

Date: 2026-06-25

Purpose: lock the Results section tone before full manuscript polishing. These topic sentences prevent the paper from drifting into an overclaim that SFRM universally beats global uncertainty.

## 4.1 Global uncertainty remains useful for macro-area failures

Topic sentence:

> We first examine coarse segmentation failures and find that global uncertainty and foreground-area features remain highly competitive when the error is dominated by macro-scale probability-mass shifts.

Function:

- Opens with intellectual honesty.
- Establishes that global uncertainty has a valid operating regime.
- Prevents reviewers from thinking the paper is attacking all uncertainty estimation.

Required emphasis:

- Macro-area failure is not SFRM's main target.
- Coarse Dice collapse is often area/probability-mass driven.
- SFRM should not be evaluated as a universal bad-Dice detector.

Avoid:

> SFRM fails on macro failures.

Use:

> Macro-area failures are better represented by global probability-mass and area signals, which motivates a scale-aware rather than replacement-based reliability view.

## 4.2 SFRM exposes low-boundary-quality cases missed by global entropy

Topic sentence:

> In contrast to macro-area failures, low-boundary-quality cases reveal the main blind spot of global entropy: boundary-local structural errors can persist even when the global uncertainty score remains low.

Function:

- Introduces the central positive result.
- Links directly to scale heterogeneity.
- Makes `low_boundary_dice_le_q25` the primary endpoint.

Required evidence:

- CoNSeP prototype: SFRM 0.2989 vs global max entropy 0.1772, mean gain +0.1217, positive 3/3 seeds.
- MoNuSeg prototype: SFRM 0.3016 vs global max entropy 0.0000, mean gain +0.3016, positive 3/3 seeds.
- Across all 12 pathology runs: mean gain +0.2123, positive 11/12 runs.

Avoid:

> SFRM universally outperforms global uncertainty.

Use:

> SFRM consistently improves review prioritization for low-boundary-quality cases compared with conventional global maximum entropy.

## 4.3 Trained global predictors narrow but do not eliminate the SFRM advantage

Topic sentence:

> When global descriptors are themselves trained as reliability predictors, the performance gap narrows, indicating that SFRM should be interpreted as a complementary spatial representation rather than a universal replacement for global reliability models.

Function:

- Adds the strongest caveat.
- Demonstrates fairness to stronger baselines.
- Protects the paper from reviewer criticism that raw entropy is too weak a baseline.

Required evidence:

- Mean SFRM gain over trained global-feature predictor: +0.0470.
- Positive in 8/12 pathology runs.
- MoNuSeg prototype remains the strongest family-level support: +0.0794, positive 3/3 seeds.

Avoid:

> Trained global predictors are weak.

Use:

> Trained global predictors recover part of the signal, but SFRM preserves localized structure and remains directionally complementary.

## 4.4 LECR-boundary analysis supports the local structural-error mechanism

Topic sentence:

> The high-LECR boundary endpoint provides mechanistic support that SFRM is sensitive to locally consequential boundary errors, although its source-level robustness is weaker than the primary low-boundary-Dice endpoint.

Function:

- Uses LECR without overclaiming.
- Explains why the endpoint matters clinically/mechanistically.
- Keeps it secondary.

Required evidence:

- SFRM vs global mean entropy trends positive in most runs.
- Multi-seed source-level significance is not stable enough for a main claim.

Avoid:

> LECR proves SFRM's clinical superiority.

Use:

> LECR analysis supports the interpretation that SFRM captures locally consequential boundary errors.

## 4.5 Source-diverse qualitative evidence localizes the blind spot

Topic sentence:

> Source-diverse qualitative cases show that SFRM-prioritized failures are not merely high-entropy regions, but structured boundary and topology errors that global maximum entropy fails to rank within the review budget.

Function:

- Converts numbers into visible mechanism.
- Justifies Figure 4 as a main figure.
- Emphasizes source-diverse evidence rather than adjacent-patch anecdote.

Required evidence:

- CoNSeP prototype: 4 SFRM-hit/global-miss cases from 4 sources.
- MoNuSeg prototype: 4 SFRM-hit/global-miss cases from 4 sources.
- Each case should show image contours, entropy, SFRM boundary-risk region, and FP/FN error overlay.

Avoid:

> The qualitative figure proves generalization.

Use:

> The qualitative figure illustrates the spatial mechanism behind the quantitative recall gains.

## 4.6 Supplementary external 3D mechanism check

Topic sentence:

> As a supplementary mechanism check, we tested whether the low-boundary-quality signal also appears in cached 3D FeTS/BraTS segmentation artifacts, while deliberately avoiding a broad external-benchmark claim.

Function:

- Keeps external 3D useful but scoped.
- Prevents scope creep.
- Makes it clear why Figure S1 is supplementary.

Required evidence:

- 44 3D cases from three sites.
- SFRM 0.3636 vs global max entropy 0.0000 for `low_boundary_dice_le_q25`.
- CI excludes zero for the entropy comparison.
- SFRM vs trained global features is positive but CI crosses zero.

Avoid:

> SFRM generalizes across all 2D and 3D medical segmentation tasks.

Use:

> The 3D analysis provides preliminary cross-domain mechanism support and motivates larger external validation.

## Overall Results Arc

The Results section should move in this order:

1. Global uncertainty has a valid macro-failure regime.
2. SFRM wins where the paper claims it should win: low-boundary-quality micro-structural failures.
3. Stronger global baselines narrow the gap, forcing a complementarity claim.
4. LECR and qualitative cases explain the boundary-local mechanism.
5. External 3D evidence is supportive and supplementary, not central.

One-sentence Results thesis:

> The experiments support a scale-aware reliability view: global descriptors remain appropriate for macro-area failures, while SFRM restores spatial boundary-failure information that conventional global entropy discards.
