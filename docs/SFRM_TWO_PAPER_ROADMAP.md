# SFRM Two-Paper Roadmap

## Core Decision

Do not merge the audit paper and the repair paper into one oversized study.

Use a two-paper sequence:

1. Paper 1: SFRM as a plug-in spatial audit layer.
2. Paper 2: SFRM-guided feedback repair for U-Net.

This keeps the first paper scientifically focused while making its downstream
application path explicit.

## Paper 1: SFRM Audit

### Main Question

Can structured spatial failure signals be extracted from the output of existing
medical segmentation models, and do these signals reveal failures missed by
global uncertainty?

### Core Contribution

SFRM converts segmentation output into interpretable failure descriptors:

- boundary-risk descriptors,
- topology-risk descriptors,
- uncertainty-cluster descriptors,
- local overconfidence descriptors,
- foreground/component geometry descriptors.

### Main Claim

SFRM is a plug-in audit layer that can be attached after U-Net, nnU-Net,
SwinUNETR, SAM/MedSAM, or other segmentation models without retraining the base
model.

The strongest supported claim is:

> SFRM complements global uncertainty by exposing micro-structural boundary
> failures that are poorly captured by global entropy.

### Required Application Bridge

Paper 1 must include a short but concrete section:

> From audit to feedback: using SFRM for model optimization.

This section should not claim that Paper 1 already solves repair. It should
explain that SFRM produces model-actionable signals:

- risk maps,
- boundary failure regions,
- topology abnormality regions,
- local overconfidence regions,
- high-risk review candidates.

These signals can be used to optimize U-Net in two ways:

1. as training weights for difficult regions;
2. as extra feedback channels for a repair network.

### Paper 1 Experiments

Primary experiments:

- SFRM feature extraction from existing U-Net/prototype outputs.
- Feature-discrimination audit.
- Review-budget simulation.
- Source-level bootstrap.
- Multi-seed robustness.
- External 3D feasibility validation.

Optional but useful application teaser:

- A very small proof-of-concept showing that high-SFRM-risk regions are exactly
  where a repair module should focus.
- Do not include full U-Net repair training in Paper 1 unless the results are
  already mature.

### Paper 1 Ending

The conclusion should explicitly state:

> Because SFRM produces localized, model-actionable failure maps, it naturally
> enables a feedback pathway for segmentation model optimization. This motivates
> SFRM-guided repair networks, explored in the subsequent study.

## Paper 2: SFRM-Guided U-Net Repair

### Main Question

Can SFRM feedback signals be used to improve U-Net segmentation performance and
reliability?

### Core Contribution

Paper 2 turns SFRM from an audit layer into a feedback-control layer.

Pipeline:

```text
Image
  -> baseline U-Net
  -> initial probability
  -> SFRM audit maps
  -> repair U-Net / gated decoder
  -> corrected segmentation
```

### Main Claim

SFRM feedback improves U-Net by selectively repairing high-risk boundary and
topology regions, rather than blindly increasing global model capacity.

### Required Baselines

Paper 2 must include strict controls:

- baseline U-Net,
- two-pass U-Net without SFRM,
- U-Net + entropy feedback,
- U-Net + random risk map,
- U-Net + SFRM boundary map,
- U-Net + full SFRM maps.

These baselines are essential. Without them, reviewers can argue that any gain
comes from extra capacity or extra input channels rather than SFRM.

### Main Metrics

Primary metrics:

- Boundary Dice,
- AJI/PQ for dense nuclei,
- high-risk-region Dice,
- SFRM post-repair risk reduction.

Secondary metrics:

- global Dice,
- IoU,
- false-positive leakage,
- topology/component error.

The ideal result pattern:

- global Dice improves slightly or remains stable,
- Boundary Dice improves clearly,
- AJI/PQ improves,
- SFRM risk decreases after repair,
- improvements concentrate in initially high-risk regions.

## Relationship Between the Two Papers

Paper 1 should not look like an incomplete prelude. It must stand alone as a
complete audit and reliability paper.

Paper 2 should not repeat Paper 1. It should cite Paper 1's audit results as the
motivation and then focus on the feedback-repair mechanism.

The relationship is:

```text
Paper 1: find and explain failures
Paper 2: use those failure signals to repair U-Net
```

## Strategic Benefit

This two-paper design has three advantages:

1. Paper 1 has a clean scientific contribution and can be defended without a
   large repair system.
2. Paper 1 still has clear application value because it explains how SFRM
   produces actionable feedback maps for U-Net optimization.
3. Paper 2 becomes a natural continuation rather than a disconnected new idea.

## Writing Guardrails

For Paper 1, do not write:

> We propose a complete segmentation optimization framework.

Write:

> We propose a plug-in spatial audit layer that exposes model-actionable failure
> regions and establishes the basis for feedback-driven segmentation repair.

For Paper 2, do not write:

> We simply add SFRM maps as extra channels.

Write:

> We close the loop between failure diagnosis and segmentation optimization by
> using SFRM-derived spatial feedback to guide U-Net repair.
