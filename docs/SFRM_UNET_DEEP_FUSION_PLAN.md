# SFRM-U-Net Deep Fusion Plan

## 1. Research Purpose

The goal is to move SFRM from a plug-in audit layer to a failure-aware U-Net
optimization layer.

Paper 1 asks:

> Can SFRM discover structured segmentation failures from existing model
> outputs?

This branch asks:

> Can SFRM feed structured failure signals back into U-Net so that the model
> reduces boundary, topology, and local overconfidence failures?

The target is not only higher global Dice. The target is a more reliable U-Net:

- higher Boundary Dice,
- fewer topology defects,
- better object separation,
- lower local failure-risk score,
- no unacceptable loss of global Dice.

## 2. Core Hypothesis

Standard U-Net optimizes segmentation from image features alone. It does not
explicitly know where its own prediction is structurally unreliable.

SFRM provides a second-order signal derived from the prediction itself:

- boundary abnormality,
- topology instability,
- uncertainty clustering,
- local overconfidence,
- foreground-area and component irregularity.

If these structured failure signals are fed back into U-Net, the model should
learn a repair behavior:

> preserve confident easy regions, and selectively revise high-risk boundary or
> topology regions.

## 3. Three Fusion Levels

### Level 1: Post-hoc Refinement U-Net

Architecture:

```text
Image + initial probability + SFRM maps
        -> lightweight refinement U-Net
        -> corrected probability
```

Input channels:

- original RGB or grayscale image,
- initial U-Net probability,
- prediction boundary map,
- boundary risk map,
- topology/connected-component risk map,
- uncertainty or entropy map.

Advantages:

- easiest to implement,
- does not disturb the baseline U-Net,
- can reuse existing checkpoints,
- good first experiment.

Limitation:

- this is still a second-stage repair module, not true internal U-Net fusion.

### Level 2: SFRM-Gated U-Net Decoder

Architecture:

```text
Encoder features
      |
Decoder block k ---- SFRM gate map
      |
Risk-aware decoder features
      |
Segmentation head
```

The SFRM map is downsampled to each decoder resolution and used as a spatial
gate:

```text
F'_k = F_k * (1 + alpha * G_k)
```

where:

- `F_k` is the decoder feature map,
- `G_k` is the SFRM risk gate at the same resolution,
- `alpha` is a learnable or fixed gate strength.

Purpose:

- amplify feature processing in high-risk boundary/topology regions,
- avoid unnecessary changes in easy background or stable foreground.

Advantages:

- deeper integration than post-hoc refinement,
- still simple and interpretable,
- fits U-Net naturally.

Risk:

- SFRM maps depend on an initial prediction. This requires either a two-pass
  U-Net or teacher-student training.

### Level 3: Iterative SFRM-U-Net

Architecture:

```text
Pass 1: U-Net(image) -> probability_1
SFRM(probability_1) -> risk maps
Pass 2: U-Net(image, probability_1, risk maps) -> probability_2
```

The same U-Net can be extended with extra input channels for pass 2, or a small
shared-weight refinement branch can be added.

This is the most conceptually complete version:

> U-Net first predicts, SFRM diagnoses, U-Net repairs.

Advantages:

- clean closed-loop story,
- directly aligned with the "notify U-Net" requirement,
- naturally supports visualizing before/after failure regions.

Risk:

- more moving parts,
- needs careful ablation to prove gains come from SFRM, not merely from adding
  a second U-Net pass.

## 4. Recommended First Architecture

Use Level 3 as the final research target, but implement it in a conservative
two-stage way:

```text
Frozen or pretrained baseline U-Net
        -> initial probability
        -> SFRM map extraction
        -> compact repair U-Net
        -> corrected probability
```

This gives the cleanest development path:

1. keep the original U-Net unchanged,
2. verify SFRM maps improve repair,
3. later merge the repair branch into a deeper gated U-Net if results justify it.

The first working model can be called:

> SFRM-Repair U-Net

## 5. Training Targets

The repair module should not be trained only with Dice or BCE. It needs losses
that match the SFRM purpose.

Recommended loss:

```text
L = L_seg + lambda_b * L_boundary + lambda_r * L_risk_weighted + lambda_t * L_topology
```

Where:

- `L_seg`: standard BCE + Dice loss.
- `L_boundary`: boundary Dice or boundary BCE.
- `L_risk_weighted`: segmentation loss weighted higher in SFRM high-risk
  regions.
- `L_topology`: lightweight component/fragment penalty if stable enough.

The most important term is `L_risk_weighted`:

```text
L_risk_weighted = mean((1 + beta * R) * BCE(pred, gt))
```

where `R` is the SFRM risk map from the initial prediction.

## 6. Required Ablations

To prove SFRM is useful, the comparison cannot be only baseline U-Net vs
SFRM-U-Net.

Minimum ablations:

| Model | Purpose |
|---|---|
| Baseline U-Net | original reference |
| Two-pass U-Net without SFRM | controls for extra capacity/pass |
| U-Net + entropy map | controls for uncertainty-only feedback |
| U-Net + random risk map | controls for extra input channels |
| U-Net + SFRM boundary map only | tests boundary contribution |
| U-Net + full SFRM maps | proposed model |

Primary metrics:

- Dice,
- Boundary Dice,
- AJI/PQ for nuclei if using pathology,
- SFRM post-repair risk score,
- high-risk-region Dice,
- failure-recall reduction under fixed review budget.

## 7. Expected Positive Pattern

The ideal result is not necessarily a large Dice jump.

A strong and believable result is:

- global Dice improves slightly or stays similar,
- Boundary Dice improves clearly,
- AJI/PQ improves on dense nuclei,
- false-positive leakage in confounder regions decreases,
- SFRM risk score decreases after repair,
- high-risk regions show larger gains than low-risk regions.

This pattern would prove that SFRM does not simply add capacity. It directs
U-Net to repair the exact regions it diagnosed.

## 8. Failure Conditions

Stop or revise if:

- Two-pass U-Net without SFRM performs the same as SFRM-U-Net.
- Entropy-only feedback performs the same as full SFRM.
- Gains appear only in global Dice but not boundary/topology endpoints.
- SFRM risk maps are too noisy and cause overcorrection.
- The repair module suppresses true positives near difficult boundaries.

## 9. First Experiment

Dataset:

- start with MoNuSeg because it is smaller and boundary/topology failures are
  visually clear.

Initial model:

- use existing point-supervised U-Net and prototype U-Net checkpoints.

Experiment:

1. Generate initial probability maps.
2. Generate SFRM risk maps.
3. Train a compact repair U-Net on:
   - image,
   - initial probability,
   - entropy,
   - SFRM boundary/topology maps.
4. Compare against entropy-only and two-pass controls.

Decision rule:

- Continue if full SFRM improves Boundary Dice and AJI/PQ more than both
  entropy-only and two-pass controls.
- If only Dice improves, the method is not aligned with SFRM's core value.

## 10. Paper Positioning

This should be Paper 2, not Paper 1.

Paper 1:

- SFRM as a plug-in audit layer.
- Main claim: structured failure discovery.

Paper 2:

- SFRM-guided U-Net repair.
- Main claim: structured failure feedback improves segmentation reliability.

Do not merge the two papers unless Paper 1 is too weak alone. Merging would make
the story too large and harder to defend.
