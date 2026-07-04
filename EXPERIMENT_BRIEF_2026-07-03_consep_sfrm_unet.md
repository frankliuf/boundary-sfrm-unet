# CoNSeP SFRM-UNet Experiment Brief (2026-07-03)

## Goal

Audit whether the current `Boundary-SFRM-UNet` feedback loop truly improves U-Net on CoNSeP, or whether observed gains are mainly caused by the generic two-pass refinement structure.

## Fixed comparison protocol

- Baseline: fully supervised U-Net
- External dataset: CoNSeP
- Selection rule: best validation epoch by `refined_boundary_dice` for two-pass variants, best `boundary_dice` for baseline
- Same baseline checkpoint:
  - `experiments/boundary_sfrm_runs/consep_fullsup_unet_seed42/best_boundary_model.pt`

## Main comparison table

| Run | Best epoch | Dice | Boundary Dice | AJI | PQ | Confounder FPR | Delta Dice | Delta Boundary | Delta AJI | Delta PQ | Delta Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 7 | 0.7315 | 0.6236 | 0.3762 | 0.3000 | 0.4461 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| two_pass_no_risk | 7 | 0.7593 | 0.6439 | 0.3568 | 0.2954 | 0.4904 | +0.0278 | +0.0203 | -0.0194 | -0.0046 | +0.0444 |
| entropy_only | 8 | 0.7502 | 0.6399 | 0.3816 | 0.3187 | 0.4706 | +0.0187 | +0.0163 | +0.0054 | +0.0187 | +0.0245 |
| teacher10 | 6 | 0.7508 | 0.6359 | 0.3722 | 0.2970 | 0.5179 | +0.0193 | +0.0123 | -0.0040 | -0.0029 | +0.0719 |
| contactrestrict_05 | 4 | 0.7510 | 0.6361 | 0.3716 | 0.3078 | 0.5028 | +0.0195 | +0.0125 | -0.0046 | +0.0078 | +0.0567 |
| contactrestrict_025 | 8 | 0.7584 | 0.6427 | 0.3686 | 0.3011 | 0.5106 | +0.0268 | +0.0190 | -0.0075 | +0.0011 | +0.0645 |
| contactrestrict_05_mix07 | 8 | 0.7548 | 0.6379 | 0.3439 | 0.2964 | 0.5238 | +0.0232 | +0.0143 | -0.0323 | -0.0036 | +0.0777 |

## What is now clear

1. `two_pass_no_risk` explains most of the raw Dice / boundary Dice gain on CoNSeP.
2. Original `teacher10` is too aggressive:
   - object-level metrics do not improve;
   - `confounder_fpr` worsens the most.
3. `entropy_only` is the strongest CoNSeP variant overall:
   - better AJI/PQ than baseline;
   - smallest confounder penalty among non-baseline two-pass variants.
4. The best SFRM repair variant so far is `contactrestrict_05`:
   - it improves over original `teacher10` on PQ and confounder FPR;
   - it still does not surpass `entropy_only`.

## Interpretation

The CoNSeP evidence currently supports a constrained claim:

- SFRM-style failure feedback is not yet a universally superior external-data repair signal.
- On CoNSeP, naive failure supervision over-amplifies dense confounder structure.
- Constraining the structural failure target to contact/boundary neighborhoods partially fixes this issue.
- The current best external-dataset narrative is:
  - two-pass refinement boosts pixel overlap;
  - entropy guidance is the most stable external regularizer;
  - structured failure feedback remains promising, but requires tighter structural localization to match entropy on dense pathology scenes.

## Recommended manuscript use

- Keep MoNuSeg as the primary evidence that SFRM feedback can genuinely optimize U-Net.
- Use CoNSeP as an external stress test showing:
  - why naive failure feedback is unsafe;
  - why contact-restricted structural signals are necessary;
  - why dense pathology confounders remain the main generalization bottleneck.

## Relevant run directories

- `experiments/boundary_sfrm_runs/consep_fullsup_unet_seed42`
- `experiments/boundary_sfrm_runs/consep_fullsup_two_pass_no_risk_seed42`
- `experiments/boundary_sfrm_runs/consep_fullsup_entropy_only_seed42`
- `experiments/boundary_sfrm_runs/consep_fullsup_learned_failure_head_v3_teacher10_seed42_rerun`
- `experiments/boundary_sfrm_runs/consep_fullsup_learned_failure_head_v3_teacher10_contactrestrict_gatescale05_seed42`
- `experiments/boundary_sfrm_runs/consep_fullsup_learned_failure_head_v3_teacher10_contactrestrict_gatescale025_seed42`
- `experiments/boundary_sfrm_runs/consep_fullsup_learned_failure_head_v3_teacher10_contactrestrict_gatescale05_mix07_seed42`
