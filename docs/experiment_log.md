# Experiment Log

## 2026-06-24

- Created independent project directory.
- Defined program direction:
  **failure-region-centric reliable medical image analysis under low
  annotation**.
- Decided not to include ChatGPT/LLM modules in the first-stage framework.
- First paper will focus on spatial failure-region reliability modeling rather
  than a new segmentation backbone.
- Added a deep strategy memo for Paper 1 after reviewing current failure
  detection, segmentation quality prediction, spatial uncertainty aggregation,
  OOD, weak-supervision, and local MedIA/CVPR 2026 evidence.
- Corrected an important protocol issue: false-positive and false-negative
  regions require ground truth and must not be used as deployable predictor
  features. They are evaluation/training targets only.
- Integrated expert feedback into the Paper 1 strategy: anatomical/topological
  consistency, calibration, human-review budget simulation, multicollinearity
  audit, and inter-observer ambiguity discussion.
- Set the next execution decision: run feature-discrimination validation before
  training the full reliability predictor.
- Chose MoNuSeg test patches as the first audit dataset because checkpoints and
  per-patch metrics already exist locally. CoNSeP will be the second pathology
  validation dataset; a non-pathology dataset should be added only after the
  MoNuSeg/CoNSeP audit is stable.
- Added the Stage 0 MoNuSeg feature extraction script with a strict cache
  contract: deployable features are separated from audit-only evaluation
  metrics, and leakage-prone terms are blocked by assertion.

### Stage 0A: MoNuSeg point-only seed42 full feature extraction

Purpose:

- Start the SFRM feature-discrimination audit from an already validated
  segmentation checkpoint.
- Use MoNuSeg test patches and the point-only U-Net seed42 checkpoint from the
  confounder-prompting project.
- Save probability, entropy, prediction, ground-truth, and error maps so later
  region-localization and review-budget experiments do not need repeated model
  inference.

Pre-run code checks:

1. Static and path check:
   - `stage0_extract_monuseg_features.py` passed `py_compile`.
   - All required paths exist:
     - `data/monuseg_test_patches/images`
     - `data/monuseg_test_patches/masks`
     - `outputs/confounder_prompting/monuseg_split_point_seed42/best_model.pt`
   - MoNuSeg test set has 168 image patches and 168 mask patches.
2. Smoke run:
   - 5-patch run succeeded and produced CSV, JSONL, and per-patch `.npz` maps.
   - Each `.npz` contains `prob`, `entropy`, `pred`, `gt`, and `error`.
3. Consistency and leakage check:
   - Deployable feature names contain no GT/error/Dice/LECR fields.
   - Initial smoke run exposed a boundary metric mismatch because Stage 0 used a
     disk-shaped morphology footprint while the confounder-prompting evaluation
     used square max-pooling.
   - The boundary definition was corrected to a square footprint of
     `2 * radius + 1`, matching `confounder_mining.metrics._mask_boundary`.
   - After correction, 5-patch Dice and boundary Dice matched the old
     per-patch evaluation within approximately `5e-7`.

Full run command:

```powershell
python scripts\stage0_extract_monuseg_features.py `
  --workspace-root "D:\paper_MedIA Vol. 107-113" `
  --images-dir "D:\paper_MedIA Vol. 107-113\data\monuseg_test_patches\images" `
  --masks-dir "D:\paper_MedIA Vol. 107-113\data\monuseg_test_patches\masks" `
  --checkpoint "D:\paper_MedIA Vol. 107-113\outputs\confounder_prompting\monuseg_split_point_seed42\best_model.pt" `
  --output-csv "experiments\summaries\stage0_monuseg_point_seed42_features_full.csv" `
  --output-jsonl "experiments\summaries\stage0_monuseg_point_seed42_cache_full.jsonl" `
  --output-map-dir "experiments\maps\stage0_monuseg_point_seed42_full" `
  --limit 0 `
  --batch-size 1 `
  --device cuda
```

Note: the actual workspace folder uses the en dash in `Vol. 107–113`; the command
above is ASCII-normalized for readability.

Full run outputs:

- `experiments/summaries/stage0_monuseg_point_seed42_features_full.csv`
- `experiments/summaries/stage0_monuseg_point_seed42_cache_full.jsonl`
- `experiments/maps/stage0_monuseg_point_seed42_full/*.npz`

Full run validation:

- CSV rows: 168.
- CSV columns: 35.
- JSONL lines: 168.
- NPZ files: 168.
- NPZ total size: approximately 68.5 MB.
- Max Dice difference versus previous `test_eval_per_patch.csv`: `5.12e-7`.
- Max boundary Dice difference versus previous `test_eval_per_patch.csv`:
  `5.02e-7`.
- Leakage feature count: 0.

Selected full-run means:

| Metric / feature | Mean | Min | Max |
|---|---:|---:|---:|
| `eval__dice` | 0.660364 | 0.227494 | 0.777843 |
| `eval__boundary_dice` | 0.357695 | 0.099738 | 0.522180 |
| `eval__error_area_frac` | 0.237642 | 0.105453 | 0.754242 |
| `feat__global_uncertainty__mean_entropy` | 0.397683 | 0.287406 | 0.569380 |
| `feat__boundary_risk__boundary_mean_entropy` | 0.640204 | 0.627884 | 0.668239 |
| `feat__topology_risk__component_count` | 17.821429 | 1.000000 | 38.000000 |

### Stage 0B: MoNuSeg point-only seed42 feature-discrimination audit

Purpose:

- Run the first univariate feature-discrimination audit on the Stage 0A feature
  table.
- Compare leakage-free SFRM features against simple global uncertainty features.

Audit script checks:

1. `stage0_audit_feature_discriminability.py` passed `py_compile`.
2. `--help` displayed the expected arguments.
3. A 5-patch smoke audit completed successfully.

Full run command:

```powershell
python scripts\stage0_audit_feature_discriminability.py `
  --features-csv experiments\summaries\stage0_monuseg_point_seed42_features_full.csv `
  --output-dir experiments\summaries\stage0_monuseg_point_seed42_audit_full `
  --bad-dice-threshold 0.65 `
  --top-error-quantile 0.75
```

Full audit outputs:

- `experiments/summaries/stage0_monuseg_point_seed42_audit_full/audit_summary.json`
- `experiments/summaries/stage0_monuseg_point_seed42_audit_full/univariate_auroc.csv`
- `experiments/summaries/stage0_monuseg_point_seed42_audit_full/spearman_correlations.csv`

Audit labels:

| Label | Positives | Negatives |
|---|---:|---:|
| `bad_dice_lt_0.65` | 43 | 125 |
| `low_boundary_dice_lt_median` | 84 | 84 |
| `high_error_ge_q0.75` | 42 | 126 |

Selected audit observations:

- `feat__global_uncertainty__mean_entropy` achieved directional AUROC 0.8374
  for `bad_dice_lt_0.65` and 0.8299 for `low_boundary_dice_lt_median`.
- `feat__topology_risk__component_count` achieved directional AUROC 0.8656 for
  `bad_dice_lt_0.65` and 0.8377 for `low_boundary_dice_lt_median`.
- `feat__uncertainty_cluster__high_uncertainty_largest_component_area_frac`
  achieved directional AUROC 0.8589 for `bad_dice_lt_0.65` and 0.8143 for
  `low_boundary_dice_lt_median`.
- Several area-related features nearly perfectly identify the `high_error`
  top-quartile label. This is useful but must be interpreted cautiously because
  high total error can be coupled to foreground area and over-segmentation.

Immediate implication:

- Stage 0A/0B support continuing SFRM, but the next audit must add more local
  and clinically meaningful labels such as boundary-error concentration,
  topology error, high-Dice/high-risk gray-zone cases, and review-budget capture.

### Stage 0C: MoNuSeg prototype contrastive seed42 extraction and quality gate

Purpose:

- Run the same SFRM extraction on the prototype contrastive checkpoint.
- Test whether SFRM descriptors remain informative after the segmentation model
  is improved, rather than only detecting obvious point-only failures.

Pre-run checks:

1. Required files exist:
   - `outputs/confounder_prompting/monuseg_split_contrastive_w005_seed42/best_model.pt`
   - `outputs/confounder_prompting/monuseg_split_contrastive_w005_seed42/test_eval_per_patch.csv`
2. `stage0_extract_monuseg_features.py` and
   `stage0_audit_feature_discriminability.py` passed `py_compile`.
3. A 5-patch smoke run matched the existing prototype per-patch evaluation:
   - max Dice difference: `3.91e-7`;
   - max boundary Dice difference: `4.23e-7`;
   - leakage feature count: 0.

Full extraction outputs:

- `experiments/summaries/stage0_monuseg_proto_seed42_features_full.csv`
- `experiments/summaries/stage0_monuseg_proto_seed42_cache_full.jsonl`
- `experiments/maps/stage0_monuseg_proto_seed42_full/*.npz`

Full extraction validation:

- CSV rows: 168.
- CSV columns: 35.
- JSONL lines: 168.
- NPZ files: 168.
- NPZ total size: approximately 74.56 MB.
- Max Dice difference versus previous prototype `test_eval_per_patch.csv`:
  `4.95e-7`.
- Max boundary Dice difference versus previous prototype `test_eval_per_patch.csv`:
  `5.01e-7`.
- Leakage feature count: 0.

Selected prototype means:

| Metric / feature | Mean | Min | Max |
|---|---:|---:|---:|
| `eval__dice` | 0.704243 | 0.325388 | 0.837357 |
| `eval__boundary_dice` | 0.459960 | 0.178146 | 0.634170 |
| `eval__error_area_frac` | 0.152743 | 0.064438 | 0.277771 |
| `feat__global_uncertainty__mean_entropy` | 0.389413 | 0.320727 | 0.458929 |
| `feat__boundary_risk__boundary_mean_entropy` | 0.665037 | 0.642538 | 0.689176 |
| `feat__topology_risk__component_count` | 26.773810 | 16.000000 | 41.000000 |

Paired prototype-minus-point deltas:

| Quantity | Mean delta | Median delta | Interpretation |
|---|---:|---:|---|
| `eval__dice` | +0.043878 | +0.044761 | prototype improves overlap |
| `eval__boundary_dice` | +0.102266 | +0.089404 | prototype strongly improves boundary quality |
| `eval__error_area_frac` | -0.084899 | -0.041191 | prototype reduces total error area |
| `eval__lecr_uncertainty` | -0.138173 | -0.091028 | prototype reduces uncertainty concentration in error regions |
| `feat__global_uncertainty__mean_entropy` | -0.008270 | -0.000380 | global mean entropy changes only weakly |
| `feat__boundary_risk__boundary_mean_entropy` | +0.024833 | +0.023950 | remaining failures are more boundary-focused |
| `feat__topology_risk__component_count` | +8.952381 | +8.000000 | requires careful interpretation: more separated objects or more fragments |

Quality-gate interpretation:

- The prototype results are not a blind success story, but they support SFRM.
- Global mean entropy becomes weak for prototype `bad_dice_lt_0.65`
  (directional AUROC about 0.584), while boundary-risk descriptors remain strong.
- This supports a central SFRM argument: after the segmentation model improves,
  global uncertainty is less sufficient and spatial boundary descriptors become
  more informative.

Selected local-label audit results:

| Model | Label | Strongest feature | Directional AUROC | Note |
|---|---|---|---:|---|
| point-only | `bad_dice_lt_0.65` | `topology_risk__largest_component_area_frac` | 0.8826 | topology/cluster features beat mean entropy |
| point-only | `low_boundary_dice_le_q25` | `topology_risk__largest_component_area_frac` | 0.9919 | strong but may partly reflect over-expansion geometry |
| point-only | `high_boundary_error_ge_q0.75` | `boundary_risk__boundary_high_entropy_frac` | 0.9369 | directly supports boundary-risk design |
| point-only | `gray_high_dice_high_boundary_error` | `boundary_risk__boundary_high_entropy_frac` | 0.8750 | small positive count, promising but needs caution |
| prototype | `bad_dice_lt_0.65` | `boundary_risk__boundary_mean_entropy` | 0.9670 | strong support for boundary-risk descriptor |
| prototype | `low_boundary_dice_le_q25` | `boundary_risk__boundary_high_entropy_frac` | 0.9312 | strong support after model improvement |
| prototype | `gray_high_dice_high_boundary_error` | `boundary_risk__boundary_high_entropy_frac` | 0.9824 | excellent but must be validated on CoNSeP |

Risk flags:

- The coarse `high_error_ge_q0.75` label is too easy for area-like features and
  should not be used as the main claim.
- `high_lecr_boundary_ge_q0.75` is weaker, especially for prototype; it may be a
  supplementary diagnostic rather than a primary endpoint.
- The prototype has a higher component count than point-only. This may reflect
  better separation of touching nuclei, but it could also reflect fragmentation.
  CoNSeP and qualitative checks are needed before making a strong object-level
  claim.

### 2026-06-24: CoNSeP Fixed-Grid Stage 0 Audit

Purpose:

- Test whether the SFRM descriptors generalize beyond MoNuSeg to a denser and
  more heterogeneous nuclei dataset.
- Apply the same quality gate: compile check, 5-patch smoke test, metric
  consistency with the existing evaluation CSV, leakage check, full extraction,
  and univariate audit.

#### CoNSeP point-only

Inputs:

- Images: `../data/consep_test_grid_patches/images` with 504 patches.
- Masks: `../data/consep_test_grid_patches/masks` with 504 patches.
- Checkpoint:
  `../outputs/confounder_prompting/consep_grid_point_seed42/best_model.pt`.
- Previous metric CSV:
  `../outputs/confounder_prompting/consep_grid_point_seed42/test_eval_per_patch.csv`.

Outputs:

- `experiments/summaries/stage0_consep_grid_point_seed42_features_full.csv`
- `experiments/summaries/stage0_consep_grid_point_seed42_cache_full.jsonl`
- `experiments/maps/stage0_consep_grid_point_seed42_full/*.npz`
- `experiments/summaries/stage0_consep_grid_point_seed42_audit_full/`

Full extraction validation:

- CSV rows: 504.
- CSV columns: 35.
- JSONL lines: 504.
- NPZ files: 504.
- NPZ total size: approximately 204.96 MB.
- Max Dice difference versus previous `test_eval_per_patch.csv`, by row order:
  `5.17e-7`.
- Max boundary Dice difference versus previous `test_eval_per_patch.csv`, by
  row order: `5.00e-7`.
- Leakage feature count: 0.

Selected point-only means:

| Metric / feature | Mean | Min | Max |
|---|---:|---:|---:|
| `eval__dice` | 0.623596 | 0.000000 | 1.000000 |
| `eval__boundary_dice` | 0.396524 | 0.000000 | 1.000000 |
| `eval__error_area_frac` | 0.139737 | 0.000000 | 0.402069 |
| `feat__global_uncertainty__mean_entropy` | 0.387398 | 0.195754 | 0.582713 |
| `feat__boundary_risk__boundary_mean_entropy` | 0.677307 | 0.000000 | 0.692181 |
| `feat__topology_risk__component_count` | 18.746032 | 0.000000 | 76.000000 |

Selected point-only audit results:

| Label | Strongest feature | Directional AUROC | Note |
|---|---|---:|---|
| `bad_dice_lt_0.65` | `topology_risk__mean_component_area_frac` | 0.8170 | topology features beat global mean entropy |
| `low_boundary_dice_le_q25` | `boundary_risk__pred_boundary_area_frac` | 0.7336 | only moderate separability |
| `high_boundary_error_ge_q0.75` | `topology_risk__threshold_area_frac_std` | 0.9817 | strong but partly reducible to global entropy/area |
| `high_lecr_boundary_ge_q0.75` | `boundary_risk__boundary_mean_margin_uncertainty` | 0.7106 | supports boundary-local signal; global mean entropy is weak |
| `gray_high_dice_high_boundary_error` | `global_uncertainty__foreground_area_frac` | 0.9668 | strong but likely confounded by object/area geometry |

#### CoNSeP prototype contrastive

Inputs:

- Checkpoint:
  `../outputs/confounder_prompting/consep_grid_contrastive_w005_seed42/best_model.pt`.
- Previous metric CSV:
  `../outputs/confounder_prompting/consep_grid_contrastive_w005_seed42/test_eval_per_patch.csv`.

Outputs:

- `experiments/summaries/stage0_consep_grid_proto_seed42_features_full.csv`
- `experiments/summaries/stage0_consep_grid_proto_seed42_cache_full.jsonl`
- `experiments/maps/stage0_consep_grid_proto_seed42_full/*.npz`
- `experiments/summaries/stage0_consep_grid_proto_seed42_audit_full/`

Full extraction validation:

- CSV rows: 504.
- CSV columns: 35.
- JSONL lines: 504.
- NPZ files: 504.
- NPZ total size: approximately 224.37 MB.
- Max Dice difference versus previous `test_eval_per_patch.csv`, by row order:
  `5.17e-7`.
- Max boundary Dice difference versus previous `test_eval_per_patch.csv`, by
  row order: `5.10e-7`.
- Leakage feature count: 0.

Selected prototype means:

| Metric / feature | Mean | Min | Max |
|---|---:|---:|---:|
| `eval__dice` | 0.641072 | 0.000000 | 1.000000 |
| `eval__boundary_dice` | 0.402641 | 0.000000 | 1.000000 |
| `eval__error_area_frac` | 0.129869 | 0.000000 | 0.413620 |
| `feat__global_uncertainty__mean_entropy` | 0.286754 | 0.083626 | 0.497045 |
| `feat__boundary_risk__boundary_mean_entropy` | 0.665678 | 0.000000 | 0.689553 |
| `feat__topology_risk__component_count` | 14.432540 | 0.000000 | 32.000000 |

Selected prototype audit results:

| Label | Strongest feature | Directional AUROC | Note |
|---|---|---:|---|
| `bad_dice_lt_0.65` | `topology_risk__mean_component_area_frac` | 0.8589 | robust sample-level failure signal |
| `low_boundary_dice_le_q25` | `boundary_risk__boundary_mean_entropy` | 0.7097 | boundary family remains top but moderate |
| `high_boundary_error_ge_q0.75` | `global_uncertainty__mean_margin_uncertainty` | 0.9779 | this endpoint is not sufficient to prove SFRM superiority |
| `high_lecr_boundary_ge_q0.75` | `boundary_risk__boundary_mean_entropy` | 0.6982 | boundary-local signal persists; global mean entropy remains weak |
| `gray_high_dice_high_boundary_error` | `global_uncertainty__foreground_area_frac` | 0.9622 | likely affected by area/foreground confounding |

Paired prototype-minus-point deltas:

| Quantity | Mean delta | Median delta | Directional note |
|---|---:|---:|---|
| `eval__dice` | +0.017476 | +0.020829 | improved in 75.60% patches |
| `eval__boundary_dice` | +0.006117 | +0.002424 | improved in 52.18% patches |
| `eval__error_area_frac` | -0.009868 | -0.009651 | decreased in 82.54% patches |
| `eval__boundary_error_area_frac` | -0.007732 | -0.007278 | decreased in 82.14% patches |
| `eval__lecr_boundary_error` | -0.003842 | -0.002400 | decreased in 53.37% patches |
| `eval__lecr_uncertainty` | +0.030370 | +0.026721 | increased in 91.47% patches |
| `feat__global_uncertainty__mean_entropy` | -0.100645 | -0.102300 | decreased in all patches |
| `feat__boundary_risk__boundary_mean_entropy` | -0.011629 | -0.012695 | decreased in 98.61% patches |
| `feat__topology_risk__component_count` | -4.313492 | -2.000000 | decreased in 65.67% patches |

Quality-gate interpretation:

- CoNSeP does not invalidate SFRM, but it requires a more precise claim.
- Stable support exists for:
  - sample-level bad-case detection via topology/geometry descriptors;
  - boundary-local error concentration via boundary-risk descriptors;
  - the idea that model improvement reduces global entropy, making residual
    local failure harder to capture with a single global score.
- The endpoint `high_boundary_error_ge_q0.75` should not be used as a primary
  proof of SFRM superiority because global mean entropy and margin uncertainty
  are also very strong on CoNSeP.
- The gray-zone label is promising but may be confounded by foreground/object
  area. It needs qualitative inspection and, if used, a stricter definition that
  controls for foreground area.
- Next recommended step: build review-budget simulation using multiple risk
  scores, but report it with endpoint-specific nuance rather than claiming that
  SFRM universally dominates global uncertainty.

### 2026-06-24: Stage 1 Fixed Review-Budget Simulation

Purpose:

- Simulate a human-in-the-loop scenario where only a fixed fraction of patches
  can be reviewed.
- Compare global uncertainty rankings with leakage-free SFRM rankings.
- Use evaluation labels only after ranking. The risk scores themselves are
  computed from deployable features only.

Script:

- `scripts/stage1_review_budget_simulation.py`

Outputs:

- `experiments/summaries/stage1_review_budget_monuseg_point_seed42/`
- `experiments/summaries/stage1_review_budget_monuseg_proto_seed42/`
- `experiments/summaries/stage1_review_budget_consep_grid_point_seed42/`
- `experiments/summaries/stage1_review_budget_consep_grid_proto_seed42/`
- `experiments/summaries/stage1_review_budget_all_runs/review_budget_metrics_all.csv`
- `experiments/summaries/stage1_review_budget_all_runs/score_discriminability_all.csv`

Quality-control notes:

- The first implementation assumed that higher boundary uncertainty always
  means higher risk. The MoNuSeg point-only simulation contradicted this for
  `high_lecr_boundary_ge_q0.75`: high local boundary-error contribution can
  also appear as overconfident boundary failure.
- The script was therefore revised to add:
  - `boundary_overconfidence_score`;
  - `boundary_abnormality_score`;
  - `boundary_dual_risk_score`;
  - `sfrm_balanced_score`.
- These scores still use no ground-truth-derived features. They model the fact
  that local failure can arise from both uncertainty and overconfidence.

Selected 10% review-budget findings:

| Dataset | Model | Endpoint | Best SFRM-style score | SFRM recall | Best global score | Global recall | Interpretation |
|---|---|---|---:|---:|---|---:|---|
| MoNuSeg | point | `bad_dice_lt_0.65` | `sfrm_composite_score` | 0.3953 | `global_mean_entropy` | 0.3721 | small SFRM gain |
| MoNuSeg | point | `gray_high_dice_high_boundary_error` | `boundary_overconfidence_score` | 0.6250 | global entropy family | 0.0000 | strong support for overconfidence-aware boundary risk |
| MoNuSeg | point | `high_lecr_boundary_ge_q0.75` | `boundary_overconfidence_score` | 0.1429 | `global_max_entropy` | 0.3095 | SFRM does not dominate this endpoint |
| MoNuSeg | prototype | `bad_dice_lt_0.65` | `boundary_risk_score` | 0.5484 | `global_mean_entropy` | 0.0323 | strong support after model improvement |
| MoNuSeg | prototype | `low_boundary_dice_le_q25` | `boundary_risk_score` | 0.4048 | `global_mean_entropy` | 0.0238 | strong support for boundary-risk score |
| MoNuSeg | prototype | `gray_high_dice_high_boundary_error` | `sfrm_balanced_score` | 0.7000 | `global_mean_entropy` | 0.4500 | SFRM improves gray-zone capture |
| CoNSeP | point | `high_lecr_boundary_ge_q0.75` | `boundary_overconfidence_score` | 0.1825 | `global_mean_entropy` | 0.1587 | modest SFRM gain |
| CoNSeP | prototype | `low_boundary_dice_le_q25` | `boundary_risk_score` | 0.2857 | `global_max_entropy` | 0.1587 | SFRM improves low-boundary-quality capture |
| CoNSeP | point | `gray_high_dice_high_boundary_error` | `sfrm_composite_score` | 0.3690 | `global_mean_entropy` | 0.5357 | global entropy remains stronger |
| CoNSeP | prototype | `gray_high_dice_high_boundary_error` | `sfrm_balanced_score` | 0.3804 | `global_mean_entropy` | 0.4348 | global entropy remains stronger |

Quality-gate interpretation:

- The review-budget experiment supports Paper 1 only if the claim is
  endpoint-specific:
  - SFRM is useful for boundary-local and topology/structure-aware failure
    discovery.
  - SFRM is especially useful after a stronger segmentation model reduces
    coarse global uncertainty, as seen on MoNuSeg prototype.
  - Boundary overconfidence is a distinct failure mode that global mean entropy
    can miss.
- The experiment does not support a blanket claim that a single SFRM composite
  universally dominates every global uncertainty baseline.
- CoNSeP gray-zone labels appear partly confounded by foreground/object area and
  should not be the primary proof without stricter matching or qualitative
  evidence.

### 2026-06-24: Stage 2 Qualitative Mechanism Audit

Purpose:

- Verify whether the "boundary overconfidence" signal corresponds to visible
  structured boundary failures, not only to a statistical artifact.
- Generate diagnostic plots only. These are not final manuscript figures.

Script:

- `scripts/stage2_qualitative_case_audit.py`

MoNuSeg point-only gray-zone audit:

- Output directory:
  `experiments/summaries/stage2_qualitative_monuseg_point_gray_overconfidence/`
- Endpoint: `gray_high_dice_high_boundary_error`.
- SFRM score: `boundary_overconfidence_score`.
- Global score: `global_mean_entropy`.
- Review fraction: 10%.
- Positive cases: 8.
- Category counts:
  - `sfrm_hit_global_miss`: 5.
  - `global_hit_sfrm_miss`: 0.
  - `both_hit`: 0.
  - `both_miss`: 3.
- Rendered diagnostic figures: 5.

MoNuSeg prototype gray-zone audit:

- Output directory:
  `experiments/summaries/stage2_qualitative_monuseg_proto_gray_sfrm_balanced/`
- Endpoint: `gray_high_dice_high_boundary_error`.
- SFRM score: `sfrm_balanced_score`.
- Global score: `global_mean_entropy`.
- Review fraction: 10%.
- Positive cases: 20.
- Category counts:
  - `sfrm_hit_global_miss`: 8.
  - `global_hit_sfrm_miss`: 3.
  - `both_hit`: 6.
  - `both_miss`: 3.
- Rendered diagnostic figures: 8.

Qualitative interpretation:

- The rendered cases show structured FP/FN boundary disagreement and foreground
  probability maps with confident object-like predictions in regions that do not
  match the ground-truth contour.
- However, the strict phrase "low global entropy failure" is not supported by
  the selected gray-zone cases. Their global entropy ranks are not in the top
  10% review set, but they are usually moderate to high in absolute percentile.
- The more accurate claim is therefore:
  "global mean entropy misses some structured local boundary failures under a
  fixed review budget, while spatial boundary/overconfidence descriptors can
  prioritize them."
- Several point-only cases come from adjacent patches of the same source image.
  This is useful for regional mechanism inspection but must not be counted as
  fully independent qualitative evidence.

### 2026-06-24: Stage 2 CoNSeP Area-Controlled Review Simulation

Purpose:

- Test whether CoNSeP results are confounded by foreground/object area.
- Use predicted foreground area as a control variable, split patches into five
  quantile bins, and allocate the same review fraction inside each bin.

Script:

- `scripts/stage2_area_controlled_review.py`

Outputs:

- `experiments/summaries/stage2_area_controlled_consep_grid_point_seed42/`
- `experiments/summaries/stage2_area_controlled_consep_grid_proto_seed42/`

Area bins:

- Both CoNSeP point-only and prototype runs use five nearly balanced bins:
  101, 101, 100, 101, and 101 patches.

Selected 10% review-budget results after area stratification:

| Model | Endpoint | Best SFRM-style score | SFRM recall | Global comparison | Global recall | Interpretation |
|---|---|---|---:|---|---:|---|
| point | `high_lecr_boundary_ge_q0.75` | `boundary_overconfidence_score__area_ranked` | 0.1905 | `global_mean_entropy__area_ranked` | 0.0397 | area control reveals boundary-overconfidence advantage |
| point | `low_boundary_dice_le_q25` | `boundary_overconfidence_score__area_ranked` | 0.1349 | `global_mean_entropy__area_ranked` | 0.0794 | modest SFRM advantage |
| point | `gray_high_dice_high_boundary_error` | `sfrm_balanced_score__area_ranked` | 0.1905 | `global_mean_entropy__area_ranked` | 0.2024 | no clear SFRM win |
| prototype | `high_lecr_boundary_ge_q0.75` | `boundary_overconfidence_score__area_ranked` | 0.1349 | `global_mean_entropy__area_ranked` | 0.0317 | area control supports overconfidence signal |
| prototype | `low_boundary_dice_le_q25` | `boundary_risk_score__area_ranked` | 0.2698 | `global_mean_entropy__area_ranked` | 0.2540 | small SFRM advantage |
| prototype | `gray_high_dice_high_boundary_error` | `sfrm_balanced_score__area_ranked` | 0.1630 | `global_mean_entropy__area_ranked` | 0.1522 | approximately tied |

Area-control interpretation:

- Foreground area was a real confounder in CoNSeP, especially for global mean
  entropy in boundary-local endpoints.
- After stratification, the overconfidence-aware boundary score becomes the
  strongest or near-strongest signal for `high_lecr_boundary_ge_q0.75`.
- The gray-zone endpoint remains mixed and should not be used as the primary
  CoNSeP claim.
- CoNSeP can support the paper as a stress-test dataset showing that SFRM is
  endpoint-specific and mechanism-aware, not as a clean universal win.

### 2026-06-24: Stage 3 Lightweight Reliability Predictors

Purpose:

- Test whether leakage-free SFRM descriptors remain useful when combined by
  simple supervised predictors.
- Keep the predictor as an auxiliary vehicle rather than the paper's main
  contribution.

Script:

- `scripts/stage3_lightweight_predictor.py`

Design:

- Models:
  - L2 logistic regression.
  - L1 logistic regression.
  - Random forest with shallow depth.
- Feature sets:
  - `global_features`: only global uncertainty and foreground-area features.
  - `sfrm_features`: boundary, uncertainty-cluster, topology, and anatomical
    consistency features.
  - `all_deployable_features`: all leakage-free `feat__*` columns.
- Validation:
  - GroupKFold by source image.
  - 14 source groups for every run.
  - 5 folds.
  - Outputs include raw AUROC, directional AUROC, AUPRC, and fixed review-budget
    metrics.
- Leakage control:
  - Feature names containing `gt`, `dice`, `error`, or `lecr` are rejected.

Outputs:

- `experiments/summaries/stage3_predictor_monuseg_point_seed42/`
- `experiments/summaries/stage3_predictor_monuseg_proto_seed42/`
- `experiments/summaries/stage3_predictor_consep_grid_point_seed42/`
- `experiments/summaries/stage3_predictor_consep_grid_proto_seed42/`
- `experiments/summaries/stage3_predictor_all_runs/predictor_review_budget_metrics_all.csv`
- `experiments/summaries/stage3_predictor_all_runs/predictor_discriminability_all.csv`

Selected findings:

| Dataset | Model | Endpoint | Best SFRM predictor / score | 10% recall | Key global comparator | 10% recall | Interpretation |
|---|---|---|---|---:|---|---:|---|
| MoNuSeg | point | `high_lecr_boundary_ge_q0.75` | L1 logistic, SFRM features | 0.3095 | global max entropy | 0.3095 | SFRM matches the strongest global endpoint and gives much higher AUROC/AUPRC than mean entropy |
| MoNuSeg | point | `gray_high_dice_high_boundary_error` | boundary overconfidence score | 0.6250 | global mean entropy | 0.0000 | strong fixed-budget complementarity |
| MoNuSeg | prototype | `bad_dice_lt_0.65` | boundary-risk score | 0.5484 | global mean entropy | 0.0323 | strong support after improved segmentation suppresses coarse uncertainty |
| MoNuSeg | prototype | `low_boundary_dice_le_q25` | L2 logistic, SFRM features | 0.3810 | global mean entropy | 0.0238 | SFRM strongly improves boundary-quality screening |
| CoNSeP | point | `high_lecr_boundary_ge_q0.75` | L2 logistic, SFRM features | 0.2222 | global mean entropy | 0.1587 | SFRM improves local boundary-error screening |
| CoNSeP | prototype | `high_lecr_boundary_ge_q0.75` | L1 logistic, SFRM features | 0.2222 | global mean entropy | 0.1032 | SFRM improves local boundary-error screening |
| CoNSeP | prototype | `low_boundary_dice_le_q25` | L2/L1 logistic, SFRM features | 0.3175 | global max entropy | 0.1587 | SFRM improves low-boundary-quality screening |
| CoNSeP | point/prototype | `bad_dice_lt_0.65` | global logistic features | 0.2000 / 0.2563 | best SFRM models | 0.1765 / 0.2462 | global features remain better for coarse bad-case detection |
| CoNSeP | point/prototype | `gray_high_dice_high_boundary_error` | global/area-related features | 0.5357 / 0.5000 | SFRM models | mixed | gray-zone remains area/foreground-confounded |

Stage 3 interpretation:

- The predictor experiment supports SFRM as an endpoint-specific reliability
  representation, especially for boundary-local and LECR-style failures.
- It does not support a universal "SFRM beats global uncertainty" claim.
- Global/foreground-area features remain strong for coarse bad-case and CoNSeP
  gray-zone labels, confirming that those endpoints are partly area driven.
- The strongest manuscript framing is therefore:
  "SFRM provides complementary, mechanism-aware failure descriptors that improve
  boundary-local review prioritization, while global uncertainty remains useful
  for coarse or area-driven failures."

## Open Decisions

- Define local critical-error labels that are not reducible to foreground area.
- Run qualitative inspection for CoNSeP gray-zone and LECR-boundary cases.
- Add review-budget simulation once MoNuSeg point-only/prototype and CoNSeP
  feature tables are available.

### 2026-06-24: Multi-seed robustness extension

Purpose:

- Strengthen Paper 1 beyond the original seed-42 reliability audit.
- Test whether SFRM's main boundary-local finding is stable across segmentation
  model seeds rather than a single checkpoint artifact.
- Add a stronger baseline comparison against trained global-feature predictors,
  not only predefined global entropy scores.

New Stage 0 feature tables:

- `stage0_monuseg_point_seed7_features_full.csv`
- `stage0_monuseg_point_seed123_features_full.csv`
- `stage0_monuseg_proto_seed7_features_full.csv`
- `stage0_monuseg_proto_seed123_features_full.csv`
- `stage0_consep_grid_point_seed7_features_full.csv`
- `stage0_consep_grid_point_seed123_features_full.csv`
- `stage0_consep_grid_proto_seed7_features_full.csv`
- `stage0_consep_grid_proto_seed123_features_full.csv`

Integrity checks:

- All 12 pathology feature tables have 28 deployable feature columns and zero
  leakage columns.
- Dice and boundary Dice match the original `test_eval_per_patch.csv` files by
  order with maximum absolute differences around `5e-7`.
- Every run uses 14 source-image groups for GroupKFold.

Extended Stage 4 outputs:

- `experiments/summaries/stage4_multiseed_summary/source_bootstrap_comparisons_multiseed.csv`

Key multi-seed findings:

| Endpoint | Comparator | Mean 10% recall difference | Positive runs | Interpretation |
|---|---|---:|---:|---|
| `low_boundary_dice_le_q25` | SFRM logistic vs global max entropy | +0.2123 | 11/12 | Strongest and most stable paper endpoint |
| `low_boundary_dice_le_q25` | SFRM logistic vs trained global-feature logistic | +0.0470 | 8/12 | Positive but modest; do not overclaim superiority over trained global predictors |
| `high_lecr_boundary_ge_q0.75` | SFRM L1 logistic vs global mean entropy | +0.0939 | 11/12 | Supportive mechanism endpoint |
| `high_lecr_boundary_ge_q0.75` | SFRM L1 logistic vs trained global-feature logistic | +0.0536 | 9/12 | Directionally positive but not strong enough as a primary claim |
| `bad_dice_lt_0.65` | trained global-feature logistic vs SFRM logistic | +0.0310 | 7/12 | Macro Dice failure remains mixed/global-friendly |

Current interpretation:

- The main statistically defensible claim should be anchored on
  `low_boundary_dice_le_q25`.
- The strongest comparison is against conventional global entropy, where SFRM
  exposes a clear blind spot.
- Against a trained global-feature predictor, SFRM remains directionally useful
  but not uniformly dominant. This must be written as complementarity, not
  replacement.
- `high_lecr_boundary` is supportive and mechanistic, not the central endpoint.

### 2026-06-24: External 3D FeTS/BraTS feasibility validation

Purpose:

- Test whether the SFRM boundary-local claim transfers beyond pathology patches.
- Reuse cached 3D segmentation artifacts instead of retraining a model.

Input artifacts:

- `D:\paper4\papers\paper2_uncertainty_tta_collected_docs_20260601\experiments\q2_extended_eval_20260616_site18_C_n20\case_artifacts\single`
- `D:\paper4\papers\paper2_uncertainty_tta_collected_docs_20260601\experiments\q2_extended_eval_20260616_site19_C_n4\case_artifacts\single`
- `D:\paper4\papers\paper2_uncertainty_tta_collected_docs_20260601\experiments\q2_extended_eval_20260616_site20_C_n20\case_artifacts\single`

New script:

- `scripts/stage0_extract_3d_npz_features.py`

Output:

- `experiments/summaries/stage0_fets_single_sites18_19_20_features_full.csv`
- `experiments/summaries/stage1_budget_fets_single_sites18_19_20/`
- `experiments/summaries/stage3_predictor_fets_single_sites18_19_20/`
- `experiments/summaries/stage4_bootstrap_fets_single_sites18_19_20/`

Design:

- 44 cases from three sites.
- 44 unique source groups.
- Deployable features use only `pred`, `probability`, and `uncertainty`.
- `label` and `error` are used only for evaluation endpoints.

Selected findings:

| Endpoint | SFRM 10% recall | Comparator | Comparator 10% recall | Source bootstrap result |
|---|---:|---|---:|---|
| `low_boundary_dice_le_q25` | 0.3636 | global max entropy | 0.0000 | diff +0.3636, CI [+0.1818, +0.6667] |
| `low_boundary_dice_le_q25` | 0.3636 | trained global features | 0.1818 | diff +0.1818, CI crosses zero |
| `high_lecr_boundary_ge_q0.75` | 0.2727 | global mean entropy | 0.0000 | diff +0.2727, CI touches zero |

Interpretation:

- The external 3D data support the boundary-local SFRM mechanism.
- Because the external set has only 44 cases and no `bad_dice_lt_0.65`
  positives, it should be used as feasibility/mechanism validation rather than
  the primary benchmark.
