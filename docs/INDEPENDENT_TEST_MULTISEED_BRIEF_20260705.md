# Independent Test and Multi-Seed Brief

Updated: 2026-07-05

## Scope

This brief records the independent held-out test, three-seed evaluation, and paired statistical testing added to the Boundary-SFRM-UNet study on 2026-07-05.

## Protocol decision

Two CryoNuSeg source-level protocols were audited:

1. `5/3/2` source split (`cryonuseg_holdout_*`):
   - train sources: `Human_LymphNodes`, `Human_Mediastinum`, `Human_Pleura`, `Human_Skin`, `Human_Thymus`
   - val sources: `Human_AdrenalGland`, `Human_Larynx`, `Human_Pancreas`
   - test sources: `Human_Testes`, `Human_ThyroidGland`
   - outcome: unstable baseline behavior across seeds and systematic confounder-FPR deterioration.

2. `6/2/2` source split (`cryonuseg_holdout622_*`):
   - train sources: `Human_LymphNodes`, `Human_Mediastinum`, `Human_Pleura`, `Human_Skin`, `Human_Thymus`, `Human_ThyroidGland`
   - val sources: `Human_AdrenalGland`, `Human_Pancreas`
   - test sources: `Human_Larynx`, `Human_Testes`
   - outcome: substantially more stable and should be treated as the main independent-test protocol.

Main conclusion: the `6/2/2` source-level protocol is the publishable one. The earlier `5/3/2` split should be retained only as an internal failed audit showing that too-small CryoNuSeg training partitions produce unstable baselines.

## CryoNuSeg independent test

Artifact roots:

- seed summaries:
  - `experiments/boundary_sfrm_runs/cryonuseg_holdout622_test_multiseed_stats/seed_test_summary.csv`
  - `experiments/boundary_sfrm_runs/cryonuseg_holdout622_test_multiseed_stats/paired_test_statistics.csv`
- per-seed held-out evaluations:
  - `experiments/boundary_sfrm_runs/cryonuseg_holdout622_test_eval_seed42`
  - `experiments/boundary_sfrm_runs/cryonuseg_holdout622_test_eval_seed7`
  - `experiments/boundary_sfrm_runs/cryonuseg_holdout622_test_eval_seed123`

Three-seed independent-test deltas (refined minus baseline, except Confounder FPR where lower is better):

- Dice: mean `+0.1427`, bootstrap 95% CI `[+0.0144, +0.3193]`, positive seeds `3/3`
- Boundary Dice: mean `+0.1202`, bootstrap 95% CI `[+0.0223, +0.2576]`, positive seeds `3/3`
- AJI: mean `+0.0690`, bootstrap 95% CI `[+0.0162, +0.1281]`, positive seeds `3/3`
- PQ: mean `+0.0504`, bootstrap 95% CI `[+0.0184, +0.0854]`, positive seeds `3/3`
- Confounder FPR improvement: mean `-0.0953`, bootstrap 95% CI `[-0.2260, +0.0240]`, positive seeds `1/3`

Interpretation:

- The independent test now supports the segmentation-side claim cleanly: Boundary-SFRM-UNet improves overlap, boundary quality, and object-level quality on unseen CryoNuSeg sources.
- The confounder-suppression claim does **not** generalize cleanly under three-seed held-out testing. It should be softened to a validation-set tendency or reported as mixed on the independent CryoNuSeg test.

## CoNSeP independent test

Artifact root:

- `experiments/boundary_sfrm_runs/consep_test_eval_seed42/test_summary.json`

Current held-out test result for the existing seed-42 run:

- Dice: `+0.0252`
- Boundary Dice: `+0.0176`
- AJI: `+0.0081`
- PQ: `+0.0048`
- Confounder FPR: `+0.0348` worse

Interpretation:

- The CoNSeP test still behaves as a stress-test dataset rather than a clean all-metric win.
- The current manuscript should frame CoNSeP as bounded secondary evidence: overlap/boundary/object metrics improve modestly, while confounder leakage remains unresolved.

## Statistical testing

The new multi-seed test statistics use:

- source-level aggregation before paired testing;
- per-metric hierarchical bootstrap over seeds and sources;
- per-seed paired Wilcoxon p-values for audit context.

Practical note:

- Because CryoNuSeg has only two held-out source groups per seed under the `6/2/2` split, the per-seed Wilcoxon p-values are coarse (`0.5` or `1.0`) and should not be treated as the primary significance evidence.
- The hierarchical bootstrap confidence intervals are the more informative inferential summary for the current dataset size.

## Recommended manuscript update

Use the following stance in the paper:

1. Main positive evidence:
   - three-seed independent CryoNuSeg held-out test under the `6/2/2` source split;
   - claim consistent gains on Dice, Boundary Dice, AJI, and PQ.
2. Bounded limitation:
   - confounder suppression is not consistently retained on the independent CryoNuSeg test and remains unresolved on CoNSeP.
3. Secondary stress-test evidence:
   - CoNSeP independent seed-42 test can be reported as a harder crowded-scene setting with modest segmentation gains but persistent leakage.

## Scripts added in this round

- `scripts/materialize_cryonuseg_holdout_split.py`
- `scripts/evaluate_boundary_sfrm_on_split.py`
- `scripts/compute_boundary_sfrm_multiseed_stats.py`

## Reproducibility note

Both training scripts now seed Python, NumPy, and PyTorch, and explicitly force deterministic cuDNN behavior:

- `scripts/train_full_supervised_unet.py`
- `scripts/train_boundary_sfrm_unet.py`
