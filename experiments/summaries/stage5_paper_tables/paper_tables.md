# SFRM Paper Tables

## Table 1. Primary endpoint: low boundary Dice vs conventional global max entropy
| dataset | model_family | endpoint | n_seeds | sfrm_recall_mean | comparator_recall_mean | recall_diff_mean | recall_diff_min | recall_diff_max | positive_seeds | source_bootstrap_significant_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consep | point | low_boundary_dice_le_q25 | 3 | 0.2434 | 0.1587 | 0.0847 | -0.0238 | 0.1429 | 2/3 | 2/3 |
| consep | proto | low_boundary_dice_le_q25 | 3 | 0.2989 | 0.1772 | 0.1217 | 0.0873 | 0.1587 | 3/3 | 2/3 |
| monuseg | point | low_boundary_dice_le_q25 | 3 | 0.3492 | 0.0079 | 0.3413 | 0.2857 | 0.3810 | 3/3 | 0/3 |
| monuseg | proto | low_boundary_dice_le_q25 | 3 | 0.3016 | 0.0000 | 0.3016 | 0.1905 | 0.3810 | 3/3 | 2/3 |

## Table 2. Strong baseline: low boundary Dice vs trained global-feature predictor
| dataset | model_family | endpoint | n_seeds | sfrm_recall_mean | comparator_recall_mean | recall_diff_mean | recall_diff_min | recall_diff_max | positive_seeds | source_bootstrap_significant_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consep | point | low_boundary_dice_le_q25 | 3 | 0.2434 | 0.2249 | 0.0185 | -0.0476 | 0.0635 | 2/3 | 0/3 |
| consep | proto | low_boundary_dice_le_q25 | 3 | 0.2989 | 0.2407 | 0.0582 | -0.0079 | 0.1746 | 2/3 | 1/3 |
| monuseg | point | low_boundary_dice_le_q25 | 3 | 0.3492 | 0.3175 | 0.0317 | -0.0238 | 0.1190 | 1/3 | 0/3 |
| monuseg | proto | low_boundary_dice_le_q25 | 3 | 0.3016 | 0.2222 | 0.0794 | 0.0238 | 0.1190 | 3/3 | 0/3 |

## Table 3. Supporting endpoint: high LECR boundary vs global mean entropy
| dataset | model_family | endpoint | n_seeds | sfrm_recall_mean | comparator_recall_mean | recall_diff_mean | recall_diff_min | recall_diff_max | positive_seeds | source_bootstrap_significant_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consep | point | high_lecr_boundary_ge_q0.75 | 3 | 0.2063 | 0.1111 | 0.0952 | 0.0556 | 0.1270 | 3/3 | 0/3 |
| consep | proto | high_lecr_boundary_ge_q0.75 | 3 | 0.2116 | 0.1455 | 0.0661 | -0.0159 | 0.1190 | 2/3 | 0/3 |
| monuseg | point | high_lecr_boundary_ge_q0.75 | 3 | 0.2063 | 0.0476 | 0.1587 | 0.0714 | 0.3095 | 3/3 | 1/3 |
| monuseg | proto | high_lecr_boundary_ge_q0.75 | 3 | 0.1905 | 0.1349 | 0.0556 | 0.0238 | 0.0952 | 3/3 | 0/3 |

## Table 4. External 3D FeTS/BraTS validation
| endpoint | sfrm_predictor | comparator | sfrm_recall | comparator_recall | recall_diff | ci95 | ci_excludes_zero | positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_lecr_boundary_ge_q0.75 | lasso_logistic_l1::sfrm_features | predefined::global_mean_entropy | 0.2727 | 0.0000 | 0.2727 | [0.0000, 0.5000] | False | 11 |
| low_boundary_dice_le_q25 | logistic_l2::sfrm_features | predefined::global_max_entropy | 0.3636 | 0.0000 | 0.3636 | [0.1818, 0.6667] | True | 11 |
| low_boundary_dice_le_q25 | logistic_l2::sfrm_features | logistic_l2::global_features | 0.3636 | 0.1818 | 0.1818 | [-0.0833, 0.4444] | False | 11 |
| high_lecr_boundary_ge_q0.75 | lasso_logistic_l1::sfrm_features | logistic_l2::global_features | 0.2727 | 0.0909 | 0.1818 | [-0.0835, 0.4000] | False | 11 |

Interpretation guardrail: Table 1 supports the main conventional-global-uncertainty blind-spot claim. Table 2 is a required strong-baseline caveat and supports complementarity rather than universal superiority.