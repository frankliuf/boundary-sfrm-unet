# Paper 1 Next Experiment Plan

## Decision

Do **feature-discrimination validation first**, then build the reliability
predictor.

Reason:

If we train the predictor immediately, a weak result would be ambiguous. It
could mean:

- SFRM features are not informative;
- the feature set is too redundant;
- the bad-case label is poorly defined;
- the predictor is underpowered or overfit;
- the dataset does not contain enough meaningful failures.

A feature-discrimination audit isolates the foundation question:

> Do boundary, uncertainty, topology, anatomical consistency, and feature
> ambiguity descriptors actually differ between reliable and unreliable
> segmentations?

## Experiment 0: Feature-Discrimination Audit

### Inputs

- image;
- model probability map;
- predicted mask;
- ground-truth mask for evaluation labels only;
- optional uncertainty maps;
- optional frozen feature maps.

### Leakage Rule

Ground truth must not be used to compute predictor features. It can only define:

- Dice / HD95 / boundary Dice / AJI / PQ;
- bad-case labels;
- true error regions for localization evaluation.

### Feature Families

1. Boundary-risk features
2. Uncertainty-cluster features
3. Topology-risk features
4. Anatomical/topological consistency features
5. Feature-ambiguity features
6. Image-quality/artifact features if easy to compute

### Analysis

For every feature:

- good vs bad distribution plots;
- Mann-Whitney U or Wilcoxon-style group comparison where appropriate;
- univariate AUROC for bad-case detection;
- Spearman correlation with Dice, HD95, boundary Dice, AJI/PQ;
- missingness and stability check.

For the feature set:

- correlation matrix;
- variance inflation factor or simple high-correlation pruning;
- Lasso feature selection;
- ablation by feature family.

## Experiment 1: First Reliability Predictor

Run only after Experiment 0 shows useful signal.

Models:

- logistic regression with L1/L2 regularization;
- random forest;
- gradient boosting;
- calibrated logistic regression.

Outputs:

- case-level risk score;
- calibrated failure probability;
- top contributing feature families.

## Experiment 2: Human-Review Budget Simulation

Clinical scenario:

> A clinician can only review 10% of AI outputs. Which ranking captures the
> most critical failures?

Compare:

- random review;
- global mean entropy;
- max entropy;
- foreground entropy;
- TTA disagreement if available;
- SFRM risk score.

Budgets:

- 5%;
- 10%;
- 20%.

Metrics:

- critical-error recall;
- accepted bad-case reduction;
- risk-coverage curve;
- boundary/object failure capture rate.

## Initial Dataset Recommendation

Start with the existing pathology branch if probability maps and masks are
readily available. It is the fastest route to test whether SFRM features detect:

- touching-object failures;
- boundary leakage;
- false-positive confounder regions;
- object fragmentation.

Then add one non-pathology dataset only after the feature audit is stable.

## Stop Criteria

Do not proceed to a full manuscript if:

- SFRM features do not outperform global uncertainty in univariate or simple
  multivariate tests;
- improvements only appear when using ground-truth-derived features;
- no consistent feature family contributes across splits;
- human-review simulation does not improve over global uncertainty ranking.

Proceed if:

- at least two SFRM feature families show stable discrimination;
- leakage-free SFRM features improve AUROC/AUPRC over global uncertainty;
- the review-budget simulation captures more critical errors at 10% review
  budget.

