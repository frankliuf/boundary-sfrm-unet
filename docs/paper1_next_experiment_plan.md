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

Start with the existing pathology branch, specifically **MoNuSeg test patches**.
This is the fastest route because the workspace already contains trained
checkpoints and per-patch evaluation metrics under:

```text
D:\paper_MedIA Vol. 107–113\outputs\confounder_prompting\monuseg_split_point_seed42
D:\paper_MedIA Vol. 107–113\outputs\confounder_prompting\monuseg_split_contrastive_w005_seed42
```

Probability maps are not currently stored as raw arrays in those run folders,
but they can be regenerated from the saved checkpoints and existing dataset
loader. This is preferable to starting with a new dataset because it avoids
turning Experiment 0 into a data-engineering project.

MoNuSeg is suitable for the first 50-100 sample audit because it contains:

- touching-object failures;
- boundary leakage;
- false-positive confounder regions;
- object fragmentation.

### Dataset Staging

1. **Stage 0: MoNuSeg feature-discrimination audit**
   - purpose: validate whether SFRM features separate good and bad predictions;
   - sample size: start with 50-100 patches, then expand to the full test split;
   - outputs: feature table, UMAP/t-SNE, Mann-Whitney tests, feature AUROC.

2. **Stage 1: CoNSeP replication**
   - purpose: verify that signals persist in a denser, more heterogeneous
     nuclei dataset;
   - expected value: stronger object-level failure analysis using AJI/PQ.

3. **Stage 2: one non-pathology dataset**
   - candidate: BraTS/FeTS if reusable predictions are available; otherwise a
     simpler Medical Segmentation Decathlon dataset;
   - purpose: show that SFRM is not only a pathology-specific framework.

Do not start with LiTS/BraTS unless prediction/probability maps are already
available. For Paper 1, the bottleneck should be validating the reliability
framework, not rebuilding a segmentation training stack.

## Visualization Plan for Experiment 0

### Figure A: Feature separability map

- UMAP or t-SNE of normalized SFRM features;
- color by good/bad case label;
- shape by failure type if available;
- annotate the overlap region as the reliability gray zone.

### Figure B: Feature impact radar

For each feature family, report normalized discriminability:

- univariate AUROC;
- absolute Spearman correlation with Dice/HD95/boundary Dice;
- group-effect size.

The radar should compare:

- global entropy baseline;
- boundary-risk family;
- uncertainty-cluster family;
- topology-risk family;
- anatomical consistency family;
- feature-ambiguity family.

### Figure C: Gray-zone diagnostic matrix

Four quadrants:

- high Dice / low SFRM risk;
- high Dice / high SFRM risk;
- low Dice / low SFRM risk;
- low Dice / high SFRM risk.

The most important cases are:

- high Dice / high SFRM risk: global Dice looks acceptable but local critical
  failure may exist;
- low Dice / low SFRM risk: failure may be diffuse, label ambiguous, or outside
  the currently modeled failure families.

### Figure D: Fixed review-budget curve

At top 5%, 10%, and 20% reviewed cases, compare:

- random review;
- mean entropy;
- max entropy;
- foreground entropy;
- SFRM risk.

Report critical-error recall and accepted bad-case reduction.

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
