# Paper 1 Deep Research Strategy

## Decision

The first paper should not be written as another uncertainty-estimation paper
and should not start by proposing a new segmentation backbone.

The strongest first paper is:

**Beyond Average Uncertainty: Spatial Failure-Region Modeling for Reliable
Medical Image Segmentation**

Chinese working description:

**超越平均不确定性：面向可靠医学图像分割的空间失败区域建模**

## Core Claim

Global confidence or uncertainty scores are insufficient for medical image
segmentation reliability because they discard where the risk occurs. A small
amount of uncertainty at a tumor boundary, organ interface, touching object, or
confounder-like structure can be more clinically important than diffuse
uncertainty in irrelevant background.

Therefore, segmentation reliability should be modeled through structured
failure regions rather than image-level averages alone.

## Literature-Derived Positioning

### 1. Failure detection and confidence aggregation already exist

Zenk et al. benchmarked failure detection methods for medical image
segmentation and emphasized confidence aggregation. Their result makes clear
that failure detection is a real deployment problem, but the benchmark is still
mostly about which confidence score or aggregation works best.

Our distinction:

- not only aggregate confidence;
- decompose risk into medically meaningful spatial regions;
- report both case-level failure prediction and region-level failure
  localization;
- preserve interpretability through named region types.

### 2. Spatial uncertainty aggregation is an emerging CVPR-level direction

The CVPR 2026 paper "Better than Average" explicitly argues that global average
uncertainty ignores spatial and structural uncertainty patterns, and shows that
spatial aggregation improves OOD and failure detection.

Our distinction:

- bring this idea into a medical-image-specific failure taxonomy;
- add boundary, topology, object, and confounder-region descriptors;
- evaluate region semantics, not only aggregation formulae;
- use medical metrics such as HD95, boundary Dice, AJI/PQ where appropriate.

### 3. Spatially-aware uncertainty evaluation already warns against voxel-wise metrics

Recent spatially-aware uncertainty evaluation work argues that voxel-wise
uncertainty metrics can treat scattered uncertainty and boundary-aligned
uncertainty as equivalent even though they have different clinical meanings.

Our distinction:

- move from evaluating uncertainty-map quality to predicting actual
  segmentation failure;
- use region decomposition as the bridge between uncertainty maps and clinical
  failure modes.

### 4. Segmentation quality prediction is related but not identical

QCResUNet predicts subject-level quality and voxel-level error maps. RCA and
In-Context RCA estimate segmentation quality without ground truth.

Our distinction:

- model-agnostic and interpretable rather than a monolithic quality-prediction
  network;
- focuses on why and where the failure occurs;
- does not require training a reverse segmenter for every case;
- produces deployable region descriptors that can later support optimization or
  low-cost annotation.

### 5. OOD and weak-supervision papers validate the problem, but solve a different one

OOD-SEG uses OOD detection for sparse positive-only segmentation learning.
Weakly supervised papers such as GloW-VSNet target annotation reduction.

Our distinction:

- Paper 1 is not a weak-supervised segmentation method;
- it is a reliability measurement and failure-region framework;
- later papers can use the same failure regions for low-cost annotation and
  optimization.

## Non-Negotiable Design Rules

### Rule 1: No ground-truth leakage

Deployable reliability features must be computed from:

- input image;
- predicted mask;
- probability map;
- uncertainty map;
- optional frozen feature map;
- optional metadata.

Ground-truth masks are only allowed for:

- defining failure labels;
- evaluating localization quality;
- constructing oracle upper bounds explicitly marked as non-deployable.

### Rule 2: Separate analysis regions from deployable regions

Analysis regions can include true FP/FN components because they are computed
after comparing prediction and ground truth.

Deployable regions must be computable before seeing ground truth:

- predicted boundary band;
- high-uncertainty connected components;
- low-margin regions;
- topology-risk components;
- shape-irregular predicted components;
- feature-ambiguous regions from frozen encoders;
- image-quality artifact regions if available.

### Rule 3: Do not overclaim clinical deployment

Paper 1 should claim:

> structured failure regions improve reliability assessment and failure
> localization under controlled public-dataset evaluation.

It should not claim:

> the system is ready for autonomous clinical deployment.

### Rule 4: The first paper must be benchmark-like

To become the foundation for later work, Paper 1 must produce:

- a clear taxonomy;
- reusable code;
- reproducible datasets/splits;
- transparent baselines;
- case-level and region-level metrics;
- statistical testing.

## Proposed Method: SFRM

Name:

**SFRM: Spatial Failure-Region Modeling**

### Inputs

For each image or volume:

- image `X`;
- probability map `P`;
- predicted mask `M`;
- optional uncertainty maps `U`;
- optional frozen feature map `F`.

### Deployable Region Types

1. **Boundary-risk region**
   - band around predicted object boundaries;
   - probability-gradient or signed-distance instability;
   - captures boundary leakage and contour ambiguity.

2. **Uncertainty-cluster region**
   - connected components of high entropy, high TTA disagreement, or low
     foreground-background margin;
   - captures spatially coherent uncertainty rather than isolated noisy pixels.

3. **Topology-risk region**
   - thin bridges, holes, small islands, fragmented components, or abnormal
     connected-component counts;
   - captures merge/split errors and object integrity failures.

4. **Feature-ambiguity region**
   - frozen feature neighborhoods similar to target or boundary prototypes but
     spatially inconsistent with confident prediction;
   - captures visually plausible confounders.

5. **Image-quality/artifact-risk region**
   - blur, low contrast, staining artifacts, scanner noise, or motion-like
     degradation;
   - optional in Paper 1 unless easy to compute.

### Region Descriptors

For every deployable region:

- area fraction;
- number of connected components;
- largest component area;
- compactness / eccentricity;
- boundary contact ratio;
- mean and max entropy;
- mean and max low-margin uncertainty;
- probability variance under TTA or ensemble when available;
- distance-to-boundary statistics;
- topology descriptors such as holes, bridges, and small islands;
- frozen-feature similarity or entropy statistics when available.

### Reliability Models

Start simple:

- logistic regression;
- random forest;
- gradient boosting;
- calibrated linear model;
- score-level weighted aggregation.

Avoid a heavy neural predictor in Paper 1 unless simple models fail. The point
is to prove region structure matters, not to hide the result inside another
network.

## Baselines

### Global baselines

- mean entropy;
- max entropy;
- mean max-softmax probability;
- mean margin uncertainty;
- foreground-only mean entropy;
- predicted foreground area ratio;
- calibration error if validation data permit.

### Existing stronger baselines

- pairwise Dice between ensemble predictions if multiple predictions are
  available;
- TTA disagreement;
- RCA or In-Context RCA if implementation is feasible;
- QCResUNet-style quality predictor only as a later strong baseline, not
  necessary for the first prototype.

### Spatial aggregation baselines

- patch-wise uncertainty aggregation;
- boundary-band uncertainty;
- class-wise uncertainty;
- thresholded high-uncertainty area;
- CVPR-style spatial uncertainty aggregators where reproducible.

## Evaluation Matrix

### Case-level failure prediction

Targets:

- low Dice;
- high HD95;
- low boundary Dice;
- low AJI/PQ for dense object datasets.

Metrics:

- AUROC;
- AUPRC;
- risk-coverage curve;
- selective segmentation performance;
- Spearman correlation with continuous quality metrics.

### Region-level failure localization

Targets:

- true error regions from prediction-vs-ground-truth comparison;
- boundary error regions;
- false-positive components;
- false-negative components;
- merge/split regions when instance-like masks exist.

Metrics:

- region IoU;
- error-region Dice;
- boundary-error recall;
- top-k failure-region hit rate;
- precision at fixed review budget.

### Statistical testing

Use paired tests over cases:

- Wilcoxon signed-rank test for AUROC/AUPRC via bootstrap folds;
- bootstrap confidence intervals for metric differences;
- report effect size, not only p-values.

## Dataset Strategy

### Stage A: Pathology first

Use MoNuSeg/CoNSeP outputs from the existing pathology branch if available.

Reason:

- dense touching objects;
- boundary and topology failures are visible;
- AJI/PQ can test object-level failure;
- existing code and metrics reduce startup cost.

Risk:

- if only pathology is used, the paper may look like a pathology-specific
  reliability study.

### Stage B: Add one non-pathology segmentation dataset

Add one public MRI/CT dataset after the pipeline works.

Preferred choice:

- BraTS/FeTS if predictions and masks can be reused;
- Medical Segmentation Decathlon prostate or heart if setup is easier.

Reason:

- shows the framework is not only for nuclei;
- lets HD95 and boundary failure matter;
- strengthens the claim of medical segmentation reliability.

### Stage C: PACS data later

Do not start Paper 1 with PACS. Use PACS after the public-data framework is
stable.

## Minimum Publishable Contribution

Paper 1 is publishable only if it proves all three:

1. spatial failure-region descriptors outperform global uncertainty for
   case-level failure prediction;
2. the framework localizes error-prone regions better than thresholded
   uncertainty alone;
3. the improvement is consistent on at least two failure archetypes, such as
   dense pathology objects and volumetric lesion/organ segmentation.

If only one of these holds, the paper should be reframed as a technical report
or workshop paper, not a main journal submission.

## Recommended First Experiments

### Experiment 1: Leakage-free baseline table

Compare:

- mean entropy;
- max entropy;
- foreground mean entropy;
- high-uncertainty area;
- boundary-band entropy;
- uncertainty connected-component descriptors;
- full SFRM descriptors.

Target:

- bad-case detection for Dice and boundary Dice.

### Experiment 2: Region localization

Compare:

- top entropy pixels;
- high-entropy connected components;
- boundary-risk regions;
- full SFRM candidate regions.

Target:

- true error regions computed from prediction and ground truth.

### Experiment 3: Cross-dataset generalization

Train reliability model on one split/source and test on another.

Target:

- demonstrate that failure-region descriptors are not just overfit to one
  validation split.

### Experiment 4: Ablation by region type

Remove one region type at a time:

- no boundary;
- no topology;
- no uncertainty cluster;
- no feature ambiguity.

Target:

- show which medical failure mode each region family captures.

## Manuscript Story

### Introduction thesis

Current medical segmentation reliability methods often compress a complex
spatial failure pattern into one scalar. This is unsafe because clinically
important errors are structured and localized.

### Method thesis

SFRM decomposes a prediction into interpretable, deployable failure-risk
regions and uses their geometry, uncertainty, topology, and feature ambiguity
to predict segmentation reliability.

### Results thesis

Spatial failure-region descriptors improve bad-case detection, error-region
localization, and boundary/object-level reliability assessment over global
uncertainty aggregation.

### Discussion thesis

The framework establishes a foundation for later failure-aware optimization,
low-cost annotation, and clinical quality-control triage.

## Target Venues

If results are strong across two modalities:

- Medical Image Analysis;
- IEEE TMI.

If results are solid but narrower:

- Artificial Intelligence in Medicine;
- Computer Methods and Programs in Biomedicine;
- Biomedical Signal Processing and Control.

If the first version is mainly benchmark/protocol:

- MICCAI workshop or MIDL short paper before journal expansion.

## Sources Consulted

- Guarino et al., "Better than Average: Spatially-Aware Aggregation of
  Segmentation Uncertainty Improves Downstream Performance", CVPR 2026 /
  arXiv:2603.29941.
- Zenk et al., "Comparative Benchmarking of Failure Detection Methods in Medical
  Image Segmentation: Unveiling the Role of Confidence Aggregation", Medical
  Image Analysis, 2024/2025.
- Zeevi et al., "Spatially-Aware Evaluation of Segmentation Uncertainty",
  arXiv:2506.16589.
- Valindria et al., "Reverse Classification Accuracy: Predicting Segmentation
  Performance in the Absence of Ground Truth", IEEE TMI, 2017.
- Cosarinsky et al., "In-Context Reverse Classification Accuracy: Efficient
  Estimation of Segmentation Quality without Ground-Truth", arXiv:2503.04522.
- Qiu et al., "QCResUNet: Joint subject-level and voxel-level segmentation
  quality prediction", Medical Image Analysis, 2025/2026.
- Wang et al., "OOD-SEG: Exploiting out-of-distribution detection techniques
  for learning image segmentation from sparse multi-class positive-only
  annotations", Medical Image Analysis, 2026.
- Hong et al., "Out-of-distribution Detection in Medical Image Analysis: A
  survey", arXiv:2404.18279.
- Local CVPR 2026 highlighted-paper notes on Spatial-SAM, Similarity-as-
  Evidence, Keep It Frozen, and CARE.

