# Paper 1 Protocol

## Working Title

Spatial Failure-Region Reliability Modeling for Low-Annotation Medical Image
Segmentation

## Research Question

Can spatially structured failure-region descriptors predict segmentation failure
better than global uncertainty or mean confidence?

## Hypothesis

Global uncertainty averages discard clinically important spatial information.
Failure concentrated near boundaries, connected-component breaks, confounder
regions, or low-contrast interfaces is more predictive of segmentation failure
than image-level uncertainty alone.

## Minimum Viable Experiment

### Inputs

- medical image;
- ground-truth mask;
- model probability map;
- predicted mask;
- optional frozen feature map;
- optional uncertainty map from TTA, entropy, or margin confidence.

### Region Decomposition

Compute the following candidate regions:

1. boundary band around predicted and ground-truth contours;
2. high-entropy connected components;
3. false-positive and false-negative regions;
4. topology-risk regions such as holes, bridges, small islands, and merged
   objects;
5. frozen-feature ambiguous regions if DINOv2/UNI features are available.

### Features

For each case and region:

- area ratio;
- mean entropy;
- max entropy;
- boundary contact length;
- distance to nearest object boundary;
- connected-component count;
- compactness;
- false-positive ratio;
- false-negative ratio;
- object fragmentation score.

### Prediction Targets

Case-level targets:

- low Dice;
- high HD95;
- low boundary Dice;
- low AJI or PQ when instance-like masks are available.

Region-level targets:

- false-positive region;
- false-negative region;
- boundary error region;
- merged-object region.

### Baselines

- mean softmax confidence;
- mean entropy;
- max entropy;
- MC dropout or TTA uncertainty when available;
- calibration error;
- image-level quality metrics.

### Evaluation

- AUROC for bad-case detection;
- AUPRC for bad-case detection under class imbalance;
- Spearman correlation with Dice, HD95, boundary Dice, AJI, and PQ;
- region-level IoU or Dice for failure localization;
- paired statistical tests against global uncertainty baselines.

## First Dataset Choice

Use a dataset where masks and existing segmentation code are easy to access.
The first candidate should be the current pathology resources because code and
metrics already exist locally. A second modality can be added only after the
pipeline is stable.

## Exclusion

Do not include ChatGPT or other large language models in Paper 1. The paper must
remain an image-centered reliability study.

