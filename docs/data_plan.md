# Data Plan

## Initial Strategy

Use existing local data and prediction infrastructure first. Public datasets
should be used to validate the framework, not to create a large download burden
at the beginning.

## Paper 1 Dataset Decision

The first feature-discrimination audit should start with **MoNuSeg test
patches** from the existing pathology branch.

Rationale:

- existing checkpoints are already available;
- per-patch metrics already exist;
- probability maps can be regenerated from the saved models;
- boundary leakage, object touching, and topology errors match the SFRM thesis;
- the first experiment can focus on reliability features rather than dataset
  setup.

Second dataset: CoNSeP.

Rationale:

- denser and more heterogeneous nuclei scenes;
- stronger stress test for object-level failure and gray-zone cases;
- can reuse the current pathology tooling.

Third dataset: one non-pathology segmentation dataset only after MoNuSeg and
CoNSeP are stable.

## Candidate Data Sources

### Pathology

- MoNuSeg;
- CoNSeP;
- PanNuke if needed later.

Rationale: dense objects, boundary ambiguity, object merging, and confounder
regions match the failure-region thesis.

### Brain Tumor MRI

- BraTS or FeTS-derived experiments if reusable predictions are available.

Rationale: boundary uncertainty, site shift, lesion heterogeneity, and HD95
failure are suitable for reliability prediction.

## Required Artifacts Per Dataset

- images;
- ground-truth masks;
- baseline probability maps;
- predicted masks;
- data split file;
- per-case metrics;
- optional metadata.

## Do Not Do Yet

- Do not bulk-download many datasets.
- Do not start with PACS ingestion before the public-data prototype works.
- Do not train a foundation model.
- Do not start Paper 1 with LiTS/BraTS unless prediction and probability maps
  are already available; otherwise the project will drift into baseline
  training instead of feature validation.
