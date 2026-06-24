# Data Plan

## Initial Strategy

Use existing local data and prediction infrastructure first. Public datasets
should be used to validate the framework, not to create a large download burden
at the beginning.

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

