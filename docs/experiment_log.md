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

## Open Decisions

- Select first dataset and baseline prediction source.
- Decide whether to reuse pathology model outputs or train a small baseline
  from scratch.
- Implement first version of failure-region decomposition.
