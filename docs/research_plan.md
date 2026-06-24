# Research Plan

## Program Thesis

The long-term research program is:

**Failure-region-centric reliable medical image analysis under low annotation.**

The main claim is that medical image model reliability should be evaluated and
improved through spatially structured failure regions rather than global
averages alone.

## First-Principles Assumption

In medical images, the most important model errors are concentrated in specific
spatial structures:

- lesion or organ boundaries;
- touching objects;
- low-contrast interfaces;
- visually similar non-target structures;
- scanner or protocol artifact regions;
- local out-of-distribution anatomy or tissue appearance.

Therefore, a clinically useful reliability framework should answer:

1. where is the model likely to fail?
2. why is that region high risk?
3. can the risk be predicted before full manual correction?
4. can the region guide model optimization or annotation?

## Paper Series

### Paper 1: Spatial Failure-Region Reliability Modeling

Goal: prove that region-level spatial descriptors predict segmentation failure
better than global uncertainty.

Expected outputs:

- failure-region decomposition module;
- per-case and per-region reliability table;
- bad-case detection benchmark;
- qualitative failure-region figure.

### Paper 2: Failure-Region Optimization Network

Goal: use mined failure regions to optimize segmentation training.

Expected components:

- region-weighted loss;
- hard-negative contrastive loss;
- boundary repair loss;
- reliability map output.

### Paper 3: Failure-Region-Guided Low-Cost Annotation

Goal: determine which annotation action yields the highest information gain.

Candidate annotation types:

- center point;
- boundary point;
- negative point;
- failure-region confirmation;
- local mask correction.

### Paper 4: PACS Reliability Quality Control

Goal: use failure-region evidence for clinical deployment triage.

Expected components:

- unreliable case detection;
- scan/protocol metadata analysis;
- output audit report;
- human review prioritization.

## Immediate Decision

Start with Paper 1. Do not begin with a new backbone. Establish the measurement
and reliability framework first.

