# Public data notes

This repository does not redistribute the raw pathology images or manual annotations used in the study.

The experiments are based on the following public datasets:

- CryoNuSeg
- CoNSeP
- MoNuSeg
- TNBC

Please obtain each dataset from its official source and follow the original license and citation requirements.

What is released in this repository:

- dataset-preparation scripts
- split-materialization utilities
- training and evaluation code
- experiment summaries and per-sample result tables
- figure-generation scripts and selected paper figures

What is intentionally excluded:

- raw image files
- raw annotation files copied from the public datasets
- cached intermediate maps tied to those raw files
- model checkpoints and weight files that would unnecessarily inflate the repository

After local download, place the datasets in your own workspace and pass the corresponding image, mask, instance-label, and confounder-map paths to the training or evaluation scripts.
