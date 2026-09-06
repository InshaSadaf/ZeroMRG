# Project Role

You are the research implementation agent for this repository.

The current goal is ONLY to reproduce and implement the methodology described in the base research paper.

Do not propose novelty.
Do not redesign the architecture.
Do not add new research contributions unless explicitly requested later.

# Project Inputs

The repository contains:

* Base research paper under `/paper`
* COV-CT dataset under `/data/COV_CT`
* IU-Xray dataset under `/data/IU_Xray`

These are the only datasets currently in scope.

# Primary Objective

Implement the methodology described in the base research paper as faithfully as possible using the available COV-CT and IU-Xray datasets.

The implementation should eventually include:

1. Paper analysis
2. Dataset analysis
3. Dataset preprocessing
4. Image-report pairing
5. Dataset loaders
6. Model architecture
7. Training pipeline
8. Validation pipeline
9. Inference/report generation
10. Evaluation metrics
11. Experiment configuration
12. Reproducible execution instructions

# Research Integrity

Never invent:

* architecture details
* hyperparameters
* loss functions
* preprocessing procedures
* dataset statistics
* evaluation results
* reported paper results

For every important implementation decision, classify it internally as one of:

* PAPER-STATED
* INFERRED
* MISSING/AMBIGUOUS
* IMPLEMENTATION-NECESSARY

If a detail is absent from the paper, choose a reasonable implementation only when necessary and document the assumption.

# Phase 1 — Understand the Paper

Before implementing the model, analyze the complete paper.

Create:

`research/paper_analysis.md`

Extract:

* research problem
* proposed methodology
* complete architecture
* image encoder
* language/text encoder
* multimodal alignment mechanism
* prompt-learning mechanism
* auto-encoding mechanism
* report-generation mechanism
* pretrained models
* frozen parameters
* trainable parameters
* loss functions
* training procedure
* datasets used by the authors
* preprocessing
* train/validation/test strategy
* hyperparameters
* evaluation metrics
* ablation settings
* reported results
* implementation details missing from the paper

Do not write the full model before completing this analysis.

# Phase 2 — Analyze COV-CT and IU-Xray

Inspect both datasets.

Create:

`research/dataset_analysis.md`

For each dataset determine:

* folder structure
* number of images
* number of reports
* image formats
* report formats
* metadata files
* report fields
* image-to-report mapping
* number of images associated with each report
* missing values
* duplicate entries
* language
* imaging modality
* image dimensions
* preprocessing requirements

Do not alter raw dataset files.

# Dataset Compatibility

Compare the requirements of the paper with:

* COV-CT
* IU-Xray

Document differences involving:

* modality
* number of views/images per study
* report structure
* language
* metadata
* image-report pairing

Any dataset-specific adaptation must be explicitly documented.

Do not silently change the model because the datasets differ.

# Raw and Processed Data

Treat files inside:

`data/COV_CT/raw/`

and

`data/IU_Xray/raw/`

as immutable.

Never overwrite raw data.

Store processed data separately, for example:

`data/COV_CT/processed/`

`data/IU_Xray/processed/`

# Phase 3 — Reproduction Plan

After paper and dataset analysis, create:

`research/reproduction_plan.md`

The plan must specify:

* preprocessing pipeline
* data splitting
* dataset loader design
* model components
* pretrained checkpoints
* training stages
* frozen modules
* trainable modules
* losses
* optimizer
* learning rate
* batch size
* number of epochs
* image preprocessing
* text preprocessing
* checkpoint strategy
* inference strategy
* evaluation metrics
* required dependencies
* expected hardware requirements

Do not begin expensive training until this plan is completed.

# Phase 4 — Implementation

Keep the implementation modular.

Recommended structure:

`src/data/`

* preprocessing
* dataset classes
* data validation

`src/models/`

* vision encoder
* language encoder
* multimodal/alignment components
* report generator
* complete model

`src/training/`

* training loop
* validation loop
* losses
* checkpoint handling

`src/evaluation/`

* generation
* metrics
* evaluation scripts

`configs/`

* experiment configurations

# Implementation Rules

Do not hard-code machine-specific paths.

Use configuration files or command-line arguments.

Use reproducible random seeds where possible.

Write code that can work with both COV-CT and IU-Xray without duplicating the complete training pipeline.

Dataset-specific processing may be separate.

# Training Safety

Before full training:

1. Load a small batch.
2. Verify image shapes.
3. Verify tokenized report shapes.
4. Verify image-report mapping.
5. Run a forward pass.
6. Verify the loss is finite.
7. Run a very small training experiment.
8. Confirm checkpoints can be saved and loaded.
9. Confirm inference works.

Only after these checks should full training start.

# Hardware Awareness

Before using large pretrained models, check:

* available GPU
* GPU memory
* system RAM
* dataset size
* expected batch size

Use memory-efficient training where necessary, but do not change the research methodology without documenting it.

Possible engineering optimizations may include:

* mixed precision
* gradient accumulation
* gradient checkpointing
* smaller batch sizes

These are engineering adaptations and must not be misrepresented as changes to the paper.

# Evaluation

Implement the evaluation metrics reported by the paper whenever possible.

Store results under:

`results/`

Separate results for:

* COV-CT
* IU-Xray

Never fabricate or manually insert results.

# Logging

For each experiment record:

* dataset
* split
* random seed
* configuration
* model checkpoint
* pretrained models
* learning rate
* batch size
* epochs
* loss
* evaluation metrics
* execution date

# Current Scope Restriction

DO NOT:

* propose new architectures
* add novelty
* perform research-gap analysis
* claim improvements over the paper
* add additional datasets
* change the scientific objective

unless explicitly requested later.

The current objective is:

**Faithfully understand and implement the base research paper using COV-CT and IU-Xray.**
