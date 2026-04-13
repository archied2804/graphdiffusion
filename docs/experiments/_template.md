---
experiment_id: "EXP-NNN"
title: ""
date: YYYY-MM-DD
status: planned | running | complete | abandoned
parent: null
tags: []
config: "configs/EXP-NNN_slug.yaml"
output_dir: "outputs/EXP-NNN_slug/"
---

# EXP-NNN: Title

> **One-line summary:** …

## Motivation

Why this experiment?  What question does it answer?
Link to parent experiment if this is a follow-up: [[EXP-parent]]

## Hypothesis

State the expected outcome *before* running.

## Changes from parent

> Skip this section for the first experiment in a lineage.

- **Config diff:** list changed hyperparameters vs parent
- **Code changes:** list new/modified classes or functions
- **Data changes:** dataset, transforms, pre-processing

## Method

### Architecture

Describe the model configuration — reference the config file.

### Dataset

Describe the data: geometry, number of graphs, features, connectivity.

### Training

Epochs, optimiser, learning rate, schedule, hardware, wall-clock time.

## Results

### Metrics

| Metric           | Train | Val  |
|-----------------|-------|------|
| Final loss       |       |      |
| Best val loss    |       |      |

### Plots

Embed or link to generated figures:

![[outputs/EXP-NNN_slug/generated_shapes.png]]

### Observations

Bullet-point observations from inspecting outputs.

## Conclusions

- Did the hypothesis hold?
- Key takeaway for next experiment

## Next steps

- [ ] Follow-up experiment idea → [[EXP-next]]
- [ ] Code improvement needed
