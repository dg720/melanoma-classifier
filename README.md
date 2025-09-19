# Melanoma Classifier
A lightweight research sandbox for prototyping melanoma image classification workflows.

## Overview
This project assembles a reproducible pipeline for researchers exploring dermoscopic image classification. It bundles data preparation, convolutional model training, offline explainability experiments, and an interactive demo surface so experiments can progress without rebuilding infrastructure from scratch.

## Architecture
Dermoscopic images and labels are preprocessed into NumPy archives, which feed a Keras-based convolutional neural network. The trained model is reused for inference experiments and is exposed through a FastAPI application that mounts a Gradio interface for hands-on validation. Optional notebook analyses and reports sit alongside the codebase for exploratory work.

## Key techniques & patterns
- NumPy and OpenCV preprocessing to standardise ISIC 2019 images and persist reusable datasets.
- Keras Sequential CNN with dropout regularisation and early-model checkpointing.
- LIME-based explainability workflow to visualise class contributions.
- FastAPI + Gradio integration to deliver a shareable inference playground.

## Limitations & future work
- File paths are hard-coded for a Windows development machine and require manual edits for other environments.
- The training script executes on import rather than via a CLI, so automation and parameter tuning are limited.
- Dataset preparation and prediction scripts run eagerly; wrapping them in functions or notebooks would improve composability and testability.
- No deployment or monitoring tooling is provided yet.

## Project structure
```text
melanoma-classifier/
├── data/                     # Raw, interim, processed, and external datasets
├── models/                   # Saved models and evaluation artifacts
├── notebooks/                # Experimental notebooks for exploration
├── reports/                  # Generated figures and reports
├── src/
│   ├── config.py             # Central location for future configuration constants
│   ├── dataset.py            # Scripted preprocessing for ISIC 2019 imagery
│   ├── features.py           # Placeholder for feature engineering utilities
│   ├── modeling/
│   │   ├── gradio_ui.py      # Gradio interface exposing inference results
│   │   ├── predict.py        # Prediction helpers plus LIME explainability flow
│   │   ├── run.py            # FastAPI entry point mounting the Gradio demo
│   │   └── train.py          # CNN training loop and checkpointing logic
│   └── plots.py              # Placeholder for reusable plotting helpers
└── requirements.txt          # Python dependencies
```
