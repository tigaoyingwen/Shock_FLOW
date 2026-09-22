# Shock-aware flow matching for PDEBench 1D

This repository contains the reproducible implementation of a shock-aware neural operator for the one-dimensional PDEBench CFD shock dataset. The proposed model is a two-stage system:

1. **SWFMO** — Shock-Frame Wavelet Flow Matching Operator, which predicts the global trajectory from the observed initial field.
2. **Metric-Aligned Shock Residual Flow**, which is trained on top of a frozen SWFMO checkpoint to correct shock location, sharpness, gradient structure, and local overshoot.


## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for the machine first if a CUDA wheel is required, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[viz,dev]'
```

For GPU runs, follow the PyTorch installation selector for the desired CUDA version before running `pip install -e .`.
