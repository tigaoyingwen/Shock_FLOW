# `pdebench_1d`

This package contains the public implementation of the proposed PDEBench 1D shock method:

- `data.py`: float32 cache creation, train-only normalization, and strict ID/OOD manifests;
- `models.py`: SWFMO, shared flow blocks, and differentiable initial-condition descriptors;
- `residual_flow.py`: metric-aligned shock residual correction over a frozen SWFMO;
- `train.py` and `train_residual_flow.py`: direct and two-stage training;
- `evaluate_ours.py` and `visualize_ours.py`: proposed-method metrics and figures.

Run modules from the repository root, for example:

```bash
python -m pdebench_1d.prepare_cache --help
python -m pdebench_1d.train --help
python -m pdebench_1d.train_residual_flow --help
```

The complete setup and reproducibility protocol is in the root [`README.md`](../README.md) and [`docs/method.md`](../docs/method.md).
