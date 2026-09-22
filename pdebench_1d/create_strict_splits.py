"""Create a leakage-free ID/OOD manifest for the PDEBench 1D cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import _batched, make_strict_ood_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-name", default="strict_id_ood_splits.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strength-quantile", type=float, default=0.9)
    parser.add_argument("--position-tail-quantile", type=float, default=0.1)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    descriptors = np.load(cache_dir / "initial_descriptors.npy")
    manifest = make_strict_ood_split_manifest(
        descriptors,
        seed=args.seed,
        strength_quantile=args.strength_quantile,
        position_tail_quantile=args.position_tail_quantile,
    )

    fields = np.load(metadata["cache"], mmap_mode="r")
    sums = np.zeros(fields.shape[-1], dtype=np.float64)
    square_sums = np.zeros(fields.shape[-1], dtype=np.float64)
    element_count = 0
    train_ids = np.sort(np.asarray(manifest["train"], dtype=np.int64))
    for ids in _batched(train_ids, 128):
        values = np.asarray(fields[ids], dtype=np.float64)
        sums += values.sum(axis=(0, 1, 2))
        square_sums += np.square(values).sum(axis=(0, 1, 2))
        element_count += values.shape[0] * values.shape[1] * values.shape[2]
    mean = sums / element_count
    std = np.sqrt(np.maximum(square_sums / element_count - mean**2, 1e-12))
    manifest["normalization_mean"] = mean.astype(float).tolist()
    manifest["normalization_std"] = std.astype(float).tolist()
    manifest["normalization_source"] = "strict_train_trajectories_only"
    manifest["allowed_test_inputs"] = ["initial_field", "x_coordinate", "query_time"]
    manifest["forbidden_test_inputs"] = [
        "future_field",
        "analytic_riemann_solution",
        "exact_wave_speed",
        "test_label",
    ]

    output = cache_dir / args.output_name
    output.write_text(json.dumps(manifest, indent=2))
    sizes = {
        key: len(value)
        for key, value in manifest.items()
        if isinstance(value, list) and key.endswith(("train", "val", "test"))
    }
    print(json.dumps({"output": str(output), "sizes": sizes, "thresholds": {
        "strength": manifest["strength_threshold"],
        "position": manifest["position_thresholds"],
    }}), flush=True)


if __name__ == "__main__":
    main()
