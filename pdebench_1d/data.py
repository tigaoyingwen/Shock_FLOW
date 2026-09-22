"""Memory-safe data preparation and loading for PDEBench 1D CFD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FIELD_NAMES = ("density", "pressure", "Vx")


def _batched(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def linear_resample_1d(
    values: np.ndarray, source_x: np.ndarray, target_x: np.ndarray
) -> np.ndarray:
    """Linearly resample the last axis without materializing an interpolation matrix."""
    source_x = np.asarray(source_x)
    target_x = np.asarray(target_x)
    if values.shape[-1] != len(source_x):
        raise ValueError("The last value axis must match source_x")
    if len(source_x) < 2 or np.any(np.diff(source_x) <= 0):
        raise ValueError("source_x must be strictly increasing")
    if target_x.min() < source_x[0] or target_x.max() > source_x[-1]:
        raise ValueError("target_x must remain inside the source domain")

    right = np.searchsorted(source_x, target_x, side="left")
    right = np.clip(right, 1, len(source_x) - 1)
    left = right - 1
    weight = (target_x - source_x[left]) / (source_x[right] - source_x[left])
    return (
        values[..., left] * (1.0 - weight)
        + values[..., right] * weight
    ).astype(np.float32, copy=False)


def initial_descriptors(initial: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Extract input-observable Riemann descriptors without future information."""
    scales = np.maximum(np.ptp(initial, axis=1), 1e-6)[:, None, :]
    sensor = np.sum(np.abs(np.diff(initial, axis=1)) / scales, axis=-1)
    cut = np.argmax(sensor, axis=1)
    out = np.empty((len(initial), 7), dtype=np.float32)
    for row, index in enumerate(cut):
        # Robust medians avoid the one pooled cell that may straddle the jump.
        left = np.median(initial[row, : max(index, 1) + 1], axis=0)
        right = np.median(initial[row, min(index + 1, initial.shape[1] - 1) :], axis=0)
        out[row] = (0.5 * (x[index] + x[index + 1]), *left, *right)
    return out


def make_split_manifest(descriptors: np.ndarray, seed: int = 2026) -> dict[str, list[int]]:
    """Create disjoint ID splits plus diagnostic OOD subsets."""
    count = len(descriptors)
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    n_train = int(0.8 * count)
    n_val = int(0.1 * count)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]

    x0 = descriptors[:, 0]
    rho_l, p_l, u_l, rho_r, p_r, u_r = descriptors[:, 1:].T
    strength = (
        np.log(np.maximum(rho_l, rho_r) / np.maximum(np.minimum(rho_l, rho_r), 1e-6))
        + np.log(np.maximum(p_l, p_r) / np.maximum(np.minimum(p_l, p_r), 1e-6))
        + np.abs(u_l - u_r)
    )
    strength_cut = np.quantile(strength, 0.9)
    position_lo, position_hi = np.quantile(x0, (0.1, 0.9))
    return {
        "seed": seed,
        "train": train.astype(int).tolist(),
        "val": val.astype(int).tolist(),
        "test": test.astype(int).tolist(),
        "strength_ood": np.flatnonzero(strength >= strength_cut).astype(int).tolist(),
        "position_ood": np.flatnonzero((x0 <= position_lo) | (x0 >= position_hi)).astype(int).tolist(),
    }


def make_strict_ood_split_manifest(
    descriptors: np.ndarray,
    seed: int = 2026,
    strength_quantile: float = 0.9,
    position_tail_quantile: float = 0.1,
) -> dict:
    """Create disjoint ID and OOD sets using initial-condition inputs only."""
    if not 0.5 < strength_quantile < 1.0:
        raise ValueError("strength_quantile must be between 0.5 and 1")
    if not 0.0 < position_tail_quantile < 0.5:
        raise ValueError("position_tail_quantile must be between 0 and 0.5")

    x0 = descriptors[:, 0]
    rho_l, p_l, u_l, rho_r, p_r, u_r = descriptors[:, 1:].T
    strength = (
        np.log(np.maximum(rho_l, rho_r) / np.maximum(np.minimum(rho_l, rho_r), 1e-6))
        + np.log(np.maximum(p_l, p_r) / np.maximum(np.minimum(p_l, p_r), 1e-6))
        + np.abs(u_l - u_r)
    )
    strength_cut = float(np.quantile(strength, strength_quantile))
    position_lo, position_hi = np.quantile(
        x0, (position_tail_quantile, 1.0 - position_tail_quantile)
    )
    high_strength = strength >= strength_cut
    extreme_position = (x0 <= position_lo) | (x0 >= position_hi)

    core = np.flatnonzero(~high_strength & ~extreme_position)
    strength_ood = np.flatnonzero(high_strength & ~extreme_position)
    position_ood = np.flatnonzero(~high_strength & extreme_position)
    joint_ood = np.flatnonzero(high_strength & extreme_position)

    rng = np.random.default_rng(seed)
    shuffled_core = rng.permutation(core)
    id_val_count = int(round(0.1 * len(shuffled_core)))
    id_test_count = int(round(0.1 * len(shuffled_core)))
    train_end = len(shuffled_core) - id_val_count - id_test_count
    train = shuffled_core[:train_end]
    val = shuffled_core[train_end : train_end + id_val_count]
    test = shuffled_core[train_end + id_val_count :]

    named_sets = {
        "train": train,
        "val": val,
        "test": test,
        "strength_ood_test": strength_ood,
        "position_ood_test": position_ood,
        "joint_ood_test": joint_ood,
    }
    seen: set[int] = set()
    for name, indices in named_sets.items():
        current = set(map(int, indices))
        overlap = seen.intersection(current)
        if overlap:
            raise RuntimeError(f"Strict split overlap at {name}: {len(overlap)} cases")
        seen.update(current)
    if len(seen) != len(descriptors):
        raise RuntimeError("Strict splits do not cover every case exactly once")

    manifest = {
        "schema": "pdebench_1d_strict_id_ood_v1",
        "seed": seed,
        "split_basis": "initial_descriptors_only",
        "strength_quantile": strength_quantile,
        "strength_threshold": strength_cut,
        "position_tail_quantile": position_tail_quantile,
        "position_thresholds": [float(position_lo), float(position_hi)],
    }
    manifest.update({name: indices.astype(int).tolist() for name, indices in named_sets.items()})
    return manifest


def prepare_cache(
    source: str | Path,
    output_dir: str | Path,
    spatial_size: int = 256,
    time_stride: int = 4,
    batch_size: int = 16,
    spatial_reduction: str = "conservative",
    overwrite: bool = False,
) -> Path:
    """Build a float32, memory-mappable development cache.

    Spatial restriction uses conservative cell averaging by default. Linear
    interpolation is available for target sizes that do not divide the source
    grid. Time restriction keeps exact stored frames. The complete source file
    is never materialized.
    """
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if spatial_reduction not in {"conservative", "linear"}:
        raise ValueError("spatial_reduction must be 'conservative' or 'linear'")
    suffix = "" if spatial_reduction == "conservative" else f"_{spatial_reduction}"
    data_path = output_dir / f"fields_x{spatial_size}_ts{time_stride}{suffix}.npy"
    metadata_path = output_dir / "metadata.json"
    split_path = output_dir / "splits.json"
    descriptor_path = output_dir / "initial_descriptors.npy"

    if data_path.exists() and metadata_path.exists() and split_path.exists() and not overwrite:
        return data_path

    with h5py.File(source, "r") as handle:
        case_count, frame_count, source_x = handle[FIELD_NAMES[0]].shape
        time_indices = np.arange(0, frame_count, time_stride, dtype=np.int64)
        x_source = handle["x-coordinate"][:].astype(np.float32)
        linear_identity = spatial_reduction == "linear" and spatial_size == source_x
        if spatial_reduction == "conservative":
            if source_x % spatial_size:
                raise ValueError(f"Source grid {source_x} is not divisible by {spatial_size}")
            pool = source_x // spatial_size
            x = x_source.reshape(spatial_size, pool).mean(axis=1)
        else:
            pool = None
            # A full-resolution cache must retain the exact stored grid and
            # field values.  Re-interpolating onto a freshly generated grid
            # can introduce small numerical changes even at identical size.
            x = (
                x_source.copy()
                if linear_identity
                else np.linspace(x_source[0], x_source[-1], spatial_size, dtype=np.float32)
            )
        raw_t = handle["t-coordinate"][:frame_count].astype(np.float32)
        t = raw_t[time_indices]
        shape = (case_count, len(time_indices), spatial_size, len(FIELD_NAMES))
        target = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32, shape=shape)

        all_cases = np.arange(case_count)
        for batch_number, case_ids in enumerate(_batched(all_cases, batch_size)):
            for channel, name in enumerate(FIELD_NAMES):
                raw = handle[name][case_ids[0] : case_ids[-1] + 1, time_indices, :]
                raw = np.asarray(raw, dtype=np.float32)
                if spatial_reduction == "conservative":
                    reduced = raw.reshape(
                        len(case_ids), len(time_indices), spatial_size, pool
                    ).mean(axis=-1)
                elif linear_identity:
                    reduced = raw
                else:
                    reduced = linear_resample_1d(raw, x_source, x)
                target[case_ids, ..., channel] = reduced
            if batch_number % 25 == 0:
                target.flush()
                print(f"prepared {case_ids[-1] + 1}/{case_count} cases", flush=True)
        target.flush()

    descriptors = initial_descriptors(np.asarray(target[:, 0]), x)
    np.save(descriptor_path, descriptors)
    splits = make_split_manifest(descriptors)
    split_path.write_text(json.dumps(splits, indent=2))

    train_ids = np.asarray(splits["train"], dtype=np.int64)
    sums = np.zeros(3, dtype=np.float64)
    square_sums = np.zeros(3, dtype=np.float64)
    element_count = 0
    for ids in _batched(np.sort(train_ids), 128):
        values = np.asarray(target[ids], dtype=np.float64)
        sums += values.sum(axis=(0, 1, 2))
        square_sums += np.square(values).sum(axis=(0, 1, 2))
        element_count += values.shape[0] * values.shape[1] * values.shape[2]
    mean = sums / element_count
    std = np.sqrt(np.maximum(square_sums / element_count - mean**2, 1e-12))

    metadata = {
        "source": str(source),
        "cache": str(data_path),
        "shape": list(shape),
        "field_names": list(FIELD_NAMES),
        "x": x.astype(float).tolist(),
        "t": t.astype(float).tolist(),
        "time_indices": time_indices.astype(int).tolist(),
        "spatial_reduction": spatial_reduction,
        "spatial_pool": int(pool) if pool is not None else None,
        "normalization_mean": mean.astype(float).tolist(),
        "normalization_std": std.astype(float).tolist(),
        "gamma": 5.0 / 3.0,
        "boundary_condition": "transmissive",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return data_path


class PDEBench1DCache(Dataset):
    """Read selected trajectories from a NumPy memmap cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        split_manifest: str = "splits.json",
    ):
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads((self.cache_dir / "metadata.json").read_text())
        self.split_manifest = split_manifest
        self.splits = json.loads((self.cache_dir / split_manifest).read_text())
        if split not in self.splits or not isinstance(self.splits[split], list):
            choices = tuple(key for key, value in self.splits.items() if isinstance(value, list))
            raise KeyError(f"Unknown split {split!r}; choices: {choices}")
        self.case_ids = np.asarray(self.splits[split], dtype=np.int64)
        self.data_path = Path(self.metadata["cache"])
        self._data = None
        self.mean = np.asarray(
            self.splits.get("normalization_mean", self.metadata["normalization_mean"]),
            dtype=np.float32,
        )
        self.std = np.asarray(
            self.splits.get("normalization_std", self.metadata["normalization_std"]),
            dtype=np.float32,
        )

    @property
    def data(self):
        if self._data is None:
            self._data = np.load(self.data_path, mmap_mode="r")
        return self._data

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_id = int(self.case_ids[index])
        physical = np.array(self.data[case_id], dtype=np.float32, copy=True)
        normalized = (physical - self.mean) / self.std
        return {
            "case_id": torch.tensor(case_id, dtype=torch.long),
            "field": torch.from_numpy(normalized),
        }


def load_coordinates(cache_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = json.loads((Path(cache_dir) / "metadata.json").read_text())
    return torch.tensor(metadata["x"], dtype=torch.float32), torch.tensor(
        metadata["t"], dtype=torch.float32
    )
