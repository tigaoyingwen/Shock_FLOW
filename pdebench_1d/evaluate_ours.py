"""Evaluate a SWFMO checkpoint plus its metric-aligned residual corrector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import PDEBench1DCache, load_coordinates
from .models import build_model
from .residual_flow import MetricAlignedShockResidualFlow
from .train_residual_flow import evaluate


def load_ours(base_path: str | Path, residual_path: str | Path, time_count: int, device: torch.device):
    base_checkpoint = torch.load(base_path, map_location=device)
    base = build_model("swfmo", time_count=time_count).to(device)
    base.load_state_dict(base_checkpoint["model"])
    base.eval()

    residual_checkpoint = torch.load(residual_path, map_location=device)
    config = residual_checkpoint.get("args", {})
    corrector = MetricAlignedShockResidualFlow(
        hidden_dim=config.get("hidden_dim", 64),
        layers=config.get("layers", 3),
        heads=config.get("heads", 4),
        slices=config.get("slices", 16),
        wavelet_levels=config.get("wavelet_levels", 2),
    ).to(device)
    corrector.load_state_dict(residual_checkpoint["corrector"])
    corrector.eval()
    return base, corrector, base_checkpoint, residual_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-manifest", default="splits.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--base-steps", type=int, default=2)
    parser.add_argument("--residual-steps", type=int, default=2)
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PDEBench1DCache(args.cache_dir, args.split, args.split_manifest)
    evaluated = Subset(dataset, range(args.max_cases)) if args.max_cases else dataset
    loader = DataLoader(
        evaluated,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    x, t = load_coordinates(args.cache_dir)
    x, t = x.to(device), t.to(device)
    base, corrector, base_checkpoint, residual_checkpoint = load_ours(
        args.base_checkpoint, args.residual_checkpoint, len(t) - 1, device
    )
    metrics = evaluate(
        base,
        corrector,
        loader,
        x,
        t,
        dataset.mean,
        dataset.std,
        device,
        args.base_steps,
        args.residual_steps,
    )
    result = {
        "model": "swfmo_plus_metric_aligned_shock_residual_flow",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "split": args.split,
        "evaluated_cases": len(evaluated),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "residual_checkpoint": str(Path(args.residual_checkpoint).resolve()),
        "base_epoch": base_checkpoint.get("epoch"),
        "residual_epoch": residual_checkpoint.get("epoch"),
        "base_steps": args.base_steps,
        "residual_steps": args.residual_steps,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
