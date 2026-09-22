"""Render pressure and pressure-gradient profiles for the proposed method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import PDEBench1DCache, load_coordinates
from .evaluate_ours import load_ours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-manifest", default="splits.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--base-steps", type=int, default=2)
    parser.add_argument("--residual-steps", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PDEBench1DCache(args.cache_dir, args.split, args.split_manifest)
    if not 0 <= args.case_index < len(dataset):
        raise IndexError(f"case-index must be in [0, {len(dataset) - 1}]")
    x, t = load_coordinates(args.cache_dir)
    x, t = x.to(device), t.to(device)
    base, corrector, _, _ = load_ours(
        args.base_checkpoint, args.residual_checkpoint, len(t) - 1, device
    )
    sample = dataset[args.case_index]
    field = sample["field"][None].to(device)
    physical_t = t[None, 1:]
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        base_prediction, shock_map = base.predict(
            field[:, 0], x, physical_t, steps=args.base_steps
        )
        prediction = corrector.refine(
            base_prediction,
            shock_map,
            field[:, 0],
            x,
            physical_t,
            steps=args.residual_steps,
        )
    mean = torch.as_tensor(dataset.mean, device=device).view(1, 1, 1, 3)
    std = torch.as_tensor(dataset.std, device=device).view(1, 1, 1, 3)
    target = field[:, 1:] * std + mean
    prediction = prediction * std + mean
    x_np, time_np = x.cpu().numpy(), t.cpu().numpy()
    truth_np, pred_np = target[0].float().cpu().numpy(), prediction[0].float().cpu().numpy()
    frame_indices = np.linspace(0, len(time_np) - 2, 4).round().astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for col, frame in enumerate(frame_indices):
        axes[0, col].plot(x_np, truth_np[frame, :, 1], "k-", lw=2, label="Ground truth")
        axes[0, col].plot(x_np, pred_np[frame, :, 1], "r--", lw=1.5, label="Ours")
        true_gradient = np.diff(truth_np[frame, :, 1]) / np.diff(x_np)
        pred_gradient = np.diff(pred_np[frame, :, 1]) / np.diff(x_np)
        axes[1, col].plot(x_np[:-1], true_gradient, "k-", lw=2)
        axes[1, col].plot(x_np[:-1], pred_gradient, "r--", lw=1.5)
        axes[0, col].set_title(f"t={time_np[frame + 1]:.3f}")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Pressure")
    axes[1, 0].set_ylabel("dPressure/dx")
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    fig.suptitle(f"Ours: case {int(sample['case_id'])}")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
