"""Train a metric-aligned residual flow on top of a frozen SWFMO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import PDEBench1DCache, load_coordinates
from .models import build_model
from .residual_flow import MetricAlignedShockResidualFlow
from .train import MetricAccumulator, validation_objective


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    max_cases: int | None = None,
) -> DataLoader:
    if max_cases is not None and max_cases < len(dataset):
        dataset = Subset(dataset, range(max_cases))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(
    base_model,
    corrector,
    loader,
    x,
    t,
    mean,
    std,
    device,
    base_steps: int,
    residual_steps: int,
) -> dict:
    base_model.eval()
    corrector.eval()
    metrics = MetricAccumulator(mean, std)
    total_time = 0.0
    total_cases = 0
    for batch in loader:
        field = batch["field"].to(device, non_blocking=True)
        initial = field[:, 0]
        target = field[:, 1:]
        physical_t = t[None, 1:].expand(len(field), -1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            base, shock_map = base_model.predict(
                initial, x, physical_t, steps=base_steps
            )
            prediction = corrector.refine(
                base,
                shock_map,
                initial,
                x,
                physical_t,
                steps=residual_steps,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - started
        total_cases += len(field)
        metrics.update(prediction.float(), target)
    result = metrics.compute()
    result["inference_ms_per_trajectory"] = 1000.0 * total_time / total_cases
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-manifest", default="splits.json")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--base-model",
        choices=("swfmo",),
        default="swfmo",
    )
    parser.add_argument("--init-corrector-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-cases", type=int)
    parser.add_argument("--val-cases", type=int)
    parser.add_argument("--test-cases", type=int)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--time-samples", type=int, default=12)
    parser.add_argument("--base-steps", type=int, default=2)
    parser.add_argument("--residual-steps", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--wavelet-levels", type=int, default=2)
    parser.add_argument("--shock-weight", type=float, default=2.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--gradient-transport-weight", type=float, default=0.0)
    parser.add_argument("--wavelet-weight", type=float, default=0.1)
    parser.add_argument("--overshoot-weight", type=float, default=0.05)
    parser.add_argument("--robust-weight", type=float, default=0.0)
    parser.add_argument("--hard-case-weight", type=float, default=0.0)
    parser.add_argument("--robust-shock-weight", type=float, default=2.0)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument(
        "--selection-objective",
        choices=("global_l2re", "shock_balanced", "shock_visual"),
        default="global_l2re",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Residual-flow training requires exactly one visible GPU, found "
            f"{torch.cuda.device_count()}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = PDEBench1DCache(args.cache_dir, "train", args.split_manifest)
    val_set = PDEBench1DCache(args.cache_dir, "val", args.split_manifest)
    test_set = PDEBench1DCache(args.cache_dir, "test", args.split_manifest)
    train_loader = make_loader(
        train_set, args.batch_size, args.workers, True, args.train_cases
    )
    val_loader = make_loader(
        val_set, args.batch_size, args.workers, False, args.val_cases
    )
    test_loader = make_loader(
        test_set, args.batch_size, args.workers, False, args.test_cases
    )
    x, t = load_coordinates(args.cache_dir)
    x, t = x.to(device), t.to(device)

    base_model = build_model(args.base_model, time_count=len(t) - 1).to(device)
    base_checkpoint = torch.load(args.base_checkpoint, map_location=device)
    base_model.load_state_dict(base_checkpoint["model"])
    base_model.requires_grad_(False)
    base_model.eval()

    corrector = MetricAlignedShockResidualFlow(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        slices=args.slices,
        wavelet_levels=args.wavelet_levels,
    ).to(device)
    if args.init_corrector_checkpoint:
        initial_corrector = torch.load(
            args.init_corrector_checkpoint, map_location=device
        )
        corrector.load_state_dict(initial_corrector["corrector"])
    parameter_count = sum(parameter.numel() for parameter in corrector.parameters())
    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    std = torch.as_tensor(train_set.std, device=device)
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    start_epoch = 1
    elapsed_training_seconds = 0.0
    history = []
    if args.resume and last_checkpoint_path.exists():
        resume_checkpoint = torch.load(last_checkpoint_path, map_location=device)
        corrector.load_state_dict(resume_checkpoint["corrector"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_score = float(resume_checkpoint["best_score"])
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        elapsed_training_seconds = float(
            resume_checkpoint.get("training_seconds", 0.0)
        )
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
        initial_validation = history[0].get("validation") if history else None
    else:
        initial_validation = evaluate(
            base_model,
            corrector,
            val_loader,
            x,
            t,
            train_set.mean,
            train_set.std,
            device,
            args.base_steps,
            args.residual_steps,
        )
        best_score = validation_objective(initial_validation, args.selection_objective)
        best_epoch = 0
        torch.save(
            {
                "corrector": corrector.state_dict(),
                "epoch": 0,
                "best_score": best_score,
                "best_validation_objective": best_score,
                "selection_objective": args.selection_objective,
                "base_checkpoint": args.base_checkpoint,
                "args": vars(args),
            },
            checkpoint_path,
        )
        history = [{"epoch": 0, "validation": initial_validation}]
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(
        json.dumps(
            {
                "model": "metric_aligned_shock_residual_flow",
                "corrector_parameters": parameter_count,
                "base_parameters": sum(p.numel() for p in base_model.parameters()),
                "device": str(device),
                "visible_gpus": torch.cuda.device_count(),
                "train_cases": len(train_loader.dataset),
                "val_cases": len(val_loader.dataset),
                "test_cases": len(test_loader.dataset),
                "initial_validation": initial_validation,
            }
        ),
        flush=True,
    )

    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        corrector.train()
        epoch_loss = 0.0
        epoch_parts: dict[str, float] = {}
        seen = 0
        for batch in train_loader:
            field = batch["field"].to(device, non_blocking=True)
            initial = field[:, 0]
            full_target = field[:, 1:]
            time_count = full_target.shape[1]
            sample_count = min(args.time_samples, time_count)
            indices = torch.randperm(time_count, device=device)[:sample_count].sort().values
            target = full_target.index_select(1, indices)
            physical_t = t[1:].index_select(0, indices)[None].expand(len(field), -1)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                base, shock_map = base_model.predict(
                    initial, x, physical_t, steps=args.base_steps
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss, parts = corrector.flow_matching_loss(
                    base.detach(),
                    shock_map.detach(),
                    initial,
                    target,
                    x,
                    physical_t,
                    std,
                    shock_weight=args.shock_weight,
                    endpoint_weight=args.endpoint_weight,
                    gradient_weight=args.gradient_weight,
                    gradient_transport_weight=args.gradient_transport_weight,
                    wavelet_weight=args.wavelet_weight,
                    overshoot_weight=args.overshoot_weight,
                    robust_weight=args.robust_weight,
                    hard_case_weight=args.hard_case_weight,
                    robust_shock_weight=args.robust_shock_weight,
                    tail_weight=args.tail_weight,
                    tail_fraction=args.tail_fraction,
                    endpoint_steps=args.residual_steps,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(field)
            for name, value in parts.items():
                epoch_parts[name] = epoch_parts.get(name, 0.0) + float(value) * len(field)
            seen += len(field)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / seen,
            "learning_rate": scheduler.get_last_lr()[0],
            "loss_parts": {name: value / seen for name, value in epoch_parts.items()},
        }
        if epoch % args.val_every == 0 or epoch == 1 or epoch == args.epochs:
            validation = evaluate(
                base_model,
                corrector,
                val_loader,
                x,
                t,
                train_set.mean,
                train_set.std,
                device,
                args.base_steps,
                args.residual_steps,
            )
            record["validation"] = validation
            validation_score = validation_objective(validation, args.selection_objective)
            record["validation_objective"] = validation_score
            if validation_score < best_score:
                best_score = validation_score
                best_epoch = epoch
                torch.save(
                    {
                        "corrector": corrector.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "best_validation_objective": best_score,
                        "selection_objective": args.selection_objective,
                        "base_checkpoint": args.base_checkpoint,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(
            {
                "corrector": corrector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "training_seconds": elapsed_training_seconds
                + time.perf_counter()
                - training_started,
                "args": vars(args),
            },
            last_checkpoint_path,
        )
        print(json.dumps(record), flush=True)

    training_seconds = elapsed_training_seconds + time.perf_counter() - training_started
    checkpoint = torch.load(checkpoint_path, map_location=device)
    corrector.load_state_dict(checkpoint["corrector"])
    test_metrics = None
    if not args.skip_test:
        test_metrics = evaluate(
            base_model,
            corrector,
            test_loader,
            x,
            t,
            train_set.mean,
            train_set.std,
            device,
            args.base_steps,
            args.residual_steps,
        )
    result = {
        "model": "metric_aligned_shock_residual_flow",
        "corrector_parameters": parameter_count,
        "base_parameters": sum(p.numel() for p in base_model.parameters()),
        "best_epoch": checkpoint["epoch"],
        "selection_objective": args.selection_objective,
        "best_validation_score": checkpoint.get(
            "best_validation_objective", checkpoint["best_score"]
        ),
        "training_seconds": training_seconds,
        "test": test_metrics,
        "configuration": vars(args),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
