"""Single-GPU training and evaluation for the PDEBench 1D benchmark."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import PDEBench1DCache, load_coordinates
from .models import build_model, shock_sensor, uniform_time_indices


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def direct_loss(prediction: torch.Tensor, target: torch.Tensor, shock_weight: float) -> torch.Tensor:
    sensor = shock_sensor(target)
    weights = 1.0 + shock_weight * sensor[..., None]
    return ((prediction - target).square() * weights).mean()


def validation_objective(metrics: dict, mode: str) -> float:
    """Choose checkpoints using shock structure, not only global field error."""
    if mode == "global_l2re":
        return float(metrics["l2re"])
    if mode == "shock_visual":
        # Stronger emphasis on the pressure shock band and jump geometry.
        return float(
            0.20 * metrics["l2re"]
            + 0.15 * metrics["pressure_centered_l2re"]
            + 0.30 * metrics["pressure_shock_band_dynamic_range_nmae"]
            + 0.15 * (metrics["pressure_gradient_transport_error_cells"] / 100.0)
            + 0.10 * metrics["pressure_effective_gradient_width_relative_error"]
            + 0.10 * metrics["pressure_overshoot_mae"]
        )
    # All terms are dimensionless after evaluation.  Transport is normalized
    # by a conservative 100-cell reference so it cannot dominate the score.
    return float(
        0.35 * metrics["l2re"]
        + 0.25 * metrics["pressure_centered_l2re"]
        + 0.25 * metrics["pressure_shock_band_dynamic_range_nmae"]
        + 0.15 * (metrics["pressure_gradient_transport_error_cells"] / 100.0)
    )


class MetricAccumulator:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float64)
        self.std = torch.as_tensor(std, dtype=torch.float64)
        self.error_sq = torch.zeros(3, dtype=torch.float64)
        self.target_sq = torch.zeros(3, dtype=torch.float64)
        self.abs_error = torch.zeros(3, dtype=torch.float64)
        self.max_abs_error = torch.zeros(3, dtype=torch.float64)
        self.trajectory_max_abs_error = torch.zeros(3, dtype=torch.float64)
        self.trajectory_max_abs_error_all = 0.0
        self.trajectory_count = 0
        self.count = 0
        self.wave_position_error = 0.0
        self.wave_count = 0
        self.pressure_gradient_error = 0.0
        self.pressure_gradient_count = 0
        self.pressure_tv_absolute_error = 0.0
        self.pressure_tv_target = 0.0
        self.pressure_overshoot = 0.0
        self.pressure_overshoot_count = 0
        self.pressure_gradient_transport_error = 0.0
        self.pressure_gradient_transport_count = 0
        self.pressure_centered_error_sq = 0.0
        self.pressure_centered_target_sq = 0.0
        self.pressure_profile_nmae = 0.0
        self.pressure_profile_count = 0
        self.pressure_shock_band_nmae = 0.0
        self.pressure_smooth_region_nmae = 0.0
        self.pressure_shock_band_count = 0
        self.pressure_smooth_region_count = 0
        self.shock_band_points = 0
        self.active_profile_points = 0
        self.pressure_wrong_sign_mass = 0.0
        self.pressure_true_gradient_mass = 0.0
        self.pressure_effective_width_error = 0.0
        self.pressure_effective_width_count = 0
        self.pressure_excess_tv = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().double() * self.std + self.mean
        target = target.detach().cpu().double() * self.std + self.mean
        error = prediction - target
        self.error_sq += error.square().sum(dim=(0, 1, 2))
        self.target_sq += target.square().sum(dim=(0, 1, 2))
        self.abs_error += error.abs().sum(dim=(0, 1, 2))
        self.max_abs_error = torch.maximum(
            self.max_abs_error, error.abs().amax(dim=(0, 1, 2))
        )
        trajectory_max = error.abs().amax(dim=(1, 2))
        self.trajectory_max_abs_error += trajectory_max.sum(dim=0)
        self.trajectory_max_abs_error_all += trajectory_max.amax(dim=-1).sum().item()
        self.trajectory_count += target.shape[0]
        self.count += int(np.prod(target.shape[:3]))

        pred_grad = torch.diff(prediction[..., 1], dim=-1).abs()
        true_grad = torch.diff(target[..., 1], dim=-1).abs()
        pred_signed_grad = torch.diff(prediction[..., 1], dim=-1)
        true_signed_grad = torch.diff(target[..., 1], dim=-1)
        pressure_prediction = prediction[..., 1]
        pressure_target = target[..., 1]
        pressure_range = pressure_target.amax(dim=-1) - pressure_target.amin(dim=-1)
        relative_range = pressure_range / pressure_target.abs().mean(dim=-1).clamp_min(1e-8)
        self.pressure_gradient_error += (pred_grad - true_grad).abs().sum().item()
        self.pressure_gradient_count += pred_grad.numel()
        pred_tv = pred_grad.sum(dim=-1)
        true_tv = true_grad.sum(dim=-1)
        self.pressure_tv_absolute_error += (pred_tv - true_tv).abs().sum().item()
        self.pressure_tv_target += true_tv.sum().item()
        # Shape metrics ignore nearly constant frames whose dynamic range is
        # below 1% of the local pressure scale; they are still included in the
        # global field metrics above.
        active = relative_range > 1e-2
        if active.any():
            pred_mass = pred_grad / pred_tv[..., None].clamp_min(1e-8)
            true_mass = true_grad / true_tv[..., None].clamp_min(1e-8)
            transport = (pred_mass.cumsum(dim=-1) - true_mass.cumsum(dim=-1)).abs().sum(dim=-1)
            self.pressure_gradient_transport_error += transport[active].sum().item()
            self.pressure_gradient_transport_count += active.sum().item()
            centered_prediction = pressure_prediction - pressure_prediction.mean(
                dim=-1, keepdim=True
            )
            centered_target = pressure_target - pressure_target.mean(dim=-1, keepdim=True)
            self.pressure_centered_error_sq += (
                centered_prediction[active] - centered_target[active]
            ).square().sum().item()
            self.pressure_centered_target_sq += centered_target[active].square().sum().item()

            normalized_abs_error = (pressure_prediction - pressure_target).abs() / pressure_range[
                ..., None
            ].clamp_min(1e-8)
            self.pressure_profile_nmae += normalized_abs_error[active].mean(dim=-1).sum().item()
            self.pressure_profile_count += active.sum().item()

            true_node_gradient = torch.cat((true_grad, true_grad[..., -1:]), dim=-1)
            shock_seed = true_node_gradient >= 0.2 * true_node_gradient.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            flat_seed = shock_seed.reshape(-1, 1, shock_seed.shape[-1]).float()
            shock_band = torch.nn.functional.max_pool1d(
                flat_seed, kernel_size=5, stride=1, padding=2
            ).reshape_as(shock_seed).bool()
            shock_band &= active[..., None]
            smooth_region = active[..., None] & ~shock_band
            self.pressure_shock_band_nmae += normalized_abs_error[shock_band].sum().item()
            self.pressure_smooth_region_nmae += normalized_abs_error[smooth_region].sum().item()
            self.pressure_shock_band_count += shock_band.sum().item()
            self.pressure_smooth_region_count += smooth_region.sum().item()
            self.shock_band_points += shock_band.sum().item()
            self.active_profile_points += active.sum().item() * pressure_target.shape[-1]

            wrong_sign = pred_signed_grad * true_signed_grad < 0
            self.pressure_wrong_sign_mass += true_grad[wrong_sign].sum().item()
            self.pressure_true_gradient_mass += true_grad.sum().item()
            pred_width = pred_tv.square() / pred_grad.square().sum(dim=-1).clamp_min(1e-8)
            true_width = true_tv.square() / true_grad.square().sum(dim=-1).clamp_min(1e-8)
            self.pressure_effective_width_error += (
                (pred_width[active] - true_width[active]).abs()
                / true_width[active].clamp_min(1e-8)
            ).sum().item()
            self.pressure_effective_width_count += active.sum().item()
            self.pressure_excess_tv += torch.relu(pred_tv - true_tv).sum().item()
        lower = target[..., 1].amin(dim=-1, keepdim=True)
        upper = target[..., 1].amax(dim=-1, keepdim=True)
        overshoot = torch.relu(prediction[..., 1] - upper) + torch.relu(
            lower - prediction[..., 1]
        )
        self.pressure_overshoot += overshoot.sum().item()
        self.pressure_overshoot_count += overshoot.numel()
        pred_pos = self._ordered_peaks(pred_grad, peak_count=3)
        true_pos = self._ordered_peaks(true_grad, peak_count=3)
        self.wave_position_error += (pred_pos - true_pos).abs().double().sum().item()
        self.wave_count += pred_pos.numel()

    @staticmethod
    def _ordered_peaks(gradient: torch.Tensor, peak_count: int) -> torch.Tensor:
        shape = gradient.shape
        flat = gradient.reshape(-1, 1, shape[-1])
        local_max = torch.nn.functional.max_pool1d(flat, kernel_size=9, stride=1, padding=4)
        peak_scores = torch.where(flat >= local_max, flat, torch.full_like(flat, -1.0))
        positions = peak_scores.topk(peak_count, dim=-1).indices.squeeze(1)
        return positions.sort(dim=-1).values.reshape(*shape[:-1], peak_count)

    def compute(self) -> dict:
        names = ("density", "pressure", "Vx")
        per_l2 = torch.sqrt(self.error_sq / self.target_sq.clamp_min(1e-30))
        mse = self.error_sq / self.count
        mae = self.abs_error / self.count
        return {
            "l2re": float(torch.sqrt(self.error_sq.sum() / self.target_sq.sum())),
            "mse": float(self.error_sq.sum() / (self.count * 3)),
            "mae": float(self.abs_error.sum() / (self.count * 3)),
            "linf": float(self.max_abs_error.max()),
            "mean_trajectory_linf": self.trajectory_max_abs_error_all
            / max(self.trajectory_count, 1),
            "per_variable_l2re": {name: float(per_l2[i]) for i, name in enumerate(names)},
            "per_variable_mse": {name: float(mse[i]) for i, name in enumerate(names)},
            "per_variable_mae": {name: float(mae[i]) for i, name in enumerate(names)},
            "per_variable_linf": {
                name: float(self.max_abs_error[i]) for i, name in enumerate(names)
            },
            "per_variable_mean_trajectory_linf": {
                name: float(self.trajectory_max_abs_error[i] / max(self.trajectory_count, 1))
                for i, name in enumerate(names)
            },
            "ordered_top3_pressure_wave_error_cells": self.wave_position_error / self.wave_count,
            "pressure_gradient_mae": self.pressure_gradient_error
            / self.pressure_gradient_count,
            "pressure_total_variation_relative_error": self.pressure_tv_absolute_error
            / max(self.pressure_tv_target, 1e-30),
            "pressure_overshoot_mae": self.pressure_overshoot
            / self.pressure_overshoot_count,
            "pressure_gradient_transport_error_cells": self.pressure_gradient_transport_error
            / max(self.pressure_gradient_transport_count, 1),
            "pressure_centered_l2re": (
                self.pressure_centered_error_sq
                / max(self.pressure_centered_target_sq, 1e-30)
            )
            ** 0.5,
            "pressure_profile_dynamic_range_nmae": self.pressure_profile_nmae
            / max(self.pressure_profile_count, 1),
            "pressure_shock_band_dynamic_range_nmae": self.pressure_shock_band_nmae
            / max(self.pressure_shock_band_count, 1),
            "pressure_smooth_region_dynamic_range_nmae": self.pressure_smooth_region_nmae
            / max(self.pressure_smooth_region_count, 1),
            "shock_band_cell_fraction": self.shock_band_points
            / max(self.active_profile_points, 1),
            "pressure_gradient_wrong_sign_mass_fraction": self.pressure_wrong_sign_mass
            / max(self.pressure_true_gradient_mass, 1e-30),
            "pressure_effective_gradient_width_relative_error": self.pressure_effective_width_error
            / max(self.pressure_effective_width_count, 1),
            "pressure_excess_tv_relative": self.pressure_excess_tv
            / max(self.pressure_tv_target, 1e-30),
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    x: torch.Tensor,
    t: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    flow_steps: int,
) -> tuple[dict, float]:
    model.eval()
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
        amp_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            if model_name == "swfmo":
                prediction, _ = model.predict(initial, x, physical_t, steps=flow_steps)
            else:
                prediction = model(initial, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - started
        total_cases += len(field)
        metrics.update(prediction.float(), target)
    result = metrics.compute()
    result["inference_ms_per_trajectory"] = 1000.0 * total_time / total_cases
    return result, result["l2re"]


def limited_loader(dataset, batch_size: int, workers: int, max_cases: int | None, shuffle: bool):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("swfmo",),
        required=True,
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-manifest", default="splits.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--train-cases", type=int)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--shock-weight", type=float, default=2.0)
    parser.add_argument("--endpoint-weight", type=float, default=0.2)
    parser.add_argument("--profile-weight", type=float, default=0.0)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--overshoot-weight", type=float, default=0.0)
    parser.add_argument("--midpoint-endpoint", action="store_true")
    parser.add_argument("--endpoint-steps", type=int, default=0)
    parser.add_argument("--transport-weight", type=float, default=0.0)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-steps", type=int, default=0)
    parser.add_argument("--wavelet-weight", type=float, default=0.0)
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-model", choices=("swfmo",))
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--validate-initial", action="store_true")
    parser.add_argument("--endpoint-time-samples", type=int, default=0)
    parser.add_argument("--hard-case-weight", type=float, default=0.0)
    parser.add_argument("--time-weight", type=float, default=0.0)
    parser.add_argument("--pressure-band-weight", type=float, default=0.0)
    parser.add_argument("--pressure-band-gradient-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-objective",
        choices=("global_l2re", "shock_balanced", "shock_visual"),
        default="global_l2re",
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()

    seed_everything(args.seed)
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This benchmark requires exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = PDEBench1DCache(args.cache_dir, "train", args.split_manifest)
    val_set = PDEBench1DCache(args.cache_dir, "val", args.split_manifest)
    test_set = PDEBench1DCache(args.cache_dir, "test", args.split_manifest)
    train_loader = limited_loader(train_set, args.batch_size, args.workers, args.train_cases, True)
    val_loader = limited_loader(val_set, args.batch_size, args.workers, args.val_cases, False)
    test_loader = limited_loader(test_set, args.batch_size, args.workers, None, False)
    x, t = load_coordinates(args.cache_dir)
    x, t = x.to(device), t.to(device)

    time_count = len(t) - 1
    model = build_model(args.model, time_count=time_count).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(initial_checkpoint["model"], strict=False)
        print(
            json.dumps(
                {
                    "initialized_from": args.init_checkpoint,
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                }
            ),
            flush=True,
        )
    teacher = None
    if args.teacher_checkpoint:
        teacher_model = args.teacher_model or args.model
        teacher = build_model(teacher_model, time_count=time_count).to(device)
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(teacher_checkpoint["model"])
        teacher.requires_grad_(False)
        teacher.eval()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    start_epoch = 1
    best_validation_objective = float("inf")
    history = []
    elapsed_training_seconds = 0.0
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        best_validation_objective = float(checkpoint["best_validation_objective"])
        start_epoch = checkpoint["epoch"] + 1
        elapsed_training_seconds = float(checkpoint.get("training_seconds", 0.0))
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
    elif args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_validation_objective = float(
            checkpoint.get("best_validation_objective", checkpoint["best_l2re"])
        )
        start_epoch = checkpoint["epoch"] + 1

    if args.validate_initial and start_epoch == 1:
        initial_validation, _initial_l2re = evaluate(
            model,
            val_loader,
            args.model,
            x,
            t,
            train_set.mean,
            train_set.std,
            device,
            args.flow_steps,
        )
        best_validation_objective = validation_objective(
            initial_validation, args.selection_objective
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 0,
                "best_l2re": best_validation_objective,
                "best_validation_objective": best_validation_objective,
                "args": vars(args),
            },
            checkpoint_path,
        )
        history.append({"epoch": 0, "validation": initial_validation})

    print(
        json.dumps(
            {
                "model": args.model,
                "parameters": parameter_count,
                "trainable_parameters": trainable_parameter_count,
                "device": str(device),
                "visible_gpus": torch.cuda.device_count(),
                "train_cases": len(train_loader.dataset),
                "val_cases": len(val_loader.dataset),
                "test_cases": len(test_loader.dataset),
            }
        ),
        flush=True,
    )
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_parts = {}
        seen = 0
        for batch in train_loader:
            field = batch["field"].to(device, non_blocking=True)
            initial = field[:, 0]
            target = field[:, 1:]
            physical_t = t[None, 1:].expand(len(field), -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                if args.model == "swfmo":
                    teacher_target = None
                    if teacher is not None:
                        endpoint_indices = uniform_time_indices(
                            target.shape[1], args.endpoint_time_samples, target.device
                        )
                        with torch.no_grad():
                            teacher_target, _ = teacher.predict(
                                initial,
                                x,
                                physical_t.index_select(1, endpoint_indices),
                                steps=args.flow_steps,
                            )
                    loss, parts = model.flow_matching_loss(
                        initial,
                        target,
                        x,
                        physical_t,
                        shock_weight=args.shock_weight,
                        endpoint_weight=args.endpoint_weight,
                        profile_weight=args.profile_weight,
                        temporal_weight=args.temporal_weight,
                        overshoot_weight=args.overshoot_weight,
                        midpoint_endpoint=args.midpoint_endpoint,
                        endpoint_steps=args.endpoint_steps,
                        transport_weight=args.transport_weight,
                        tv_weight=args.tv_weight,
                        consistency_weight=args.consistency_weight,
                        consistency_steps=args.consistency_steps,
                        wavelet_weight=args.wavelet_weight,
                        wavelet_levels=args.wavelet_levels,
                        teacher_target=teacher_target,
                        teacher_weight=args.teacher_weight,
                        endpoint_time_samples=args.endpoint_time_samples,
                        hard_case_weight=args.hard_case_weight,
                        time_weight=args.time_weight,
                        pressure_band_weight=args.pressure_band_weight,
                        pressure_band_gradient_weight=args.pressure_band_gradient_weight,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        }
        if epoch_parts:
            record["loss_parts"] = {name: value / seen for name, value in epoch_parts.items()}
        if epoch % args.val_every == 0 or epoch == 1 or epoch == args.epochs:
            validation, val_l2re = evaluate(
                model,
                val_loader,
                args.model,
                x,
                t,
                train_set.mean,
                train_set.std,
                device,
                args.flow_steps,
            )
            record["validation"] = validation
            validation_score = validation_objective(validation, args.selection_objective)
            record["validation_objective"] = validation_score
            if validation_score < best_validation_objective:
                best_validation_objective = validation_score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_l2re": best_validation_objective,
                        "best_validation_objective": validation_score,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
        history.append(record)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_validation_objective": best_validation_objective,
                "training_seconds": elapsed_training_seconds
                + time.perf_counter()
                - training_started,
                "args": vars(args),
            },
            last_checkpoint_path,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_l2re": best_validation_objective,
                    "best_validation_objective": best_validation_objective,
                    "args": vars(args),
                },
                output_dir / f"epoch_{epoch:03d}.pt",
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)

    training_seconds = elapsed_training_seconds + time.perf_counter() - training_started
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics, _ = evaluate(
        model,
        test_loader,
        args.model,
        x,
        t,
        train_set.mean,
        train_set.std,
        device,
        args.flow_steps,
    )
    result = {
        "model": args.model,
        "parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "best_epoch": checkpoint["epoch"],
        "best_validation_objective": checkpoint.get(
            "best_validation_objective", checkpoint["best_l2re"]
        ),
        "selection_objective": args.selection_objective,
        "training_seconds": training_seconds,
        "test": test_metrics,
        "configuration": vars(args),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
