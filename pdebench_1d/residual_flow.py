"""Metric-aligned residual flow correction for discontinuous trajectories."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .models import (
    FiLMFlowBlock,
    differentiable_riemann_descriptor,
    haar_detail_pyramid,
    shock_sensor,
)


def _node_gradient(field: torch.Tensor) -> torch.Tensor:
    gradient = torch.diff(field, dim=-2)
    return torch.cat((gradient, gradient[..., -1:, :]), dim=-2)


def _metric_weights(std: torch.Tensor) -> torch.Tensor:
    weights = std.square()
    return weights / weights.mean().clamp_min(1e-8)


def _metric_haar_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    std: torch.Tensor,
    levels: int,
) -> torch.Tensor:
    prediction_details = haar_detail_pyramid(prediction, levels)
    target_details = haar_detail_pyramid(target, levels)
    if not prediction_details:
        return prediction.new_zeros(())
    metric_weights = _metric_weights(std).view(1, 1, 1, -1)
    losses = []
    level_weights = []
    for level, (prediction_detail, target_detail) in enumerate(
        zip(prediction_details, target_details)
    ):
        level_weight = 0.5**level
        losses.append(
            level_weight
            * ((prediction_detail - target_detail).square() * metric_weights).mean()
        )
        level_weights.append(level_weight)
    return sum(losses) / sum(level_weights)


class MetricAlignedShockResidualFlow(nn.Module):
    """Flow matching in the residual space of a frozen shock-flow operator.

    The corrector sees only quantities available at inference: the initial
    state, the frozen operator prediction and shock map, coordinates and time.
    Its zero-initialized head makes the untrained model exactly preserve the
    base prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        layers: int = 3,
        heads: int = 4,
        slices: int = 16,
        wavelet_levels: int = 2,
    ):
        super().__init__()
        self.wavelet_levels = wavelet_levels
        condition_dim = hidden_dim
        self.condition = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )
        # residual, base, initial, base gradient, shock, x, t, flow-time,
        # two shock-frame coordinates and multiscale base details.
        input_channels = 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 2 + 3 * wavelet_levels
        self.stem = nn.Conv1d(input_channels, hidden_dim, 5, padding=2)
        self.blocks = nn.ModuleList(
            [FiLMFlowBlock(hidden_dim, condition_dim, heads, slices) for _ in range(layers)]
        )
        self.velocity_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 3, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def vector_field(
        self,
        residual: torch.Tensor,
        base: torch.Tensor,
        shock_map: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        batch, times, nodes, _ = residual.shape
        descriptor = differentiable_riemann_descriptor(initial, x)
        descriptor_seq = descriptor[:, None].expand(-1, times, -1)
        condition_input = torch.cat(
            (descriptor_seq, physical_t[..., None], flow_t[..., None]), dim=-1
        )
        condition = self.condition(condition_input).reshape(batch * times, -1)

        initial_seq = initial[:, None].expand(-1, times, -1, -1)
        x_seq = x[None, None, :, None].expand(batch, times, -1, -1)
        physical_seq = physical_t[..., None, None].expand(-1, -1, nodes, 1)
        flow_seq = flow_t[..., None, None].expand(-1, -1, nodes, 1)
        dx = (x[1] - x[0]).abs().clamp_min(1e-6)
        similarity = (x[None, None] - descriptor[:, None, 0, None]) / (
            physical_t[..., None] + dx
        )
        shock_frame = torch.stack(
            (torch.tanh(similarity / 8.0), 1.0 / (1.0 + similarity.abs())), dim=-1
        )
        base_details = haar_detail_pyramid(base, self.wavelet_levels)
        features = torch.cat(
            (
                residual,
                base,
                initial_seq,
                _node_gradient(base),
                shock_map[..., None],
                x_seq,
                physical_seq,
                flow_seq,
                shock_frame,
                *base_details,
            ),
            dim=-1,
        )
        hidden = self.stem(
            features.reshape(batch * times, nodes, -1).transpose(1, 2)
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.velocity_head(hidden).transpose(1, 2).reshape(
            batch, times, nodes, 3
        )

    def integrate(
        self,
        base: torch.Tensor,
        shock_map: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        steps: int = 2,
    ) -> torch.Tensor:
        residual = torch.zeros_like(base)
        step_size = 1.0 / steps
        for step in range(steps):
            s0 = torch.full_like(physical_t, step * step_size)
            velocity = self.vector_field(
                residual, base, shock_map, initial, x, physical_t, s0
            )
            midpoint = residual + 0.5 * step_size * velocity
            smid = s0 + 0.5 * step_size
            midpoint_velocity = self.vector_field(
                midpoint, base, shock_map, initial, x, physical_t, smid
            )
            residual = residual + step_size * midpoint_velocity
        return residual

    def refine(
        self,
        base: torch.Tensor,
        shock_map: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        steps: int = 2,
    ) -> torch.Tensor:
        return base + self.integrate(
            base, shock_map, initial, x, physical_t, steps=steps
        )

    def flow_matching_loss(
        self,
        base: torch.Tensor,
        shock_map: torch.Tensor,
        initial: torch.Tensor,
        target: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        std: torch.Tensor,
        shock_weight: float = 2.0,
        endpoint_weight: float = 1.0,
        gradient_weight: float = 0.05,
        gradient_transport_weight: float = 0.0,
        wavelet_weight: float = 0.1,
        overshoot_weight: float = 0.05,
        robust_weight: float = 0.0,
        hard_case_weight: float = 0.0,
        robust_shock_weight: float = 2.0,
        tail_weight: float = 0.0,
        tail_fraction: float = 0.1,
        endpoint_steps: int = 2,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_residual = target - base
        batch, times = target.shape[:2]
        flow_t = torch.rand(batch, times, device=target.device)
        state = flow_t[..., None, None] * target_residual
        velocity = self.vector_field(
            state, base, shock_map, initial, x, physical_t, flow_t
        )
        sensor = shock_sensor(target)
        spatial_weight = 1.0 + shock_weight * sensor[..., None]
        variable_weight = _metric_weights(std).view(1, 1, 1, -1)
        flow_loss = (
            (velocity - target_residual).square() * spatial_weight * variable_weight
        ).mean()

        endpoint_residual = self.integrate(
            base, shock_map, initial, x, physical_t, steps=endpoint_steps
        )
        endpoint = base + endpoint_residual
        endpoint_loss = (
            (endpoint - target).square() * spatial_weight * variable_weight
        ).mean()
        base_physical_error = (
            (base - target).abs() * std.view(1, 1, 1, -1)
        ).mean(dim=(1, 2, 3))
        case_weight = base_physical_error / base_physical_error.mean().clamp_min(1e-6)
        case_weight = case_weight.detach().clamp(0.5, 3.0)
        case_weight = 1.0 + hard_case_weight * (case_weight - 1.0)
        physical_error = (endpoint - target) * std.view(1, 1, 1, -1)
        robust_spatial_weight = 1.0 + robust_shock_weight * sensor[..., None]
        robust_per_case = physical_error.abs()
        robust_per_case = (
            robust_per_case * robust_spatial_weight * case_weight[:, None, None, None]
        ).mean(dim=(1, 2, 3))
        robust_loss = robust_per_case.mean() / std.mean().clamp_min(1e-6)
        point_error = physical_error.abs().mean(dim=-1)
        point_count = point_error.shape[-1] * point_error.shape[-2]
        tail_count = max(1, int(round(point_count * tail_fraction)))
        tail_loss = point_error.flatten(1).topk(tail_count, dim=-1).values.mean()
        pressure_gradient = torch.diff(endpoint[..., 1], dim=2)
        target_pressure_gradient = torch.diff(target[..., 1], dim=2)
        gradient_loss = F.smooth_l1_loss(
            pressure_gradient * std[1], target_pressure_gradient * std[1]
        )
        # Match the spatial distribution of pressure-gradient mass.  This is
        # a differentiable 1-D transport loss: moving a front to its true
        # position lowers the CDF distance, while merely broadening it does
        # not provide the same shortcut as pointwise gradient losses.
        prediction_mass = pressure_gradient.abs()
        target_mass = target_pressure_gradient.abs()
        active = target_mass.sum(dim=2) > 1e-6
        prediction_mass = prediction_mass / prediction_mass.sum(dim=2, keepdim=True).clamp_min(1e-6)
        target_mass = target_mass / target_mass.sum(dim=2, keepdim=True).clamp_min(1e-6)
        transport_per_profile = (
            prediction_mass.cumsum(dim=2) - target_mass.cumsum(dim=2)
        ).abs().mean(dim=2)
        gradient_transport_loss = (
            transport_per_profile[active].mean()
            if active.any()
            else endpoint.new_zeros(())
        )
        wavelet_loss = _metric_haar_loss(
            endpoint, target, std, self.wavelet_levels
        )
        lower = target.amin(dim=2, keepdim=True)
        upper = target.amax(dim=2, keepdim=True)
        overshoot = F.relu(endpoint - upper).square() + F.relu(lower - endpoint).square()
        overshoot_loss = (overshoot * variable_weight).mean()
        total = (
            flow_loss
            + endpoint_weight * endpoint_loss
            + gradient_weight * gradient_loss
            + gradient_transport_weight * gradient_transport_loss
            + wavelet_weight * wavelet_loss
            + overshoot_weight * overshoot_loss
            + robust_weight * robust_loss
            + tail_weight * tail_loss / std.mean().clamp_min(1e-6)
        )
        return total, {
            "flow": flow_loss.detach(),
            "endpoint": endpoint_loss.detach(),
            "gradient": gradient_loss.detach(),
            "gradient_transport": gradient_transport_loss.detach(),
            "wavelet": wavelet_loss.detach(),
            "overshoot": overshoot_loss.detach(),
            "robust": robust_loss.detach(),
            "hard_case_weight": case_weight.mean().detach(),
            "tail": tail_loss.detach(),
        }


class ObservableResidualGate(nn.Module):
    """Input-only gate that controls a frozen residual correction.

    The gate does not see the future field. It uses the base prediction,
    residual, initial condition, local gradients, the predicted shock map and
    coordinates to decide whether each correction component is trustworthy.
    The positive output bias initializes the gate close to one, preserving the
    already trained residual model before gate fine-tuning.
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        layers: int = 2,
        heads: int = 4,
        slices: int = 12,
    ):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.stem = nn.Conv1d(18, hidden_dim, 5, padding=2)
        self.blocks = nn.ModuleList(
            [FiLMFlowBlock(hidden_dim, hidden_dim, heads, slices) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Conv1d(hidden_dim, 3, 1)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base: torch.Tensor,
        residual: torch.Tensor,
        shock_map: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
    ) -> torch.Tensor:
        batch, times, nodes, _ = base.shape
        initial_seq = initial[:, None].expand(-1, times, -1, -1)
        x_seq = x[None, None, :, None].expand(batch, times, -1, -1)
        time_seq = physical_t[..., None, None].expand(-1, -1, nodes, 1)
        descriptor = differentiable_riemann_descriptor(initial, x)
        condition = self.condition(
            torch.cat((descriptor[:, None].expand(-1, times, -1), physical_t[..., None]), dim=-1)
        ).reshape(batch * times, -1)
        features = torch.cat(
            (
                base,
                residual,
                initial_seq,
                _node_gradient(base),
                _node_gradient(initial_seq),
                shock_map[..., None],
                x_seq,
                time_seq,
            ),
            dim=-1,
        )
        hidden = self.stem(features.reshape(batch * times, nodes, -1).transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden, condition)
        logits = self.head(hidden).transpose(1, 2).reshape(batch, times, nodes, 3)
        return 1.0 + torch.tanh(logits)


class MonotoneShockTransportCorrector(nn.Module):
    """Input-observable transport and amplitude correction for sharp waves.

    A positive learned cell metric is integrated into an endpoint-preserving
    monotone sampling map. This moves discontinuities without averaging across
    them. A bounded additive head then corrects state amplitudes. Both heads
    are zero initialized, so the untrained module exactly preserves its input.
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        layers: int = 2,
        heads: int = 4,
        slices: int = 12,
        max_log_stretch: float = 0.75,
        max_additive: float = 0.5,
    ):
        super().__init__()
        self.max_log_stretch = max_log_stretch
        self.max_additive = max_additive
        self.condition = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # base, residual, refined prediction, initial state, two gradients,
        # predicted shock map, coordinate and physical time.
        self.stem = nn.Conv1d(21, hidden_dim, 5, padding=2)
        self.blocks = nn.ModuleList(
            [FiLMFlowBlock(hidden_dim, hidden_dim, heads, slices) for _ in range(layers)]
        )
        self.transport_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Conv1d(hidden_dim, 1, 1)
        )
        self.additive_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Conv1d(hidden_dim, 3, 1)
        )
        nn.init.zeros_(self.transport_head[-1].weight)
        nn.init.zeros_(self.transport_head[-1].bias)
        nn.init.zeros_(self.additive_head[-1].weight)
        nn.init.zeros_(self.additive_head[-1].bias)

    @staticmethod
    def monotone_grid(log_metric: torch.Tensor) -> torch.Tensor:
        """Integrate a positive cell metric into a map on ``[-1, 1]``."""
        if log_metric.shape[-1] < 2:
            raise ValueError("The transport grid requires at least two nodes")
        cell_metric = torch.exp(
            0.5 * (log_metric[..., 1:] + log_metric[..., :-1])
        )
        cumulative = torch.cat(
            (torch.zeros_like(cell_metric[..., :1]), cell_metric.cumsum(dim=-1)),
            dim=-1,
        )
        return -1.0 + 2.0 * cumulative / cumulative[..., -1:].clamp_min(1e-8)

    @staticmethod
    def warp(field: torch.Tensor, sampling_grid: torch.Tensor) -> torch.Tensor:
        """Sample a ``(B,T,N,C)`` field with a normalized 1D grid."""
        batch, times, nodes, channels = field.shape
        source = field.reshape(batch * times, nodes, channels).transpose(1, 2)
        source = source.unsqueeze(2)
        grid_x = sampling_grid.reshape(batch * times, 1, nodes)
        grid = torch.stack((grid_x, torch.zeros_like(grid_x)), dim=-1)
        warped = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.squeeze(2).transpose(1, 2).reshape(batch, times, nodes, channels)

    def forward(
        self,
        base: torch.Tensor,
        residual: torch.Tensor,
        shock_map: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, times, nodes, _ = base.shape
        prediction = base + residual
        initial_seq = initial[:, None].expand(-1, times, -1, -1)
        x_seq = x[None, None, :, None].expand(batch, times, -1, -1)
        time_seq = physical_t[..., None, None].expand(-1, -1, nodes, 1)
        descriptor = differentiable_riemann_descriptor(initial, x)
        condition = self.condition(
            torch.cat(
                (descriptor[:, None].expand(-1, times, -1), physical_t[..., None]),
                dim=-1,
            )
        ).reshape(batch * times, -1)
        features = torch.cat(
            (
                base,
                residual,
                prediction,
                initial_seq,
                _node_gradient(prediction),
                _node_gradient(initial_seq),
                shock_map[..., None],
                x_seq,
                time_seq,
            ),
            dim=-1,
        )
        hidden = self.stem(
            features.reshape(batch * times, nodes, -1).transpose(1, 2)
        )
        for block in self.blocks:
            hidden = block(hidden, condition)

        raw_transport = self.transport_head(hidden).squeeze(1)
        log_metric = self.max_log_stretch * torch.tanh(raw_transport)
        sampling_grid = self.monotone_grid(log_metric).reshape(batch, times, nodes)
        transported = self.warp(prediction, sampling_grid)

        additive_logits = self.additive_head(hidden).transpose(1, 2)
        additive = self.max_additive * torch.tanh(additive_logits)
        additive = additive.reshape(batch, times, nodes, 3)
        corrected = transported + additive
        identity = torch.linspace(
            -1.0, 1.0, nodes, device=base.device, dtype=base.dtype
        )
        return corrected, {
            "sampling_grid": sampling_grid,
            "displacement": sampling_grid - identity,
            "log_metric": log_metric.reshape(batch, times, nodes),
            "additive": additive,
        }
