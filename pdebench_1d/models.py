"""SWFMO and the shared shock-aware flow blocks."""


from __future__ import annotations


import math


import torch


from torch import nn


import torch.nn.functional as F


class PhysicsSliceAttention1D(nn.Module):
    """Physics-attention slice/token/deslice operation used by physics-slice."""

    def __init__(self, dim: int, heads: int = 8, slices: int = 32):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.slices = slices
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.to_x = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_slice = nn.Linear(self.head_dim, slices)
        nn.init.orthogonal_(self.to_slice.weight)
        self.to_qkv = nn.Linear(self.head_dim, 3 * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, dim = hidden.shape
        x = self.to_x(hidden).reshape(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        value = self.to_value(hidden).reshape(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(
            self.to_slice(x) / self.temperature.clamp(0.1, 5.0), dim=-1
        )
        normalizer = weights.sum(dim=2).clamp_min(1e-5)
        tokens = torch.einsum("bhnc,bhng->bhgc", value, weights) / normalizer[..., None]
        query, key, token_value = self.to_qkv(tokens).chunk(3, dim=-1)
        attention = torch.softmax(query @ key.transpose(-1, -2) * self.scale, dim=-1)
        tokens = attention @ token_value
        output = torch.einsum("bhgc,bhng->bhnc", tokens, weights)
        output = output.transpose(1, 2).reshape(batch, nodes, dim)
        return self.out(output), weights


def differentiable_riemann_descriptor(initial: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Recover one interface and two states using only the observed initial field."""
    scale = initial.amax(dim=1, keepdim=True) - initial.amin(dim=1, keepdim=True)
    gradient = torch.diff(initial, dim=1).abs() / scale.clamp_min(1e-4)
    sensor = gradient.sum(dim=-1)
    weights = torch.softmax(sensor / 0.025, dim=-1)
    midpoint = 0.5 * (x[:-1] + x[1:])
    x0 = (weights * midpoint[None]).sum(dim=-1)
    dx = float((x[1] - x[0]).detach())
    left_gate = torch.sigmoid((x0[:, None] - x[None]) / (2.0 * dx))
    right_gate = 1.0 - left_gate
    left = torch.einsum("bn,bnc->bc", left_gate, initial) / left_gate.sum(dim=1, keepdim=True)
    right = torch.einsum("bn,bnc->bc", right_gate, initial) / right_gate.sum(dim=1, keepdim=True)
    return torch.cat((x0[:, None], left, right), dim=-1)


def shock_sensor(field: torch.Tensor) -> torch.Tensor:
    """Normalized multi-variable gradient map in [0, 1]."""
    gradient = torch.diff(field, dim=-2).abs().mean(dim=-1)
    gradient = F.pad(gradient, (0, 1), mode="replicate")
    scale = gradient.amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (gradient / scale).detach()


def pressure_shock_band(field: torch.Tensor, threshold: float = 0.2, kernel_size: int = 5) -> torch.Tensor:
    """Return a dilated mask of the strongest pressure-gradient band."""
    if field.shape[-1] < 2:
        return torch.ones_like(field[..., 0])
    pressure_grad = torch.diff(field[..., 1], dim=-1).abs()
    pressure_grad = F.pad(pressure_grad, (0, 1), mode="replicate")
    peak = pressure_grad.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    seed = pressure_grad >= threshold * peak
    if kernel_size > 1:
        seed = torch.nn.functional.max_pool1d(
            seed.reshape(-1, 1, seed.shape[-1]).float(),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).reshape_as(seed).bool()
    return seed.detach()


def haar_detail_pyramid(field: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Return node-aligned Haar detail maps along the spatial dimension."""
    if levels < 0:
        raise ValueError("levels must be non-negative")
    original_nodes = field.shape[-2]
    current = field
    details = []
    repeat = 1
    for _ in range(levels):
        if current.shape[-2] < 2:
            break
        if current.shape[-2] % 2:
            current = torch.cat((current, current[..., -1:, :]), dim=-2)
        left = current[..., 0::2, :]
        right = current[..., 1::2, :]
        details.append(0.5 * (left - right))
        current = 0.5 * (left + right)
        repeat *= 2
    return [
        detail.repeat_interleave(2 ** (level + 1), dim=-2)[..., :original_nodes, :]
        for level, detail in enumerate(details)
    ]


def haar_detail_loss(
    prediction: torch.Tensor, target: torch.Tensor, levels: int
) -> torch.Tensor:
    """Compare discontinuity-sensitive Haar bands at several spatial scales."""
    prediction_details = haar_detail_pyramid(prediction, levels)
    target_details = haar_detail_pyramid(target, levels)
    if not prediction_details:
        return prediction.new_zeros(())
    losses = []
    weights = []
    for level, (prediction_detail, target_detail) in enumerate(
        zip(prediction_details, target_details)
    ):
        weight = 0.5**level
        losses.append(weight * F.smooth_l1_loss(prediction_detail, target_detail))
        weights.append(weight)
    return sum(losses) / sum(weights)


def uniform_time_indices(times: int, count: int, device: torch.device) -> torch.Tensor:
    """Select deterministic evolution frames for expensive trajectory losses."""
    if count <= 0 or count >= times:
        return torch.arange(times, device=device)
    return torch.linspace(0, times - 1, count, device=device).round().long().unique()


class FiLMFlowBlock(nn.Module):
    def __init__(self, dim: int, condition_dim: int, heads: int, slices: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )
        self.film = nn.Linear(condition_dim, 2 * dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = PhysicsSliceAttention1D(dim, heads, slices)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        update = self.norm1(hidden)
        update = update * (1.0 + gamma[..., None]) + beta[..., None]
        hidden = hidden + self.conv(update)
        node_hidden = hidden.transpose(1, 2)
        update, _ = self.attention(self.norm2(node_hidden))
        node_hidden = node_hidden + update
        node_hidden = node_hidden + self.mlp(self.norm3(node_hidden))
        return node_hidden.transpose(1, 2)


class CharacteristicFlowOperator(nn.Module):
    """Characteristic-conditioned flow matching operator.

    The only inference inputs are the initial field, x, and t. A learned vector
    field transports the repeated initial field to each requested physical time.
    The auxiliary shock map exposes the model's learned if/where representation.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        layers: int = 4,
        heads: int = 8,
        slices: int = 24,
    ):
        super().__init__()
        condition_dim = hidden_dim
        self.condition = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, condition_dim)
        )
        self.stem = nn.Conv1d(9, hidden_dim, 5, padding=2)
        self.blocks = nn.ModuleList(
            [FiLMFlowBlock(hidden_dim, condition_dim, heads, slices) for _ in range(layers)]
        )
        self.velocity_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Conv1d(hidden_dim, 3, 1)
        )
        self.shock_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Conv1d(hidden_dim, 1, 1)
        )

    def vector_field(
        self,
        state: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, times, nodes, _ = state.shape
        descriptor = differentiable_riemann_descriptor(initial, x)
        descriptor = descriptor[:, None].expand(-1, times, -1)
        condition_input = torch.cat((descriptor, physical_t[..., None], flow_t[..., None]), dim=-1)
        condition = self.condition(condition_input).reshape(batch * times, -1)

        initial_seq = initial[:, None].expand(-1, times, -1, -1)
        x_seq = x[None, None, :, None].expand(batch, times, -1, -1)
        physical_seq = physical_t[..., None, None].expand(-1, -1, nodes, 1)
        flow_seq = flow_t[..., None, None].expand(-1, -1, nodes, 1)
        features = torch.cat((state, initial_seq, x_seq, physical_seq, flow_seq), dim=-1)
        hidden = self.stem(features.reshape(batch * times, nodes, -1).transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden, condition)
        velocity = self.velocity_head(hidden).transpose(1, 2).reshape(batch, times, nodes, 3)
        shock_map = self.shock_head(hidden).squeeze(1).reshape(batch, times, nodes)
        return velocity, shock_map

    def flow_matching_loss(
        self,
        initial: torch.Tensor,
        target: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        shock_weight: float = 2.0,
        sensor_weight: float = 0.05,
        endpoint_weight: float = 0.2,
        profile_weight: float = 0.0,
        temporal_weight: float = 0.0,
        overshoot_weight: float = 0.0,
        midpoint_endpoint: bool = False,
        endpoint_steps: int = 0,
        transport_weight: float = 0.0,
        tv_weight: float = 0.0,
        consistency_weight: float = 0.0,
        consistency_steps: int = 0,
        wavelet_weight: float = 0.0,
        wavelet_levels: int = 3,
        teacher_target: torch.Tensor | None = None,
        teacher_weight: float = 0.0,
        endpoint_time_samples: int = 0,
        hard_case_weight: float = 0.0,
        time_weight: float = 0.0,
        pressure_band_weight: float = 0.0,
        pressure_band_gradient_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, times = target.shape[:2]
        source = initial[:, None].expand_as(target)
        flow_t = torch.rand(batch, times, device=target.device)
        interpolation = flow_t[..., None, None]
        state = (1.0 - interpolation) * source + interpolation * target
        target_velocity = target - source
        velocity, shock_logits = self.vector_field(state, initial, x, physical_t, flow_t)
        sensor = shock_sensor(target)
        weights = 1.0 + shock_weight * sensor[..., None]
        # Give difficult trajectories a controlled extra weight.  Global MSE can
        # otherwise be dominated by smooth, easy cases while the shock profile
        # remains under-trained.  The weight is computed from the target only
        # (available during training) and detached before optimization.
        pressure_grad = torch.diff(target[..., 1], dim=2).abs()
        shock_content = torch.log1p(pressure_grad.sum(dim=2)).mean(dim=1)
        shock_z = (shock_content - shock_content.mean()) / shock_content.std().clamp_min(1e-4)
        case_weight = (1.0 + hard_case_weight * shock_z.clamp(-0.5, 1.5)).detach()
        case_weight = case_weight.clamp(0.5, 3.0)
        time_scale = 1.0 + time_weight * (
            physical_t / physical_t.amax(dim=1, keepdim=True).clamp_min(1e-6)
        ) ** 2
        flow_point = ((velocity - target_velocity).square() * weights).mean(dim=-1)
        flow_per_case = (flow_point * time_scale[..., None]).mean(dim=(1, 2))
        flow_loss = (flow_per_case * case_weight).mean()
        sensor_loss = F.binary_cross_entropy_with_logits(shock_logits, sensor)

        endpoint_indices = uniform_time_indices(times, endpoint_time_samples, target.device)
        endpoint_target = target.index_select(1, endpoint_indices)
        endpoint_physical_t = physical_t.index_select(1, endpoint_indices)
        endpoint_sensor = sensor.index_select(1, endpoint_indices)
        endpoint_weights = 1.0 + shock_weight * endpoint_sensor[..., None]
        endpoint_source = initial[:, None].expand_as(endpoint_target)
        if endpoint_steps > 0:
            endpoint, _ = self.integrate(
                initial, x, endpoint_physical_t, steps=endpoint_steps
            )
        else:
            endpoint_flow_t = torch.zeros(
                batch, len(endpoint_indices), device=target.device
            )
            initial_velocity, _ = self.vector_field(
                endpoint_source, initial, x, endpoint_physical_t, endpoint_flow_t
            )
            endpoint = endpoint_source + initial_velocity
            if midpoint_endpoint:
                midpoint = endpoint_source + 0.5 * initial_velocity
                half = torch.full_like(endpoint_flow_t, 0.5)
                midpoint_velocity, _ = self.vector_field(
                    midpoint, initial, x, endpoint_physical_t, half
                )
                endpoint = endpoint_source + midpoint_velocity
        if consistency_weight > 0.0 and endpoint_steps > 0 and consistency_steps > 0:
            # Train the learned vector field to be stable under the numerical
            # resolution used to integrate it at inference time.
            consistency_endpoint, _ = self.integrate(
                initial, x, endpoint_physical_t, steps=consistency_steps
            )
            consistency_loss = F.smooth_l1_loss(endpoint, consistency_endpoint.detach())
        else:
            consistency_loss = endpoint.new_zeros(())
        if wavelet_weight > 0.0:
            wavelet_loss = haar_detail_loss(endpoint, endpoint_target, wavelet_levels)
        else:
            wavelet_loss = endpoint.new_zeros(())
        endpoint_time_scale = 1.0 + time_weight * (
            endpoint_physical_t / physical_t.amax(dim=1, keepdim=True).clamp_min(1e-6)
        ) ** 2
        endpoint_point = ((endpoint - endpoint_target).square() * endpoint_weights).mean(dim=-1)
        endpoint_per_case = (
            endpoint_point * endpoint_time_scale[..., None]
        ).mean(dim=(1, 2))
        endpoint_loss = (endpoint_per_case * case_weight).mean()
        pressure_band = pressure_shock_band(endpoint_target)
        pressure_band_weight_map = pressure_band.float()[..., None]
        pressure_band_count = pressure_band_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
        pressure_band_abs = (
            (endpoint[..., 1:2] - endpoint_target[..., 1:2]).abs()
            * pressure_band_weight_map
        ).sum(dim=(1, 2, 3)) / pressure_band_count
        pressure_band_gradient = torch.diff(endpoint[..., 1], dim=2)
        target_band_gradient = torch.diff(endpoint_target[..., 1], dim=2)
        gradient_band = (
            pressure_band[..., :-1] | pressure_band[..., 1:]
        ).float()
        gradient_band_count = gradient_band.sum(dim=(1, 2)).clamp_min(1.0)
        pressure_band_gradient_loss = (
            (
                pressure_band_gradient - target_band_gradient
            ).abs()
            * gradient_band
        ).sum(dim=(1, 2)) / gradient_band_count
        pressure_band_loss = (
            pressure_band_abs + pressure_band_gradient_loss
        ) * case_weight
        pressure_band_loss = pressure_band_loss.mean()
        endpoint_profiles = endpoint.permute(0, 1, 3, 2).flatten(0, 1)
        target_profiles = endpoint_target.permute(0, 1, 3, 2).flatten(0, 1)
        profile_loss = endpoint.new_zeros(())
        for kernel in (1, 3, 7):
            if kernel == 1:
                endpoint_scale = endpoint_profiles
                target_scale = target_profiles
            else:
                endpoint_scale = F.avg_pool1d(
                    endpoint_profiles, kernel, stride=1, padding=kernel // 2
                )
                target_scale = F.avg_pool1d(
                    target_profiles, kernel, stride=1, padding=kernel // 2
                )
            profile_loss = profile_loss + F.smooth_l1_loss(
                torch.diff(endpoint_scale, dim=-1),
                torch.diff(target_scale, dim=-1),
            )
        profile_loss = profile_loss / 3.0
        if endpoint.shape[1] > 1:
            temporal_loss = F.smooth_l1_loss(
                torch.diff(endpoint, dim=1), torch.diff(endpoint_target, dim=1)
            )
        else:
            temporal_loss = endpoint.new_zeros(())
        lower = endpoint_target.amin(dim=2, keepdim=True)
        upper = endpoint_target.amax(dim=2, keepdim=True)
        overshoot_loss = (
            F.relu(endpoint - upper).square() + F.relu(lower - endpoint).square()
        ).mean()
        endpoint_pressure_gradient = torch.diff(endpoint[..., 1], dim=2).abs()
        target_pressure_gradient = torch.diff(endpoint_target[..., 1], dim=2).abs()
        endpoint_tv = endpoint_pressure_gradient.sum(dim=2)
        target_tv = target_pressure_gradient.sum(dim=2)
        active_profiles = target_tv > 1e-4
        endpoint_mass = endpoint_pressure_gradient / endpoint_tv[..., None].clamp_min(1e-6)
        target_mass = target_pressure_gradient / target_tv[..., None].clamp_min(1e-6)
        transport_per_profile = (
            endpoint_mass.cumsum(dim=2) - target_mass.cumsum(dim=2)
        ).abs().mean(dim=2)
        if active_profiles.any():
            transport_loss = transport_per_profile[active_profiles].mean()
            tv_loss = F.smooth_l1_loss(
                torch.log1p(endpoint_tv[active_profiles]),
                torch.log1p(target_tv[active_profiles]),
            )
        else:
            transport_loss = endpoint.new_zeros(())
            tv_loss = endpoint.new_zeros(())
        if teacher_target is None:
            retention_loss = endpoint.new_zeros(())
        else:
            retention_weights = (1.0 - endpoint_sensor[..., None]).clamp_min(0.1)
            retention_loss = (
                (endpoint - teacher_target).square() * retention_weights
            ).mean()
        total = (
            flow_loss
            + sensor_weight * sensor_loss
            + endpoint_weight * endpoint_loss
            + profile_weight * profile_loss
            + temporal_weight * temporal_loss
            + overshoot_weight * overshoot_loss
            + transport_weight * transport_loss
            + tv_weight * tv_loss
            + pressure_band_weight * pressure_band_loss
            + pressure_band_gradient_weight * pressure_band_gradient_loss.mean()
            + consistency_weight * consistency_loss
            + wavelet_weight * wavelet_loss
            + teacher_weight * retention_loss
        )
        return total, {
            "flow": flow_loss.detach(),
            "sensor": sensor_loss.detach(),
            "endpoint": endpoint_loss.detach(),
            "profile": profile_loss.detach(),
            "temporal": temporal_loss.detach(),
            "overshoot": overshoot_loss.detach(),
            "transport": transport_loss.detach(),
            "tv": tv_loss.detach(),
            "pressure_band": pressure_band_loss.detach(),
            "pressure_band_gradient": pressure_band_gradient_loss.mean().detach(),
            "consistency": consistency_loss.detach(),
            "wavelet": wavelet_loss.detach(),
            "retention": retention_loss.detach(),
        }

    def integrate(
        self,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        steps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, times = physical_t.shape
        state = initial[:, None].expand(-1, times, -1, -1).clone()
        shock_logits = None
        step_size = 1.0 / steps
        for step in range(steps):
            s0 = torch.full((batch, times), step * step_size, device=state.device)
            velocity, _ = self.vector_field(state, initial, x, physical_t, s0)
            midpoint = state + 0.5 * step_size * velocity
            smid = s0 + 0.5 * step_size
            mid_velocity, shock_logits = self.vector_field(midpoint, initial, x, physical_t, smid)
            state = state + step_size * mid_velocity
        return state, torch.sigmoid(shock_logits)

    @torch.no_grad()
    def predict(
        self,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        steps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.integrate(initial, x, physical_t, steps)


class ShockFrameWaveletFlowOperator(CharacteristicFlowOperator):
    """Flow matching with input-observable shock-frame and Haar context.

    The added branch starts at exactly zero, so a flow-matching checkpoint is a valid
    initialization and its original prediction is preserved before fine-tuning.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        layers: int = 4,
        heads: int = 8,
        slices: int = 24,
        wavelet_levels: int = 3,
    ):
        super().__init__(hidden_dim=hidden_dim, layers=layers, heads=heads, slices=slices)
        self.wavelet_levels = wavelet_levels
        context_channels = 2 * 3 * wavelet_levels + 2
        self.wavelet_stem = nn.Conv1d(context_channels, hidden_dim, 5, padding=2)
        nn.init.zeros_(self.wavelet_stem.weight)
        nn.init.zeros_(self.wavelet_stem.bias)

    def vector_field(
        self,
        state: torch.Tensor,
        initial: torch.Tensor,
        x: torch.Tensor,
        physical_t: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, times, nodes, _ = state.shape
        descriptor = differentiable_riemann_descriptor(initial, x)
        repeated_descriptor = descriptor[:, None].expand(-1, times, -1)
        condition_input = torch.cat(
            (repeated_descriptor, physical_t[..., None], flow_t[..., None]), dim=-1
        )
        condition = self.condition(condition_input).reshape(batch * times, -1)

        initial_seq = initial[:, None].expand(-1, times, -1, -1)
        x_seq = x[None, None, :, None].expand(batch, times, -1, -1)
        physical_seq = physical_t[..., None, None].expand(-1, -1, nodes, 1)
        flow_seq = flow_t[..., None, None].expand(-1, -1, nodes, 1)
        features = torch.cat((state, initial_seq, x_seq, physical_seq, flow_seq), dim=-1)
        hidden = self.stem(features.reshape(batch * times, nodes, -1).transpose(1, 2))

        state_details = haar_detail_pyramid(state, self.wavelet_levels)
        initial_details = haar_detail_pyramid(initial_seq, self.wavelet_levels)
        dx = (x[1] - x[0]).abs()
        similarity = (x[None, None] - descriptor[:, None, 0, None]) / (
            physical_t[..., None] + dx
        )
        frame_context = torch.stack(
            (torch.tanh(similarity / 8.0), 1.0 / (1.0 + similarity.abs())), dim=-1
        )
        multiscale_context = torch.cat(
            (*state_details, *initial_details, frame_context), dim=-1
        )
        hidden = hidden + self.wavelet_stem(
            multiscale_context.reshape(batch * times, nodes, -1).transpose(1, 2)
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        velocity = self.velocity_head(hidden).transpose(1, 2).reshape(
            batch, times, nodes, 3
        )
        shock_map = self.shock_head(hidden).squeeze(1).reshape(batch, times, nodes)
        return velocity, shock_map


def build_model(name: str, time_count: int, **kwargs) -> nn.Module:
    if name != "swfmo":
        raise ValueError("This release exposes only the SWFMO method.")
    return ShockFrameWaveletFlowOperator(**kwargs)
