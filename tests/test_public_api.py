"""Small CPU-only checks for the public shock-aware model API."""

import torch

from pdebench_1d import (
    MetricAlignedShockResidualFlow,
    ShockFrameWaveletFlowOperator,
    differentiable_riemann_descriptor,
)


def test_descriptor_is_input_only_and_finite():
    x = torch.linspace(-1.0, 1.0, 32)
    initial = torch.zeros(2, 32, 3)
    initial[:, 16:, 0] = 1.0
    initial[:, 16:, 1] = 2.0
    initial[:, 16:, 2] = 0.5
    descriptor = differentiable_riemann_descriptor(initial, x)
    assert descriptor.shape == (2, 7)
    assert torch.isfinite(descriptor).all()
    assert torch.allclose(descriptor[:, 0], torch.zeros(2), atol=0.08)


def test_base_and_zero_initialized_corrector_shapes():
    torch.manual_seed(0)
    batch, nodes, times = 2, 32, 3
    x = torch.linspace(-1.0, 1.0, nodes)
    initial = torch.randn(batch, nodes, 3)
    physical_t = torch.linspace(0.01, 0.2, times).repeat(batch, 1)
    base = ShockFrameWaveletFlowOperator(
        hidden_dim=16, layers=1, heads=4, slices=4, wavelet_levels=2
    )
    prediction, shock = base.predict(initial, x, physical_t, steps=1)
    assert prediction.shape == (batch, times, nodes, 3)
    assert shock.shape == (batch, times, nodes)

    corrector = MetricAlignedShockResidualFlow(
        hidden_dim=16, layers=1, heads=4, slices=4, wavelet_levels=2
    )
    refined = corrector.refine(base=prediction, shock_map=shock, initial=initial,
                               x=x, physical_t=physical_t, steps=1)
    assert refined.shape == prediction.shape
    # The residual velocity head is zero initialized, so the initial corrector
    # is exactly an identity refinement.
    assert torch.equal(refined, prediction)
