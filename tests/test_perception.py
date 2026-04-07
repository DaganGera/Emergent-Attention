"""
Unit tests for PerceptionModule.

Run: pytest tests/test_perception.py -v
"""

import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.perception import PerceptionModule, FILTER_REGISTRY


class TestPerceptionModuleShape:
    """Output shape is correct for various D and M combinations."""

    def test_default_filters_shape(self):
        # Default: 3 filters (sobel_x, sobel_y, identity)
        D, Hp, Wp, B = 192, 14, 14, 2
        module = PerceptionModule(dim=D)
        x = torch.randn(B, D, Hp, Wp)
        out = module(x)
        assert out.shape == (B, D * 3, Hp, Wp), f"Expected {(B, D*3, Hp, Wp)}, got {out.shape}"

    def test_single_filter_shape(self):
        D, Hp, Wp, B = 64, 7, 7, 4
        module = PerceptionModule(dim=D, filter_names=["identity"])
        x = torch.randn(B, D, Hp, Wp)
        out = module(x)
        assert out.shape == (B, D * 1, Hp, Wp)

    def test_four_filters_shape(self):
        D, Hp, Wp, B = 192, 14, 14, 1
        module = PerceptionModule(dim=D, filter_names=["sobel_x", "sobel_y", "laplacian", "identity"])
        x = torch.randn(B, D, Hp, Wp)
        out = module(x)
        assert out.shape == (B, D * 4, Hp, Wp)

    def test_spatial_dims_preserved(self):
        # padding=1 should preserve Hp and Wp
        D, Hp, Wp = 128, 28, 28
        module = PerceptionModule(dim=D)
        x = torch.randn(1, D, Hp, Wp)
        out = module(x)
        assert out.shape[2] == Hp
        assert out.shape[3] == Wp


class TestPerceptionModuleFilterInit:
    """Weight initialisation matches the expected fixed kernels."""

    def test_identity_filter_values(self):
        # With identity-only filter, the first D output channels should equal input
        D = 8
        module = PerceptionModule(dim=D, filter_names=["identity"], learnable=False)
        x = torch.randn(1, D, 5, 5)
        out = module(x)
        # Identity convolution should reproduce input (up to boundary effects, but center is exact)
        assert torch.allclose(out[:, :, 2, 2], x[:, :, 2, 2], atol=1e-5)

    def test_sobel_x_detects_horizontal_edges(self):
        # Vertical stripe pattern: strong response to Sobel_x
        D = 4
        module = PerceptionModule(dim=D, filter_names=["sobel_x"], learnable=False)
        # Vertical edge at center column
        x = torch.zeros(1, D, 7, 7)
        x[:, :, :, 3:] = 1.0
        out = module(x)
        # Max response should be at the edge column (col 2–3)
        assert out.abs().max() > 0.1

    def test_filter_values_match_registry(self):
        D = 2
        module = PerceptionModule(dim=D, filter_names=["sobel_x", "sobel_y"], learnable=False)
        expected_x = FILTER_REGISTRY["sobel_x"].squeeze()  # (3, 3)
        # First filter for channel 0 should match sobel_x
        actual = module.weight[0, 0]  # (3, 3)
        assert torch.allclose(actual, expected_x, atol=1e-6)

    def test_weight_tiled_across_channels(self):
        D = 4
        module = PerceptionModule(dim=D, filter_names=["sobel_x"], learnable=False)
        # All D slices of the weight should be identical (tiled)
        for i in range(D):
            assert torch.allclose(module.weight[i, 0], module.weight[0, 0], atol=1e-6)


class TestPerceptionModuleGradients:
    """Gradient flow depends on learnable flag."""

    def test_no_gradient_when_frozen(self):
        module = PerceptionModule(dim=32, learnable=False)
        x = torch.randn(1, 32, 7, 7, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        # Weight is a buffer — no grad
        assert not hasattr(module.weight, "grad") or module.weight.grad is None

    def test_gradient_flows_when_learnable(self):
        module = PerceptionModule(dim=32, learnable=True)
        x = torch.randn(1, 32, 7, 7)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert module.weight.grad is not None
        assert module.weight.grad.shape == module.weight.shape
        assert module.weight.grad.abs().sum() > 0

    def test_input_gradient_flows_always(self):
        # Even with frozen weights, gradients flow to x
        module = PerceptionModule(dim=32, learnable=False)
        x = torch.randn(1, 32, 7, 7, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestPerceptionModuleProperties:
    """Module properties and registration."""

    def test_weight_is_buffer_when_frozen(self):
        module = PerceptionModule(dim=16, learnable=False)
        assert "weight" in dict(module.named_buffers())
        assert "weight" not in dict(module.named_parameters())

    def test_weight_is_parameter_when_learnable(self):
        module = PerceptionModule(dim=16, learnable=True)
        assert "weight" in dict(module.named_parameters())
        assert "weight" not in dict(module.named_buffers())

    def test_num_filters_attribute(self):
        module = PerceptionModule(dim=32, filter_names=["sobel_x", "sobel_y"])
        assert module.num_filters == 2

    def test_extra_repr(self):
        module = PerceptionModule(dim=192)
        r = module.extra_repr()
        assert "192" in r
        assert "sobel_x" in r
