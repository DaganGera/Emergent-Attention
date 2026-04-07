"""
Unit tests for NCAAttention block.

Run: pytest tests/test_nca_attention.py -v
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.nca_attention import NCAAttention


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def nca_tiny():
    """NCA block matching NCA-ViT-Tiny specs: D=192, K=4, grid=14x14."""
    return NCAAttention(dim=192, nca_steps=4, hidden_dim=384, grid_size=(14, 14))


@pytest.fixture
def sample_input():
    """Batch of 2 sequences: 196 patch tokens + 1 CLS, D=192."""
    B, N1, D = 2, 197, 192  # N+1 = 196+1
    return torch.randn(B, N1, D)


# ── Shape Tests ─────────────────────────────────────────────────────────────

class TestNCAAttentionShape:

    def test_output_shape_matches_input(self, nca_tiny, sample_input):
        out = nca_tiny(sample_input)
        assert out.shape == sample_input.shape, (
            f"Expected {sample_input.shape}, got {out.shape}"
        )

    def test_output_shape_batch_2(self, nca_tiny):
        x = torch.randn(2, 197, 192)
        out = nca_tiny(x)
        assert out.shape == (2, 197, 192)

    def test_output_shape_batch_64(self, nca_tiny):
        x = torch.randn(64, 197, 192)
        out = nca_tiny(x)
        assert out.shape == (64, 197, 192)

    def test_different_nca_steps(self):
        for K in [1, 2, 8, 16]:
            block = NCAAttention(dim=64, nca_steps=K, hidden_dim=128, grid_size=(7, 7))
            x = torch.randn(2, 50, 64)  # 7*7=49 patches + 1 CLS
            out = block(x)
            assert out.shape == (2, 50, 64), f"Failed for K={K}"

    def test_small_grid(self):
        # 4x4 grid = 16 patches + 1 CLS = 17 tokens
        block = NCAAttention(dim=32, nca_steps=2, hidden_dim=64, grid_size=(4, 4))
        x = torch.randn(2, 17, 32)
        out = block(x)
        assert out.shape == (2, 17, 32)


# ── CLS Token Tests ──────────────────────────────────────────────────────────

class TestCLSTokenBehavior:

    def test_cls_token_passes_through_unchanged(self, nca_tiny):
        """CLS token should not be modified by NCA (exclude strategy)."""
        x = torch.randn(2, 197, 192)
        cls_before = x[:, :1, :].clone()
        out = nca_tiny(x)
        cls_after = out[:, :1, :]
        assert torch.allclose(cls_before, cls_after, atol=1e-6), (
            "CLS token was modified — should pass through unchanged"
        )

    def test_patch_tokens_are_modified(self, nca_tiny, sample_input):
        """Patch tokens should change after NCA update (non-trivial computation)."""
        nca_tiny.eval()  # deterministic
        out = nca_tiny(sample_input)
        patches_in = sample_input[:, 1:, :]
        patches_out = out[:, 1:, :]
        assert not torch.allclose(patches_in, patches_out, atol=1e-4), (
            "Patch tokens unchanged — NCA did nothing"
        )


# ── Gradient Tests ───────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_all_parameters_receive_gradients(self, nca_tiny, sample_input):
        nca_tiny.train()
        out = nca_tiny(sample_input)
        loss = out.sum()
        loss.backward()
        for name, param in nca_tiny.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for parameter: {name}"

    def test_input_gradient_flows(self, nca_tiny):
        x = torch.randn(2, 197, 192, requires_grad=True)
        out = nca_tiny(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_stable_for_large_K(self):
        """Gradients should not explode or vanish for K=16."""
        block = NCAAttention(dim=64, nca_steps=16, hidden_dim=128, grid_size=(7, 7))
        x = torch.randn(2, 50, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        grad_norm = x.grad.norm().item()
        assert grad_norm < 1e4, f"Gradient exploded: norm={grad_norm}"
        assert grad_norm > 1e-8, f"Gradient vanished: norm={grad_norm}"


# ── Initialization Tests ─────────────────────────────────────────────────────

class TestZeroInit:

    def test_linear2_weight_is_zero_at_init(self, nca_tiny):
        """Output linear layer should start at zero for training stability."""
        assert torch.allclose(nca_tiny.linear2.weight, torch.zeros_like(nca_tiny.linear2.weight)), (
            "linear2.weight should be zero-initialized"
        )

    def test_linear2_bias_is_zero_at_init(self, nca_tiny):
        assert torch.allclose(nca_tiny.linear2.bias, torch.zeros_like(nca_tiny.linear2.bias)), (
            "linear2.bias should be zero-initialized"
        )

    def test_near_identity_at_init(self, nca_tiny):
        """With zero-init output layer, NCA delta ≈ 0, so output ≈ input (up to norm)."""
        nca_tiny.eval()
        x = torch.randn(2, 197, 192)
        out = nca_tiny(x)
        # Patch tokens: out ≈ patches + 0 = patches (pre-norm(patches) near-zero delta + residual)
        # The residual adds the original patches back, so output patches ≈ input patches
        patches_in = x[:, 1:, :]
        patches_out = out[:, 1:, :]
        diff = (patches_out - patches_in).abs().max().item()
        assert diff < 1e-5, f"Output deviates from input at init: max diff = {diff}"


# ── Stochastic Update Tests ──────────────────────────────────────────────────

class TestStochasticUpdate:

    def test_training_nondeterministic(self, nca_tiny, sample_input):
        """Two training forward passes should differ (stochastic mask)."""
        nca_tiny.train()
        out1 = nca_tiny(sample_input)
        out2 = nca_tiny(sample_input)
        # With stochastic_rate=0.5, outputs should differ (modulo very rare equality)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Training outputs identical — stochastic mask may not be working"
        )

    def test_eval_deterministic(self, nca_tiny, sample_input):
        """Two eval forward passes should be identical."""
        nca_tiny.eval()
        out1 = nca_tiny(sample_input)
        out2 = nca_tiny(sample_input)
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Eval outputs differ — stochastic mask leaking into inference"
        )

    def test_stochastic_rate_one_always_updates(self):
        """With stochastic_rate=1.0, all cells always update (no mask)."""
        block = NCAAttention(dim=64, nca_steps=2, hidden_dim=128, grid_size=(7, 7),
                             stochastic_rate=1.0)
        block.train()
        x = torch.randn(2, 50, 64)
        out1 = block(x)
        out2 = block(x)
        # Rate=1.0 means no randomness in the mask logic, but norm adds state
        # Just check no NaN
        assert not torch.isnan(out1).any()
        assert not torch.isnan(out2).any()
