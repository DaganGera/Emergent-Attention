"""
Integration shape tests for full NCA-ViT and Hybrid NCA-ViT.

Run: pytest tests/test_shapes.py -v
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT


# ── NCA-ViT Shape Tests ──────────────────────────────────────────────────────

class TestNCAViTShape:

    @pytest.fixture
    def model(self):
        return NCAViT(
            img_size=224, patch_size=16, num_classes=100,
            embed_dim=192, depth=12, nca_steps=4,
        )

    def test_full_forward_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 100), f"Expected (2, 100), got {logits.shape}"

    def test_batch_size_1(self, model):
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 100)

    def test_batch_size_8(self, model):
        x = torch.randn(8, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (8, 100)

    def test_forward_features_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        features = model.forward_features(x)
        assert features.shape == (2, 192), f"Expected (2, 192), got {features.shape}"

    def test_no_nan_in_output(self, model):
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert not torch.isnan(logits).any(), "NaN in logits"

    def test_param_count_reasonable(self, model):
        counts = model.count_parameters()
        # NCA-ViT-Tiny should be roughly 5–10M parameters
        assert 3_000_000 < counts["total"] < 15_000_000, (
            f"Unexpected param count: {counts['total']:,}"
        )


class TestNCAViTSmall:
    """Lightweight tests for faster CI (smaller model)."""

    @pytest.fixture
    def small_model(self):
        return NCAViT(
            img_size=32, patch_size=8, num_classes=10,
            embed_dim=64, depth=2, nca_steps=2, nca_hidden_dim=128,
        )

    def test_small_forward(self, small_model):
        # 32/8 = 4x4 grid = 16 patches + 1 CLS = 17 tokens
        x = torch.randn(4, 3, 32, 32)
        logits = small_model(x)
        assert logits.shape == (4, 10)

    def test_gradient_flows_through_full_model(self, small_model):
        x = torch.randn(2, 3, 32, 32)
        logits = small_model(x)
        loss = logits.sum()
        loss.backward()
        for name, p in small_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"

    def test_overfit_on_tiny_batch(self, small_model):
        """Model should be able to overfit 4 examples (sanity check)."""
        import torch.optim as optim
        torch.manual_seed(42)
        optimizer = optim.Adam(small_model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])

        small_model.train()
        initial_loss = None
        for step in range(200):
            optimizer.zero_grad()
            loss = criterion(small_model(x), y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, (
            f"Model failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# ── Hybrid NCA-ViT Shape Tests ───────────────────────────────────────────────

class TestHybridNCAViTShape:

    @pytest.fixture
    def hybrid_model(self):
        return HybridNCAViT(
            img_size=224, patch_size=16, num_classes=100,
            embed_dim=192, nca_depth=6, attn_depth=6, nca_steps=4,
        )

    def test_full_forward_shape(self, hybrid_model):
        x = torch.randn(2, 3, 224, 224)
        logits = hybrid_model(x)
        assert logits.shape == (2, 100)

    def test_nca_only_split(self):
        # All 12 blocks NCA, 0 self-attention
        model = HybridNCAViT(
            img_size=224, patch_size=16, num_classes=100,
            embed_dim=64, nca_depth=12, attn_depth=0, nca_steps=2, nca_hidden_dim=128,
        )
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 100)

    def test_attn_only_split(self):
        # 0 NCA blocks, 12 self-attention (pure ViT)
        model = HybridNCAViT(
            img_size=224, patch_size=16, num_classes=100,
            embed_dim=64, nca_depth=0, attn_depth=12, nca_steps=4, nca_hidden_dim=128,
        )
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 100)
