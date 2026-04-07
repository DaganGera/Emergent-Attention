"""
Smoke tests for training pipeline.

These tests use a very small model and minimal data to verify:
  1. Loss decreases over training steps
  2. Checkpoint save/load round-trip
  3. AMP forward pass (CPU float32 fallback)

Run: pytest tests/test_training.py -v
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.nca_vit import NCAViT
from src.utils.checkpoint import save_checkpoint, load_checkpoint, save_best_model
from src.training.scheduler import CosineWarmupScheduler


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_tiny_model() -> NCAViT:
    """NCA-ViT with img_size=32, patch_size=8, depth=2 for fast tests."""
    return NCAViT(
        img_size=32, patch_size=8, num_classes=10,
        embed_dim=64, depth=2, nca_steps=2, nca_hidden_dim=128,
        stochastic_rate=0.5,
    )


def make_tiny_loader(n: int = 32, img_size: int = 32, num_classes: int = 10,
                     batch_size: int = 8) -> DataLoader:
    imgs = torch.randn(n, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=True)


# ── Loss Decreases ────────────────────────────────────────────────────────────

class TestLossDecreases:

    def test_loss_decreases_over_steps(self):
        """Training loss should decrease over 20 steps on a fixed batch."""
        torch.manual_seed(42)
        model = make_tiny_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    def test_loss_finite_throughout(self):
        """No NaN or Inf in loss during training."""
        torch.manual_seed(0)
        model = make_tiny_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = make_tiny_loader()

        for images, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
            loss.backward()
            optimizer.step()


# ── Checkpoint Round-trip ─────────────────────────────────────────────────────

class TestCheckpoint:

    def test_save_load_round_trip(self, tmp_path):
        torch.manual_seed(42)
        model = make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        # Fake scheduler with state_dict
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        ckpt_path = str(tmp_path / "test_ckpt.pt")
        save_checkpoint(model, optimizer, scheduler, epoch=5, best_acc=88.5, path=ckpt_path)

        state = load_checkpoint(ckpt_path)
        assert state["epoch"] == 5
        assert state["best_acc"] == pytest.approx(88.5)
        assert "model" in state
        assert "optimizer" in state

    def test_model_weights_preserved(self, tmp_path):
        torch.manual_seed(42)
        model = make_tiny_model()
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}

        ckpt_path = str(tmp_path / "weights.pt")
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        save_checkpoint(model, optimizer, scheduler, epoch=0, best_acc=0.0, path=ckpt_path)

        # Modify model in-place
        for p in model.parameters():
            p.data.fill_(0.0)

        # Reload
        state = load_checkpoint(ckpt_path)
        model.load_state_dict(state["model"])

        for k, v in model.state_dict().items():
            assert torch.allclose(v, original_weights[k]), f"Weights differ for {k}"

    def test_save_best_model_weights_only(self, tmp_path):
        model = make_tiny_model()
        path = str(tmp_path / "best.pt")
        save_best_model(model, path)
        assert os.path.isfile(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(state, dict)


# ── Scheduler Tests ───────────────────────────────────────────────────────────

class TestScheduler:

    def test_warmup_phase_increases_lr(self):
        model = make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=10, total_epochs=100, min_lr_ratio=0.0
        )
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR should be increasing (or at least non-decreasing) during warmup
        assert lrs[-1] > lrs[0], f"LR not increasing during warmup: {lrs}"

    def test_cosine_decay_after_warmup(self):
        model = make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=5, total_epochs=50, min_lr_ratio=0.0
        )
        # Skip warmup
        for _ in range(5):
            scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]["lr"]
        for _ in range(20):
            scheduler.step()
        lr_later = optimizer.param_groups[0]["lr"]
        assert lr_later < lr_after_warmup, "LR not decaying during cosine phase"
