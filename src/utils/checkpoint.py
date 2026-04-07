"""
Checkpoint utilities: save and load model + optimizer + scheduler state.
"""

import os
import torch
import torch.nn as nn
from typing import Any


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_acc: float,
    path: str,
    ema_model: nn.Module | None = None,
) -> None:
    """Save full training state to a .pt file."""
    state = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str) -> dict:
    """Load checkpoint dict. Returns the raw state dict."""
    return torch.load(path, map_location="cpu", weights_only=False)


def save_best_model(model: nn.Module, path: str) -> None:
    """Save model weights only (for inference / evaluation)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
