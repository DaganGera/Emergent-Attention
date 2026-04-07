"""
Optimizer factory.
"""

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with optional layer-wise LR decay (not used by default).

    Args:
        model: The model to optimize
        cfg:   Hydra config node with optimizer settings

    Returns:
        Configured optimizer
    """
    opt_cfg = cfg.training.optimizer
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
    )
