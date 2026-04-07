"""
LR Scheduler factory: cosine annealing with linear warmup.
"""

import torch
import torch.optim as optim
import math


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Cosine annealing LR schedule with linear warmup.

    During warmup (epochs 0..warmup_epochs-1): LR increases linearly from 0 to base_lr.
    After warmup: LR follows a cosine decay from base_lr to min_lr.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → 1
            return (epoch + 1) / max(self.warmup_epochs, 1)
        # Cosine decay: 1 → min_lr_ratio
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


def build_scheduler(optimizer: optim.Optimizer, cfg) -> CosineWarmupScheduler:
    """Build the cosine+warmup scheduler from config."""
    train_cfg = cfg.training
    sched_cfg = cfg.training.scheduler
    min_lr_ratio = sched_cfg.min_lr / train_cfg.optimizer.lr
    return CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=sched_cfg.warmup_epochs,
        total_epochs=sched_cfg.epochs,
        min_lr_ratio=min_lr_ratio,
    )
