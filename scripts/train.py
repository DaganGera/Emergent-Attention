"""
Training entry point.

Launch examples:
    python scripts/train.py model=nca_vit_tiny data=cifar100
    python scripts/train.py model=nca_vit_tiny data=cifar100 training.training.seed=123
    python scripts/train.py model=nca_vit_hybrid data=cifar100 model.nca_steps=8
"""

import os
import sys
import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Make src importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data.datasets import build_loaders
from src.training.trainer import Trainer
from src.utils.wandb_utils import init_wandb, finish_run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Seed
    seed = cfg.training.training.get("seed", 42)
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # Model
    model = build_model(cfg.model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Data
    train_loader, val_loader = build_loaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # W&B (optional — set WANDB_MODE=disabled to skip)
    try:
        init_wandb(cfg)
    except Exception as e:
        print(f"W&B init skipped: {e}")

    # Output dir
    output_dir = os.path.join("checkpoints", cfg.model.name, cfg.data.dataset, f"seed{seed}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )

    # Resume if checkpoint exists
    resume_path = os.path.join(output_dir, "latest.pt")
    if os.path.isfile(resume_path):
        trainer.resume(resume_path)

    # Train
    trainer.fit()
    finish_run()


if __name__ == "__main__":
    main()
