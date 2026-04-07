"""
Weights & Biases helper utilities.
"""

import wandb
from omegaconf import OmegaConf


def init_wandb(cfg) -> None:
    """
    Initialise a W&B run from a Hydra config.

    Args:
        cfg: Hydra DictConfig with keys: model, data, training
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = (
        f"{cfg.model.name}-{cfg.data.dataset}"
        f"-K{cfg.model.get('nca_steps', 'N/A')}"
        f"-seed{cfg.training.get('seed', 42)}"
    )
    wandb.init(
        project="emergent-attention",
        name=run_name,
        config=config_dict,
        tags=[cfg.model.name, cfg.data.dataset],
    )


def log_metrics(metrics: dict, step: int) -> None:
    """Log a dict of metrics to W&B at a given step."""
    wandb.log(metrics, step=step)


def finish_run() -> None:
    wandb.finish()
