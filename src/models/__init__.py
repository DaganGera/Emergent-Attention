"""
Model factory — builds the right model from a Hydra config.

Usage:
    from src.models import build_model
    model = build_model(cfg.model)
"""

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT
from src.models.baselines import create_baseline


def build_model(cfg) -> "torch.nn.Module":
    """
    Build model from config node.

    Recognised cfg.name values:
        "nca_vit_tiny"    → NCAViT with tiny defaults
        "nca_vit_hybrid"  → HybridNCAViT
        anything else     → timm baseline via create_baseline()
    """
    name = cfg.name

    if name == "nca_vit_tiny":
        return NCAViT(
            num_classes=cfg.get("num_classes", 100),
            embed_dim=cfg.get("embed_dim", 192),
            depth=cfg.get("depth", 12),
            nca_steps=cfg.get("nca_steps", 4),
            filter_names=list(cfg.get("perception_filters", ["sobel_x", "sobel_y", "identity"])),
            nca_hidden_dim=cfg.get("update_hidden_dim", 384),
            stochastic_rate=cfg.get("stochastic_rate", 0.5),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            cls_strategy=cfg.get("cls_strategy", "exclude"),
        )

    elif name == "nca_vit_hybrid":
        return HybridNCAViT(
            num_classes=cfg.get("num_classes", 100),
            embed_dim=cfg.get("embed_dim", 192),
            nca_depth=cfg.get("nca_depth", 6),
            attn_depth=cfg.get("attn_depth", 6),
            nca_steps=cfg.get("nca_steps", 4),
            filter_names=list(cfg.get("perception_filters", ["sobel_x", "sobel_y", "identity"])),
            nca_hidden_dim=cfg.get("update_hidden_dim", 384),
            stochastic_rate=cfg.get("stochastic_rate", 0.5),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
        )

    else:
        return create_baseline(
            name=name,
            num_classes=cfg.get("num_classes", 100),
            pretrained=cfg.get("pretrained", False),
        )
