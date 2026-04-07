"""
Baseline model wrappers.

Thin wrappers around timm models to ensure identical API as NCAViT.
All baselines use the same num_classes and optional pretrained weights.
"""

import timm
import torch.nn as nn


TIMM_MODEL_MAP = {
    "vit_tiny": "vit_tiny_patch16_224",
    "deit_tiny": "deit_tiny_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    # Fully-qualified timm names also accepted directly
}


def create_baseline(name: str, num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    Create a timm baseline model.

    Args:
        name:        Short name (e.g. "vit_tiny") or full timm name
        num_classes: Number of output classes
        pretrained:  Load ImageNet-1K pretrained weights

    Returns:
        nn.Module ready for training/evaluation
    """
    timm_name = TIMM_MODEL_MAP.get(name, name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model
