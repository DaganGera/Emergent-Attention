"""
Augmentation helpers: Mixup / CutMix via timm.
"""

from typing import Optional
from timm.data.mixup import Mixup


def build_mixup(
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
    num_classes: int,
    label_smoothing: float,
) -> Optional[Mixup]:
    """
    Return a timm Mixup callable, or None if both alphas are 0.

    Args:
        mixup_alpha:    Alpha for Mixup Beta distribution (0 to disable).
        cutmix_alpha:   Alpha for CutMix Beta distribution (0 to disable).
        mixup_prob:     Probability of applying Mixup per batch.
        cutmix_prob:    Probability of applying CutMix per batch.
        num_classes:    Number of output classes (for one-hot targets).
        label_smoothing: Label smoothing factor applied inside Mixup.

    Returns:
        Mixup instance, or None when both augmentations are disabled.
    """
    if mixup_alpha == 0.0 and cutmix_alpha == 0.0:
        return None

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mixup_prob + cutmix_prob,   # total prob of any aug
        switch_prob=cutmix_prob / (mixup_prob + cutmix_prob) if (mixup_prob + cutmix_prob) > 0 else 0.5,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
