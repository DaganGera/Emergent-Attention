"""
Classification metrics: Top-1, Top-5, per-class accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """
    Compute Top-1 and Top-5 accuracy over a DataLoader.

    Returns:
        {"top1": float, "top5": float, "n_samples": int}
    """
    from torch.cuda.amp import autocast

    model.eval()
    correct1 = correct5 = total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)

        correct1 += (outputs.argmax(1) == targets).sum().item()
        top5 = outputs.topk(min(5, outputs.size(1)), dim=1).indices
        correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += images.size(0)

    return {
        "top1": correct1 / total * 100,
        "top5": correct5 / total * 100,
        "n_samples": total,
    }


@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 100,
) -> list[float]:
    """
    Compute per-class Top-1 accuracy.

    Returns:
        List of length num_classes, each entry is accuracy % for that class.
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images).argmax(1)
        for pred, target in zip(preds, targets):
            class_total[target.item()] += 1
            if pred == target:
                class_correct[target.item()] += 1

    return [
        (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]
