"""
Adversarial robustness evaluation using torchattacks.

Evaluates:
  - Clean accuracy (baseline)
  - FGSM (fast, weak)
  - PGD-20 (strong iterative)
  - AutoAttack (strongest, ensemble of attacks)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchattacks


def _eval_on_attacked(
    model: nn.Module,
    attack,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Evaluate accuracy of model on adversarially attacked images."""
    model.eval()
    correct = total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)
        adv_images = attack(images, targets)  # requires grad internally
        with torch.no_grad():
            preds = model(adv_images).argmax(1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    return correct / total * 100


def evaluate_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 8 / 255,
    pgd_steps: int = 20,
    pgd_alpha: float = 2 / 255,
    max_batches: int | None = 10,
    run_autoattack: bool = False,
) -> dict[str, float]:
    """
    Run clean + FGSM + PGD-20 (+ optionally AutoAttack) evaluation.

    Args:
        model:           Model to evaluate
        loader:          Evaluation DataLoader
        device:          Device
        eps:             Perturbation budget (L∞), default 8/255
        pgd_steps:       PGD iterations, default 20
        pgd_alpha:       PGD step size, default 2/255
        max_batches:     Limit batches for speed (None = full dataset)
        run_autoattack:  Run AutoAttack (slow, ~10× slower than PGD-20)

    Returns:
        dict with keys: "clean", "fgsm", "pgd20" (and "autoattack" if run_autoattack)
    """
    model.eval()

    # Clean accuracy
    correct = total = 0
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(images).argmax(1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
    clean_acc = correct / total * 100

    # FGSM
    fgsm = torchattacks.FGSM(model, eps=eps)
    fgsm_acc = _eval_on_attacked(model, fgsm, loader, device, max_batches)

    # PGD-20
    pgd = torchattacks.PGD(model, eps=eps, alpha=pgd_alpha, steps=pgd_steps)
    pgd_acc = _eval_on_attacked(model, pgd, loader, device, max_batches)

    results = {
        "clean": clean_acc,
        "fgsm": fgsm_acc,
        "pgd20": pgd_acc,
    }

    # AutoAttack (optional — very slow)
    if run_autoattack:
        aa = torchattacks.AutoAttack(model, norm="Linf", eps=eps, version="standard")
        aa_acc = _eval_on_attacked(model, aa, loader, device, max_batches)
        results["autoattack"] = aa_acc

    return results
