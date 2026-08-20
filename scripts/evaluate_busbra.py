"""
BUS-BRA evaluation with imbalance-aware metrics (68/32 benign/malignant).

Mirrors scripts/evaluate_busi.py's metric choice: top-1 is a weak signal on
an imbalanced binary set (always-benign scores 68% here without learning
anything), so balanced accuracy (mean per-class recall, chance = 50%) and
macro-F1 are reported alongside it.

Usage:
    python scripts/evaluate_busbra.py \
        --checkpoint outputs/exp/busbra_hybrid/checkpoints/nca_vit_hybrid/busbra/seed42/best.pt \
        --model nca_vit_hybrid
"""

import os
import sys
import json
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT
from src.models.baselines import create_baseline
from src.data.busbra import build_busbra_datasets, CLASSES, NUM_CLASSES, DEFAULT_ROOT
from src.data.datasets import _build_busbra_val_transform
from src.utils.checkpoint import load_checkpoint

BUILDERS = {
    "nca_vit_tiny": lambda nc: NCAViT(num_classes=nc),
    "nca_vit_hybrid": lambda nc: HybridNCAViT(
        num_classes=nc, embed_dim=192, nca_depth=6, attn_depth=6, nca_steps=4,
        filter_names=["sobel_x", "sobel_y", "identity"], nca_hidden_dim=384,
        stochastic_rate=0.5, mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
        learnable_filters=True,
    ),
    "vit_tiny": lambda nc: create_baseline("vit_tiny", num_classes=nc),
    "deit_tiny": lambda nc: create_baseline("deit_tiny", num_classes=nc),
    "swin_tiny": lambda nc: create_baseline("swin_tiny", num_classes=nc),
}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        out = model(images.to(device))
        preds.append(out.argmax(1).cpu())
        targets.append(labels)
    return torch.cat(preds), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True, choices=list(BUILDERS))
    parser.add_argument("--busbra-root", default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BUILDERS[args.model](NUM_CLASSES).to(device)
    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state["model"] if "model" in state else state)

    _, val_ds = build_busbra_datasets(
        train_transform=_build_busbra_val_transform(224),  # unused, val-only run
        val_transform=_build_busbra_val_transform(224),
        root=args.busbra_root,
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    preds, targets = predict(model, loader, device)

    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for t, p in zip(targets, preds):
        confusion[t, p] += 1

    top1 = (preds == targets).float().mean().item() * 100
    recalls, precisions, f1s = [], [], []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c].item()
        recall = tp / max(confusion[c, :].sum().item(), 1)
        precision = tp / max(confusion[:, c].sum().item(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        recalls.append(recall); precisions.append(precision); f1s.append(f1)

    balanced_acc = sum(recalls) / NUM_CLASSES * 100
    macro_f1 = sum(f1s) / NUM_CLASSES * 100
    majority = confusion.sum(1).max().item() / confusion.sum().item() * 100

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"  top-1                {top1:6.2f}%   (majority-class baseline {majority:.2f}%)")
    print(f"  balanced accuracy    {balanced_acc:6.2f}%   (chance {100 / NUM_CLASSES:.2f}%)")
    print(f"  macro-F1             {macro_f1:6.2f}%")
    print(f"\n  {'class':12s}{'precision':>11s}{'recall':>9s}{'f1':>8s}{'n':>6s}")
    for c, name in enumerate(CLASSES):
        print(f"  {name:12s}{precisions[c] * 100:10.1f}%{recalls[c] * 100:8.1f}%"
              f"{f1s[c] * 100:7.1f}%{confusion[c].sum().item():6d}")
    print(f"\n  confusion (rows = true, cols = predicted):")
    print(f"  {'':12s}" + "".join(f"{n[:9]:>11s}" for n in CLASSES))
    for c, name in enumerate(CLASSES):
        print(f"  {name:12s}" + "".join(f"{confusion[c, j].item():11d}" for j in range(NUM_CLASSES)))

    results = {
        "model": args.model, "dataset": "busbra", "checkpoint": args.checkpoint,
        "top1": round(top1, 2), "balanced_accuracy": round(balanced_acc, 2),
        "macro_f1": round(macro_f1, 2), "majority_baseline": round(majority, 2),
        "per_class": {n: {"precision": round(precisions[i] * 100, 2),
                          "recall": round(recalls[i] * 100, 2),
                          "f1": round(f1s[i] * 100, 2),
                          "support": int(confusion[i].sum())}
                      for i, n in enumerate(CLASSES)},
        "confusion": confusion.tolist(),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.model}_busbra.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
