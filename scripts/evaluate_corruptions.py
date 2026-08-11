"""
CIFAR-100-C corruption robustness evaluation.

Pure eval on existing checkpoints -- no training. Tests the "performs well
on noisy/degraded input" claim directly: 19 corruption types (Hendrycks et al.,
ICLR 2019) x 5 severities, applied to the CIFAR-100 test set.

Dataset: https://zenodo.org/records/3555552 (CIFAR-100-C.tar, ~2.9GB).
Expected layout after extraction: data/cifar-100-c/CIFAR-100-C/{corruption}.npy
(each (50000,32,32,3) uint8 -- 5 contiguous 10000-image blocks, one per
severity 1..5, in that order) plus a shared labels.npy.

Full grid (19 corruptions x 5 severities x 10000 images x 5 models) is
~475k forward passes per model and not necessary to see the effect; default
here is severities {1,3,5} x 1000 images/severity, resumable and written
incrementally so a Ctrl-C doesn't lose finished (model, corruption) pairs.
Pass --full for the exhaustive grid.

Usage:
    python scripts/evaluate_corruptions.py
    python scripts/evaluate_corruptions.py --full
    python scripts/evaluate_corruptions.py --models deit_tiny nca_vit_hybrid
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT
from src.models.baselines import create_baseline
from src.utils.checkpoint import load_checkpoint


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

CKPT_ROOT = "checkpoints"

MODEL_SPECS = {
    "deit_tiny": dict(
        ckpt=f"{CKPT_ROOT}/deit_tiny_patch16_224/cifar100/seed42/best.pt",
        build=lambda nc: create_baseline("deit_tiny", num_classes=nc),
    ),
    "vit_tiny": dict(
        ckpt=f"{CKPT_ROOT}/vit_tiny_patch16_224/cifar100/seed42/best.pt",
        build=lambda nc: create_baseline("vit_tiny", num_classes=nc),
    ),
    "swin_tiny": dict(
        ckpt=f"{CKPT_ROOT}/swin_tiny_patch4_window7_224/cifar100/seed42/best.pt",
        build=lambda nc: create_baseline("swin_tiny", num_classes=nc),
    ),
    "nca_vit_tiny": dict(
        ckpt=f"{CKPT_ROOT}/nca_vit_tiny/cifar100/seed42/best.pt",
        build=lambda nc: NCAViT(num_classes=nc),
    ),
    "nca_vit_hybrid": dict(
        ckpt=f"{CKPT_ROOT}/nca_vit_hybrid/cifar100/seed42/best.pt",
        build=lambda nc: HybridNCAViT(
            num_classes=nc, embed_dim=192, nca_depth=6, attn_depth=6, nca_steps=4,
            filter_names=["sobel_x", "sobel_y", "identity"], nca_hidden_dim=384,
            stochastic_rate=0.5, mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
            learnable_filters=True,
        ),
    ),
}


class CIFAR100CSlice(Dataset):
    """One (corruption, severity) slice of CIFAR-100-C."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])


def build_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])


def load_model(spec: dict, num_classes: int, device: torch.device) -> nn.Module:
    model = spec["build"](num_classes).to(device)
    state = load_checkpoint(spec["ckpt"])
    state = state["model"] if "model" in state else state
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def eval_slice(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            preds = model(images).argmax(1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
    return correct / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cifar-100-c/CIFAR-100-C")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()),
                         choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS, choices=CORRUPTIONS)
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5], choices=range(1, 6))
    parser.add_argument("--max-per-severity", type=int, default=1000,
                         help="Images evaluated per (corruption, severity); each block has 10000.")
    parser.add_argument("--full", action="store_true",
                         help="Override to the exhaustive grid: all 5 severities, all 10000 images.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/cifar100c_robustness.json")
    args = parser.parse_args()

    if args.full:
        args.severities = [1, 2, 3, 4, 5]
        args.max_per_severity = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100
    transform = build_transform(224)

    labels_path = os.path.join(args.data_dir, "labels.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} not found. Download+extract CIFAR-100-C first:\n"
            f"  https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
        )
    all_labels = np.load(labels_path)  # (50000,) -- 5 blocks of 10000, shared by every corruption

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming: {sum(len(v) for v in results.values())} (model,corruption,severity) cells already done.")

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        results.setdefault(model_name, {})
        print(f"\n=== {model_name} ===")
        model = load_model(spec, num_classes, device)

        for corruption in args.corruptions:
            npy_path = os.path.join(args.data_dir, f"{corruption}.npy")
            if not os.path.exists(npy_path):
                print(f"  [skip] {corruption}: {npy_path} not found")
                continue
            images = np.load(npy_path, mmap_mode="r")  # (50000, 32, 32, 3) uint8

            for sev in args.severities:
                key = f"{corruption}_sev{sev}"
                if key in results[model_name]:
                    continue
                lo, hi = (sev - 1) * 10000, (sev - 1) * 10000 + args.max_per_severity
                sev_images = np.array(images[lo:hi])
                sev_labels = all_labels[lo:hi]

                ds = CIFAR100CSlice(sev_images, sev_labels, transform)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0, pin_memory=(device.type == "cuda"))
                acc = eval_slice(model, loader, device)
                results[model_name][key] = round(acc, 2)
                print(f"  {corruption:20s} sev{sev}: {acc:5.2f}%  (n={hi - lo})")

                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary: mean accuracy across all evaluated (corruption, severity) cells per model
    print("\n=== Summary: mean corrupted accuracy per model ===")
    for model_name, cells in results.items():
        if cells:
            mean_acc = sum(cells.values()) / len(cells)
            print(f"  {model_name:20s}: {mean_acc:.2f}%  ({len(cells)} cells)")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
