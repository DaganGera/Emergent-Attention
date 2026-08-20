"""
One-time script: extracts a small, fixed gallery of sample images into
webapp/backend/static/samples/, so a visitor to the demo website can test
the system without needing to supply their own files.

Uses the same dataset files and (for BUSI) the same loader already used for
training/evaluation elsewhere in this repo -- these are held-out validation
images, not hand-picked to look good.

Usage:
    python scripts/build_web_samples.py
"""

import os
import sys
import json
import pickle
import random

from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.data.busbra import build_busbra_datasets, CLASSES as BUSBRA_CLASSES

OUT_DIR = os.path.join(_REPO_ROOT, "webapp", "backend", "static", "samples")
RNG = random.Random(42)


def build_cifar100_samples(n_total: int = 12) -> None:
    meta_path = os.path.join(_REPO_ROOT, "data", "cifar-100-python", "meta")
    test_path = os.path.join(_REPO_ROOT, "data", "cifar-100-python", "test")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    classes = meta["fine_label_names"]
    with open(test_path, "rb") as f:
        test = pickle.load(f, encoding="latin1")
    data = test["data"]  # (N, 3072) uint8, R-plane then G-plane then B-plane, 32x32
    labels = test["fine_labels"]

    chosen_classes = RNG.sample(range(len(classes)), n_total)
    out_dir = os.path.join(OUT_DIR, "cifar100")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for cls_idx in chosen_classes:
        idxs = [i for i, l in enumerate(labels) if l == cls_idx]
        i = RNG.choice(idxs)
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        pil_img = Image.fromarray(img, mode="RGB")
        fname = f"{classes[cls_idx]}.png"
        pil_img.save(os.path.join(out_dir, fname))
        manifest.append({"id": classes[cls_idx], "file": fname, "true_label": classes[cls_idx]})

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} CIFAR-100 samples to {out_dir}")


def build_busbra_samples(n_per_class: int = 4) -> None:
    _, val_ds = build_busbra_datasets(
        train_transform=None, val_transform=None,
        root=os.path.join(_REPO_ROOT, "data", "busbra", "BUSBRA", "BUSBRA"),
    )
    by_class: dict[int, list] = {i: [] for i in range(len(BUSBRA_CLASSES))}
    for i in range(len(val_ds)):
        img, label = val_ds[i]
        by_class[label].append(img)

    out_dir = os.path.join(OUT_DIR, "busbra")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for cls_idx, cls_name in enumerate(BUSBRA_CLASSES):
        pool = by_class[cls_idx]
        chosen = RNG.sample(range(len(pool)), min(n_per_class, len(pool)))
        for k, idx in enumerate(chosen):
            fname = f"{cls_name}_{k + 1}.png"
            pool[idx].save(os.path.join(out_dir, fname))
            manifest.append({"id": f"{cls_name}_{k + 1}", "file": fname, "true_label": cls_name})

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} BUS-BRA samples to {out_dir}")


if __name__ == "__main__":
    build_cifar100_samples()
    build_busbra_samples()
