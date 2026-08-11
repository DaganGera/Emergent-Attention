"""
BUSI dataset loader — 780 breast ultrasound images, 3 classes.

Source: Al-Dhabyani et al. (2020), "Dataset of breast ultrasound images",
Data in Brief. Obtained via Kaggle (aryashah2k/breast-ultrasound-images-dataset,
CC0), which unpacks to Dataset_BUSI_with_GT/{benign,malignant,normal}/.

Chosen because speckle is the defining artifact of ultrasound imaging, and
speckle noise is where the hybrid's measured advantage is largest on
CIFAR-100-C (+9.6pp). This is the same noise model, not an analogy.

Three properties of this dataset drive the code below:

1. Every image ships alongside its segmentation mask in the SAME directory
   ("benign (1).png" and "benign (1)_mask.png"). Masks must be excluded, or
   the training set doubles in size with binary blobs labelled as ultrasound.
   Some lesions have several masks (_mask_1, _mask_2), so benign holds 454
   masks for 437 images -- filtering on a "_mask" substring, not on a count.

2. Classes are imbalanced: 437 benign / 210 malignant / 133 normal. Predicting
   "benign" for everything scores 56% top-1, so top-1 alone is close to
   meaningless here. Report balanced accuracy / macro-F1 alongside it.

3. 780 images come from 600 patients and the public release carries no patient
   IDs, so a patient-level split is not constructible. Images of the same
   patient can therefore land on both sides of the split. This inflates
   absolute numbers for every architecture equally, so the comparison between
   them stays valid while the absolute figures should not be read as clean
   generalisation. State this in any write-up.
"""

import os

import torch
from torch.utils.data import Dataset
from PIL import Image

SEED = 42
# 0.2 rather than PlantVillage's 0.15: the "normal" class has only 133 images,
# and 15% of that is 19 validation images -- too few for a stable estimate.
VAL_FRACTION = 0.2
NUM_CLASSES = 3

CLASSES = ["benign", "malignant", "normal"]
DEFAULT_ROOT = os.path.join("data", "busi", "Dataset_BUSI_with_GT")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_root(root: str) -> str:
    """Locate the dataset whether or not Hydra has chdir'd into an output dir.

    Runs launched with hydra.job.chdir=true execute from outputs/<run>/, so a
    relative path like "data/busi/..." no longer points at the repo. Falling
    back to a repo-root-relative lookup keeps one config working under both
    launch styles, instead of requiring an absolute path on every command.
    """
    if os.path.isdir(root):
        return root
    if not os.path.isabs(root):
        candidate = os.path.join(_REPO_ROOT, root)
        if os.path.isdir(candidate):
            return candidate
    return root


def _index_files(root: str) -> tuple[list[str], list[int]]:
    root = _resolve_root(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"BUSI not found at {root}. Download it with:\n"
            f"  python -m kaggle datasets download "
            f"-d aryashah2k/breast-ultrasound-images-dataset -p data/busi --unzip"
        )

    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Expected class directory missing: {cls_dir}")
        for fn in sorted(os.listdir(cls_dir)):
            if "_mask" in fn:                      # segmentation ground truth, not an input
                continue
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            paths.append(os.path.join(cls_dir, fn))
            labels.append(label)

    if not paths:
        raise RuntimeError(f"No images found under {root} after filtering masks.")
    return paths, labels


def _stratified_split(labels: list[int]) -> tuple[list[int], list[int]]:
    train_idx, val_idx = [], []
    generator = torch.Generator().manual_seed(SEED)
    for label in range(NUM_CLASSES):
        idx = [i for i, l in enumerate(labels) if l == label]
        perm = torch.randperm(len(idx), generator=generator).tolist()
        idx = [idx[p] for p in perm]
        n_val = max(1, int(len(idx) * VAL_FRACTION))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return sorted(train_idx), sorted(val_idx)


class BUSIDataset(Dataset):
    """Ultrasound frames are single-channel; converted to RGB so the same
    3-channel patch-embedding stem works unmodified across every model."""

    def __init__(self, paths: list[str], labels: list[int], indices: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = Image.open(self.paths[j]).convert("RGB")
        label = self.labels[j]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_busi_datasets(train_transform, val_transform, root: str = DEFAULT_ROOT):
    """Returns (train_dataset, val_dataset) over a fixed stratified split."""
    paths, labels = _index_files(root)
    train_idx, val_idx = _stratified_split(labels)

    counts = {cls: labels.count(i) for i, cls in enumerate(CLASSES)}
    print(f"[busi] {len(paths)} images {counts} -> "
          f"train {len(train_idx)} / val {len(val_idx)} (stratified, seed={SEED})")

    return (
        BUSIDataset(paths, labels, train_idx, transform=train_transform),
        BUSIDataset(paths, labels, val_idx, transform=val_transform),
    )
