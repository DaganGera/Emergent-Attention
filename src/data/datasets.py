"""
Dataset loaders for NCA-ViT experiments.
Supports CIFAR-100 with 224x224 resizing to match ViT patch grids.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from src.data.plantvillage import build_plantvillage_datasets
from src.data.busi import build_busi_datasets
from src.data.busbra import build_busbra_datasets


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ImageNet stats -- standard default for natural (non-CIFAR-scale) RGB photos
_PLANTVILLAGE_MEAN = (0.485, 0.456, 0.406)
_PLANTVILLAGE_STD  = (0.229, 0.224, 0.225)


def _build_plantvillage_train_transform(img_size: int) -> T.Compose:
    # Native images are ~256x256, not CIFAR's 32x32, so this crops/resizes
    # rather than upsampling from a tiny source like the CIFAR-100 path does.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(_PLANTVILLAGE_MEAN, _PLANTVILLAGE_STD),
        T.RandomErasing(p=0.25),
    ])


def _build_plantvillage_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_PLANTVILLAGE_MEAN, _PLANTVILLAGE_STD),
    ])


def _build_busi_train_transform(img_size: int) -> T.Compose:
    # Ultrasound frames are ~500x500 grayscale. Two deliberate differences from
    # the PlantVillage path: no RandAugment, and no vertical flipping. Ultrasound
    # geometry is not arbitrary -- depth runs top-to-bottom from the transducer,
    # so a vertical flip produces an image that cannot occur. RandAugment's
    # posterise/solarise/equalise operations distort the speckle statistics that
    # this experiment is specifically about, so they are left out too.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(_PLANTVILLAGE_MEAN, _PLANTVILLAGE_STD),
        T.RandomErasing(p=0.25),
    ])


def _build_busi_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_PLANTVILLAGE_MEAN, _PLANTVILLAGE_STD),
    ])


def _build_busbra_train_transform(img_size: int) -> T.Compose:
    # Same rationale as BUSI (src/data/busi.py): no vertical flip (ultrasound
    # depth runs top-to-bottom from the transducer, a vertical flip is not a
    # physically valid image), no RandAugment (its posterise/solarise/equalise
    # ops distort the speckle statistics under study).
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(_PLANTVILLAGE_MEAN, _PLANTVILLAGE_STD),
        T.RandomErasing(p=0.25),
    ])


def _build_busbra_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_PLANTVILLAGE_MEAN, _PLANTVILLAGE_STD),
    ])


def _build_train_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        T.RandomErasing(p=0.25),
    ])


def _build_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])


def _subsample_train(dataset, fraction: float, seed: int = 42):
    """
    Class-stratified subsample of a training set, for data-efficiency sweeps.

    The seed is fixed and deliberately independent of the training seed, so
    every model compared at a given fraction sees the exact same images --
    otherwise the curve measures which subset was drawn, not which architecture
    learns from less data.
    """
    if fraction >= 1.0:
        return dataset

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise ValueError(
            "Dataset exposes neither .targets nor .labels; cannot stratify subsample."
        )
    targets = np.asarray(targets)

    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(targets):
        idx = np.flatnonzero(targets == cls)
        n = max(1, int(round(len(idx) * fraction)))
        keep.append(rng.choice(idx, size=n, replace=False))
    keep = np.sort(np.concatenate(keep)).tolist()

    print(f"[data] train_fraction={fraction} -> {len(keep)}/{len(targets)} images "
          f"({len(np.unique(targets))} classes, stratified, subsample_seed={seed})")
    return Subset(dataset, keep)


def build_loaders(cfg) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        cfg: Hydra DictConfig with .data and .model sub-configs.

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = cfg.data
    img_size = cfg.model.get("img_size", 224)
    root = data_cfg.get("data_root", "./data")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)

    dataset_name = data_cfg.dataset.lower()
    if dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=_build_train_transform(img_size),
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=_build_val_transform(img_size),
        )
    elif dataset_name == "plantvillage":
        train_dataset, val_dataset = build_plantvillage_datasets(
            train_transform=_build_plantvillage_train_transform(img_size),
            val_transform=_build_plantvillage_val_transform(img_size),
        )
    elif dataset_name == "busi":
        train_dataset, val_dataset = build_busi_datasets(
            train_transform=_build_busi_train_transform(img_size),
            val_transform=_build_busi_val_transform(img_size),
            root=data_cfg.get("busi_root", "data/busi/Dataset_BUSI_with_GT"),
        )
    elif dataset_name == "busbra":
        train_dataset, val_dataset = build_busbra_datasets(
            train_transform=_build_busbra_train_transform(img_size),
            val_transform=_build_busbra_val_transform(img_size),
            root=data_cfg.get("busbra_root", "data/busbra/BUSBRA/BUSBRA"),
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported: 'cifar100', 'plantvillage', 'busi', 'busbra'."
        )

    # Data-efficiency sweeps: keep only a stratified slice of the train split.
    # The val split is never subsampled -- every point on the curve is scored
    # against the identical test set.
    train_dataset = _subsample_train(
        train_dataset,
        float(data_cfg.get("train_fraction", 1.0)),
        seed=int(data_cfg.get("subsample_seed", 42)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
