"""
Dataset loaders for NCA-ViT experiments.
Supports CIFAR-100 with 224x224 resizing to match ViT patch grids.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


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
    if dataset_name != "cifar100":
        raise ValueError(f"Unsupported dataset: {dataset_name}. Only 'cifar100' is supported.")

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
