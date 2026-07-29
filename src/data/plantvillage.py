"""
PlantVillage dataset loader — 54,303 leaf images, 38 crop-disease classes.

Source: dpdl-benchmark/plant_village (Hugging Face), parquet shards with
embedded PNG bytes + integer label. No official train/val split is provided
upstream, so we build our own stratified split (kept fixed via SEED so it's
reproducible across runs, and shared between train/val instances so the full
54k-row table is only loaded into memory once).
"""

import io

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

SEED = 42
VAL_FRACTION = 0.15
NUM_CLASSES = 38


def _load_full_table() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    dfs = []
    for i in range(13):
        fn = f"data/train-{i:05d}-of-00013.parquet"
        p = hf_hub_download(repo_id="dpdl-benchmark/plant_village", repo_type="dataset", filename=fn)
        dfs.append(pd.read_parquet(p, columns=["image", "label"]))
    return pd.concat(dfs, ignore_index=True)


def _stratified_split(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    train_idx, val_idx = [], []
    rng_state = torch.Generator().manual_seed(SEED)
    for label in sorted(df["label"].unique()):
        idx = df.index[df["label"] == label].to_numpy()
        perm = torch.randperm(len(idx), generator=rng_state).numpy()
        idx = idx[perm]
        n_val = max(1, int(len(idx) * VAL_FRACTION))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return train_idx, val_idx


class PlantVillageDataset(Dataset):
    """Thin view over a shared, pre-loaded table + fixed index list.
    Use `build_plantvillage_datasets()` below rather than constructing directly,
    so train/val share one load of the underlying parquet data."""

    def __init__(self, df: pd.DataFrame, indices: list[int], transform=None):
        self.df = df
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.iloc[self.indices[i]]
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_plantvillage_datasets(train_transform, val_transform):
    """Loads the full table once, returns (train_dataset, val_dataset)."""
    df = _load_full_table()
    train_idx, val_idx = _stratified_split(df)
    train_ds = PlantVillageDataset(df, train_idx, transform=train_transform)
    val_ds = PlantVillageDataset(df, val_idx, transform=val_transform)
    return train_ds, val_ds
