"""
BUS-BRA dataset loader -- 1,875 breast ultrasound images, 1,064 patients,
binary benign/malignant, biopsy-confirmed for most cases.

Gomez-Flores, Gregorio-Calas, Pereira, "BUS-BRA: A Breast Ultrasound Dataset
for Assessing Computer-aided Diagnosis Systems," Medical Physics 51, 2024.
CC BY 4.0. Obtained via Kaggle (orvile/bus-bra-a-breast-ultrasound-dataset).

Chosen over (in addition to) BUSI for two reasons:

1. BUSI's public release carries no patient IDs, so a patient-level split is
   not constructible -- images from the same patient can land on both sides
   of the split, which the project's own paper.md §5.7 states as a caveat.
   BUS-BRA's `Case` column *is* a patient ID (both bus_0001-l and bus_0001-r
   are the same patient, left/right), so the split below is genuinely
   patient-level: no case ID appears on both sides.
2. Binary pathology label (benign/malignant) at a 68/32 split, vs BUSI's
   three-way 56/27/17 -- less severe imbalance, one fewer decision boundary.

Two images may share a Case (same patient, different view); stratification
groups by Case first and assigns the *whole* case to one side of the split,
using each case's majority pathology label if a case somehow mixed labels
(not observed in practice, but handled rather than assumed away).
"""

import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

SEED = 42
VAL_FRACTION = 0.2
NUM_CLASSES = 2

CLASSES = ["benign", "malignant"]
DEFAULT_ROOT = os.path.join("data", "busbra", "BUSBRA", "BUSBRA")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_root(root: str) -> str:
    """Same fallback as src/data/busi.py: works whether or not Hydra has
    chdir'd into an output dir."""
    if os.path.isdir(root):
        return root
    if not os.path.isabs(root):
        candidate = os.path.join(_REPO_ROOT, root)
        if os.path.isdir(candidate):
            return candidate
    return root


def _index_files(root: str) -> tuple[list[str], list[int], list[str]]:
    root = _resolve_root(root)
    csv_path = os.path.join(root, "bus_data.csv")
    images_dir = os.path.join(root, "Images")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"BUS-BRA labels not found at {csv_path}. Download it with:\n"
            f"  kaggle datasets download -d orvile/bus-bra-a-breast-ultrasound-dataset "
            f"-p data/busbra --unzip"
        )

    df = pd.read_csv(csv_path)
    paths, labels, cases = [], [], []
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, f"{row['ID']}.png")
        if not os.path.isfile(img_path):
            continue
        pathology = str(row["Pathology"]).strip().lower()
        if pathology not in CLASSES:
            continue
        paths.append(img_path)
        labels.append(CLASSES.index(pathology))
        cases.append(str(row["Case"]))

    if not paths:
        raise RuntimeError(f"No images found under {images_dir} matching {csv_path}.")
    return paths, labels, cases


def _patient_level_split(labels: list[int], cases: list[str]) -> tuple[list[int], list[int]]:
    """Group image indices by Case (patient), assign each whole case to train
    or val, stratified by each case's majority pathology label. No case ID
    appears in both splits."""
    case_to_indices: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        case_to_indices.setdefault(case, []).append(i)

    case_to_label: dict[str, int] = {}
    for case, idxs in case_to_indices.items():
        case_labels = [labels[i] for i in idxs]
        case_to_label[case] = max(set(case_labels), key=case_labels.count)  # majority vote

    case_ids = sorted(case_to_indices.keys())
    generator = torch.Generator().manual_seed(SEED)

    train_idx, val_idx = [], []
    for cls in range(NUM_CLASSES):
        cls_cases = [c for c in case_ids if case_to_label[c] == cls]
        perm = torch.randperm(len(cls_cases), generator=generator).tolist()
        cls_cases = [cls_cases[p] for p in perm]
        n_val_cases = max(1, int(round(len(cls_cases) * VAL_FRACTION)))
        for case in cls_cases[:n_val_cases]:
            val_idx.extend(case_to_indices[case])
        for case in cls_cases[n_val_cases:]:
            train_idx.extend(case_to_indices[case])

    return sorted(train_idx), sorted(val_idx)


class BUSBRADataset(Dataset):
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


def build_busbra_datasets(train_transform, val_transform, root: str = DEFAULT_ROOT):
    """Returns (train_dataset, val_dataset) over a fixed, patient-level
    stratified split -- no patient (Case) appears on both sides."""
    paths, labels, cases = _index_files(root)
    train_idx, val_idx = _patient_level_split(labels, cases)

    n_train_patients = len({cases[i] for i in train_idx})
    n_val_patients = len({cases[i] for i in val_idx})
    counts = {cls: labels.count(i) for i, cls in enumerate(CLASSES)}
    print(f"[busbra] {len(paths)} images {counts} from {n_train_patients + n_val_patients} patients -> "
          f"train {len(train_idx)} imgs / {n_train_patients} patients, "
          f"val {len(val_idx)} imgs / {n_val_patients} patients "
          f"(patient-level stratified, seed={SEED})")

    return (
        BUSBRADataset(paths, labels, train_idx, transform=train_transform),
        BUSBRADataset(paths, labels, val_idx, transform=val_transform),
    )
