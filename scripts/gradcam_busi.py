"""
Grad-CAM comparison on real BUSI ultrasound images: Hybrid NCA-ViT vs. DeiT-Tiny.

Qualitative counterpart to results/busi_v2/*.json (Chapter 37 / paper Sec.5.7):
searches the full BUSI validation split (not hand-picked) for cases where the
hybrid classifies correctly and DeiT does not, and keeps the top 3 by how
confidently wrong DeiT was -- the same non-cherry-picking discipline
scripts/gradcam_compare.py uses for the CIFAR-100 noisy-image figure.

Not one-per-class: at seed 42, DeiT's recall on the "malignant" true class is
already high (83.3%, see results/busi_v2/deit_tiny_busi.json), so there is no
DeiT-wrong/hybrid-right example to find there -- the 68 qualifying candidates
this search finds are all "benign" or "normal" true-class images. Reported as
observed, not forced into an artificial per-class split.

Both checkpoints are the corrected ("v2": no Mixup/CutMix, class-weighted CE,
seed 42) recipe -- the same one and matching the same seed used for the
recipe-fix comparison reported in the paper.

Usage:
    python scripts/gradcam_busi.py --output-dir figures/paper
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hybrid_nca_vit import HybridNCAViT
from src.models.baselines import create_baseline
from src.data.busi import build_busi_datasets, CLASSES
from src.utils.checkpoint import load_checkpoint

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])

# v2 recipe (no Mixup/CutMix, class-weighted CE), seed 42 -- the default seed
# used everywhere else in the project, not cherry-picked for this figure.
DEIT_CKPT = "outputs/exp/busi_v2_deit/checkpoints/deit_tiny_patch16_224/busi/seed42/best.pt"
HYBRID_CKPT = "outputs/exp/busi_v2_hybrid/checkpoints/nca_vit_hybrid/busi/seed42/best.pt"

MODEL_SPECS = {
    "DeiT-Tiny": dict(
        ckpt=DEIT_CKPT,
        build=lambda nc: create_baseline("deit_tiny", num_classes=nc),
        target_layer=lambda m: m.blocks[-2],
    ),
    "Hybrid NCA-ViT (ours)": dict(
        ckpt=HYBRID_CKPT,
        build=lambda nc: HybridNCAViT(
            num_classes=nc, embed_dim=192, nca_depth=6, attn_depth=6, nca_steps=4,
            filter_names=["sobel_x", "sobel_y", "identity"], nca_hidden_dim=384,
            stochastic_rate=0.5, mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
            learnable_filters=True,
        ),
        target_layer=lambda m: m.attn_blocks[-1],
    ),
}


def denormalize(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu() * _STD[:, None, None] + _MEAN[:, None, None]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def load_model(spec: dict, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = spec["build"](num_classes).to(device)
    state = load_checkpoint(spec["ckpt"])
    state = state["model"] if "model" in state else state
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image: torch.Tensor):
    logits = model(image)
    probs = F.softmax(logits[0], dim=0)
    pred = int(probs.argmax())
    return pred, probs.cpu().numpy()


def compute_gradcam(model: torch.nn.Module, spec: dict, image: torch.Tensor, target_class: int):
    activation = {}

    def hook(_module, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t.retain_grad()
        activation["value"] = t

    handle = spec["target_layer"](model).register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logits = model(image)
    score = logits[0, target_class]
    score.backward()
    handle.remove()

    act = activation["value"][0].detach()
    grad = activation["value"].grad[0].detach()

    # sequence_with_cls for both models here: drop CLS (index 0), reshape to grid.
    patch_act, patch_grad = act[1:], grad[1:]
    n = patch_act.shape[0]
    side = int(round(n ** 0.5))
    patch_act = patch_act.reshape(side, side, -1)
    patch_grad = patch_grad.reshape(side, side, -1)

    weights = patch_grad.mean(dim=(0, 1))
    cam = torch.relu((patch_act * weights).sum(dim=-1))
    cam = cam / (cam.max() + 1e-8)
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.cpu().numpy()


def overlay(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * base_rgb + alpha * heat, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures/paper")
    parser.add_argument("--data-root", default="data/busi/Dataset_BUSI_with_GT")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3

    val_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_MEAN.tolist(), _STD.tolist()),
    ])
    _, val_ds = build_busi_datasets(train_transform=None, val_transform=val_transform, root=args.data_root)
    # raw (untransformed) images for display, same split/order as val_ds
    _, val_ds_raw = build_busi_datasets(train_transform=None, val_transform=None, root=args.data_root)

    print("Loading models...")
    models = {name: load_model(spec, num_classes, device) for name, spec in MODEL_SPECS.items()}
    deit, hybrid = models["DeiT-Tiny"], models["Hybrid NCA-ViT (ours)"]

    print(f"Searching {len(val_ds)} validation images for DeiT-wrong / Hybrid-right cases...")
    candidates = []  # (deit_wrong_confidence, index, label, deit_pred)
    for i in range(len(val_ds)):
        img_t, label = val_ds[i]
        img_t = img_t.unsqueeze(0).to(device)
        deit_pred, deit_probs = predict(deit, img_t)
        hybrid_pred, _ = predict(hybrid, img_t)
        if deit_pred != label and hybrid_pred == label:
            candidates.append((float(deit_probs[deit_pred]), i, label, deit_pred))

    candidates.sort(key=lambda c: -c[0])
    # One slot per true class that has at least one candidate (its strongest one),
    # so a class with fewer/weaker candidates isn't crowded out by a class with
    # many strong ones; remaining slots (if any) filled by next-best overall.
    top3, seen_classes, seen_idx = [], set(), set()
    for c in candidates:
        _, idx, label, _ = c
        if label not in seen_classes:
            top3.append(c)
            seen_classes.add(label)
            seen_idx.add(idx)
        if len(top3) == 3:
            break
    for c in candidates:
        if len(top3) == 3:
            break
        if c[1] not in seen_idx:
            top3.append(c)
            seen_idx.add(c[1])
    top3.sort(key=lambda c: -c[0])
    print(f"{len(candidates)} qualifying candidates (DeiT wrong, hybrid right); keeping one per available true class, filled to 3 by next-best overall:")
    for conf, idx, label, deit_pred in top3:
        print(f"  val#{idx} true={CLASSES[label]:>9s} -> DeiT said '{CLASSES[deit_pred]}' ({conf*100:.1f}% confident)")

    os.makedirs(args.output_dir, exist_ok=True)
    for rank, (_, idx, _, _) in enumerate(top3, start=1):
        img_t, label = val_ds[idx]
        raw_img, _ = val_ds_raw[idx]
        img_t = img_t.unsqueeze(0).to(device)
        base_rgb = np.array(raw_img.resize((224, 224))) / 255.0

        deit_pred, deit_probs = predict(deit, img_t)
        hybrid_pred, hybrid_probs = predict(hybrid, img_t)
        deit_cam = compute_gradcam(deit, MODEL_SPECS["DeiT-Tiny"], img_t, label)
        hybrid_cam = compute_gradcam(hybrid, MODEL_SPECS["Hybrid NCA-ViT (ours)"], img_t, label)

        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
        cls_name = CLASSES[label].title()

        axes[0].imshow(base_rgb)
        axes[0].set_title(f"Input\ntrue: {cls_name}", fontsize=12)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(overlay(base_rgb, deit_cam))
        wrong = "✗" if deit_pred != label else "✓"
        axes[1].set_title(f"DeiT-Tiny {wrong}\npred: {CLASSES[deit_pred].title()} ({deit_probs[deit_pred]*100:.0f}%)",
                           fontsize=12, color=("#9c3b3b" if deit_pred != label else "#2a6b2d"))
        axes[1].set_xticks([]); axes[1].set_yticks([])

        axes[2].imshow(overlay(base_rgb, hybrid_cam))
        right = "✓" if hybrid_pred == label else "✗"
        axes[2].set_title(f"Hybrid NCA-ViT (ours) {right}\npred: {CLASSES[hybrid_pred].title()} ({hybrid_probs[hybrid_pred]*100:.0f}%)",
                           fontsize=12, color=("#2a6b2d" if hybrid_pred == label else "#9c3b3b"))
        axes[2].set_xticks([]); axes[2].set_yticks([])

        fig.tight_layout()
        out_path = os.path.join(args.output_dir, f"busi_gradcam_{rank}_{CLASSES[label]}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # The reverse case, malignant only: DeiT's malignant recall (83-100% across
    # all 3 seeds) beats Hybrid's (14-64%) -- DeiT achieves this by predicting
    # "malignant" far more indiscriminately (its benign recall is 2-9% in the
    # same seeds), but the asymmetry on this one class is real and belongs in
    # the record, not just the wins. Saved separately, clearly labeled.
    print("Searching for the reverse case (DeiT right, Hybrid wrong, true=malignant)...")
    reverse_candidates = []
    malignant_idx = CLASSES.index("malignant")
    for i in range(len(val_ds)):
        img_t, label = val_ds[i]
        if label != malignant_idx:
            continue
        img_t = img_t.unsqueeze(0).to(device)
        deit_pred, deit_probs = predict(deit, img_t)
        hybrid_pred, hybrid_probs = predict(hybrid, img_t)
        if deit_pred == label and hybrid_pred != label:
            reverse_candidates.append((float(hybrid_probs[hybrid_pred]), i, deit_pred, hybrid_pred))

    if reverse_candidates:
        reverse_candidates.sort(key=lambda c: -c[0])
        _, idx, _, _ = reverse_candidates[0]
        print(f"  {len(reverse_candidates)} candidates; using val#{idx} (Hybrid's most confident miss)")
        img_t, label = val_ds[idx]
        raw_img, _ = val_ds_raw[idx]
        img_t = img_t.unsqueeze(0).to(device)
        base_rgb = np.array(raw_img.resize((224, 224))) / 255.0

        deit_pred, deit_probs = predict(deit, img_t)
        hybrid_pred, hybrid_probs = predict(hybrid, img_t)
        deit_cam = compute_gradcam(deit, MODEL_SPECS["DeiT-Tiny"], img_t, label)
        hybrid_cam = compute_gradcam(hybrid, MODEL_SPECS["Hybrid NCA-ViT (ours)"], img_t, label)

        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
        axes[0].imshow(base_rgb)
        axes[0].set_title("Input\ntrue: Malignant", fontsize=12)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(overlay(base_rgb, deit_cam))
        axes[1].set_title(f"DeiT-Tiny ✓\npred: {CLASSES[deit_pred].title()} ({deit_probs[deit_pred]*100:.0f}%)",
                           fontsize=12, color="#2a6b2d")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        axes[2].imshow(overlay(base_rgb, hybrid_cam))
        axes[2].set_title(f"Hybrid NCA-ViT (ours) ✗\npred: {CLASSES[hybrid_pred].title()} ({hybrid_probs[hybrid_pred]*100:.0f}%)",
                           fontsize=12, color="#9c3b3b")
        axes[2].set_xticks([]); axes[2].set_yticks([])

        fig.tight_layout()
        out_path = os.path.join(args.output_dir, "busi_gradcam_4_malignant_limitation.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        print("  none found at this seed (DeiT recall was 100% on malignant here).")


if __name__ == "__main__":
    main()
