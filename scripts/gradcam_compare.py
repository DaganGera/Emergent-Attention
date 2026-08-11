"""
Grad-CAM comparison across all five trained architectures.

Produces a panel figure (rows = example images, columns = models) in the same
style as the standard "which pixels drove the prediction" comparison: an RGB
input followed by one jet-colormap CAM overlay per model.

Classification head differs per architecture, so the CAM target layer differs
too -- picking the wrong one silently gives a zero or meaningless gradient:

  - DeiT-Tiny / ViT-Tiny (timm, pool_type="token"): the head reads only the
    CLS token after the final block. The last block's own output has zero
    gradient at patch-token positions (LayerNorm is elementwise and the head
    never looks at them), so the hook goes one block earlier -- blocks[-2],
    the input to the last block, whose self-attention is what mixes patch
    information into CLS.
  - Swin-Tiny (timm, pool_type="avg") and both NCA models (GAP over patch
    tokens -- see nca_vit.py forward_features / hybrid_nca_vit.py forward):
    the head pools every spatial/token position directly, so hooking the
    true last block/stage gives a non-zero gradient everywhere.

Usage:
    python scripts/gradcam_compare.py --classes leopard bee --output figures/gradcam_comparison.png
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT
from src.models.baselines import create_baseline
from src.utils.checkpoint import load_checkpoint


_CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
_CIFAR100_STD = torch.tensor([0.2675, 0.2565, 0.2761])

CKPT_ROOT = "checkpoints"

MODEL_SPECS = {
    "DeiT-Tiny": dict(
        ckpt=f"{CKPT_ROOT}/deit_tiny_patch16_224/cifar100/seed42/best.pt",
        build=lambda nc: create_baseline("deit_tiny", num_classes=nc),
        target_layer=lambda m: m.blocks[-2],
        token_format="sequence_with_cls",
    ),
    "ViT-Tiny": dict(
        ckpt=f"{CKPT_ROOT}/vit_tiny_patch16_224/cifar100/seed42/best.pt",
        build=lambda nc: create_baseline("vit_tiny", num_classes=nc),
        target_layer=lambda m: m.blocks[-2],
        token_format="sequence_with_cls",
    ),
    "Swin-Tiny": dict(
        ckpt=f"{CKPT_ROOT}/swin_tiny_patch4_window7_224/cifar100/seed42/best.pt",
        build=lambda nc: create_baseline("swin_tiny", num_classes=nc),
        target_layer=lambda m: m.layers[-1],
        token_format="spatial_channels_last",
    ),
    "NCA-ViT (pure)": dict(
        ckpt=f"{CKPT_ROOT}/nca_vit_tiny/cifar100/seed42/best.pt",
        build=lambda nc: NCAViT(num_classes=nc),
        target_layer=lambda m: m.blocks[-1],
        token_format="sequence_with_cls",
    ),
    "Hybrid NCA-ViT (ours)": dict(
        ckpt=f"{CKPT_ROOT}/nca_vit_hybrid/cifar100/seed42/best.pt",
        build=lambda nc: HybridNCAViT(
            num_classes=nc, embed_dim=192, nca_depth=6, attn_depth=6, nca_steps=4,
            filter_names=["sobel_x", "sobel_y", "identity"], nca_hidden_dim=384,
            stochastic_rate=0.5, mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
            learnable_filters=True,
        ),
        target_layer=lambda m: m.attn_blocks[-1],
        token_format="sequence_with_cls",
    ),
}


def build_val_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_CIFAR100_MEAN.tolist(), _CIFAR100_STD.tolist()),
    ])


def denormalize(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) uint8-range float array in [0,1]."""
    img = img.detach().cpu() * _CIFAR100_STD[:, None, None] + _CIFAR100_MEAN[:, None, None]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def load_model(spec: dict, num_classes: int, device: torch.device) -> nn.Module:
    model = spec["build"](num_classes).to(device)
    state = load_checkpoint(spec["ckpt"])
    state = state["model"] if "model" in state else state
    model.load_state_dict(state)
    model.eval()
    return model


def compute_gradcam(model: nn.Module, spec: dict, image: torch.Tensor, target_class: int) -> np.ndarray:
    """
    image: (1, 3, H, W) already normalized, on the model's device.
    Returns a (H, W) heatmap in [0, 1], resized to the input's spatial size.
    """
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

    if spec["token_format"] == "sequence_with_cls":
        # (N+1, D) -> drop CLS -> (N, D) -> (Hp, Wp, D)
        patch_act = act[1:]
        patch_grad = grad[1:]
        n = patch_act.shape[0]
        side = int(round(n ** 0.5))
        patch_act = patch_act.reshape(side, side, -1)
        patch_grad = patch_grad.reshape(side, side, -1)
    else:  # spatial_channels_last: (H, W, D)
        patch_act = act
        patch_grad = grad

    weights = patch_grad.mean(dim=(0, 1))                 # (D,)
    cam = torch.relu((patch_act * weights).sum(dim=-1))   # (Hp, Wp)
    cam = cam / (cam.max() + 1e-8)

    cam = cam.unsqueeze(0).unsqueeze(0)                    # (1,1,Hp,Wp)
    cam = torch.nn.functional.interpolate(
        cam, size=image.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    return cam.cpu().numpy()


def overlay(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * base_rgb + alpha * heat, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=["leopard", "bee"],
                         help="CIFAR-100 fine class names to visualize (one row each)")
    parser.add_argument("--output", default="figures/gradcam_comparison.png")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100

    raw_test = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True)
    class_to_idx = {name: i for i, name in enumerate(raw_test.classes)}
    for name in args.classes:
        if name not in class_to_idx:
            raise ValueError(f"'{name}' is not a CIFAR-100 class. Available: {raw_test.classes}")

    transform = build_val_transform(224)
    rows = []
    for name in args.classes:
        target_idx = class_to_idx[name]
        pil_img = next(img for img, label in raw_test if label == target_idx)
        rows.append((name, target_idx, transform(pil_img).unsqueeze(0)))

    print("Loading models...")
    models = {mname: load_model(spec, num_classes, device) for mname, spec in MODEL_SPECS.items()}

    col_names = ["Input"] + list(MODEL_SPECS.keys())
    fig, axes = plt.subplots(
        len(rows), len(col_names),
        figsize=(2.4 * len(col_names), 2.4 * len(rows)),
    )
    if len(rows) == 1:
        axes = axes[None, :]

    for r, (name, target_idx, img_t) in enumerate(rows):
        img_t = img_t.to(device)
        base_rgb = denormalize(img_t[0])

        axes[r, 0].imshow(base_rgb)
        axes[r, 0].set_ylabel(name.replace("_", " ").title(), fontsize=13)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Input", fontsize=13)

        for c, (mname, spec) in enumerate(MODEL_SPECS.items(), start=1):
            cam = compute_gradcam(models[mname], spec, img_t, target_idx)
            axes[r, c].imshow(overlay(base_rgb, cam))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(mname, fontsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(args.output)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
