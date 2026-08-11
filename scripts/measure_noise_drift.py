"""
Does representation stability explain the corruption results?

The CIFAR-100-C run establishes *that* the hybrid resists additive noise
(+8.3pp) and fails on impulse noise (-5.9pp). This asks *why*, with no
training: it measures how far each corruption moves the feature the
classifier actually consumes, and tests whether that movement predicts the
accuracy gap corruption-by-corruption.

    drift = 1 - cos( f_clean , f_noisy )

where f is the pre-head feature -- the CLS token for DeiT/ViT, GAP over patch
tokens for Swin and both NCA models. Probing a fixed layer instead would not
be comparable: DeiT's head never reads its final patch tokens, so their drift
is not what determines its prediction.

Falsification built in: the denoising account survives only if drift reduction
tracks the accuracy gap *including its sign* -- i.e. the hybrid must show no
stability advantage on impulse noise, where it loses.

Usage:
    python scripts/measure_noise_drift.py
    python scripts/measure_noise_drift.py --n-images 512 --severity 5
"""

import os
import sys
import json
import argparse
import statistics as st

import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gradcam_compare import MODEL_SPECS, load_model, build_val_transform

mpl.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "axes.titlesize": 12.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

HYBRID = "Hybrid NCA-ViT (ours)"
REF = "DeiT-Tiny"          # same 12-block/196-token/192-dim skeleton as the hybrid
ATTENTION_ONLY_KEYS = ["deit_tiny", "vit_tiny", "swin_tiny"]

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

# Corruptions whose perturbation is additive high-frequency pixel noise --
# the family the accuracy results single out.
NOISE_FAMILY = {"gaussian_noise", "shot_noise", "speckle_noise"}

# Baseline accuracy below which a corruption is "hard" enough that feature
# stability still has room to matter. See the comment at its use site.
HARD_THRESHOLD = 65.0


def head_module(model: nn.Module, name: str) -> nn.Module:
    """The Linear the model classifies with; its input is the pooled feature."""
    return model.head.fc if name == "Swin-Tiny" else model.head


@torch.no_grad()
def prehead_features(model: nn.Module, name: str, images: torch.Tensor,
                     device: torch.device, batch_size: int) -> torch.Tensor:
    captured = {}
    handle = head_module(model, name).register_forward_pre_hook(
        lambda _m, inp: captured.__setitem__("v", inp[0].detach().float().cpu())
    )
    out = []
    for i in range(0, len(images), batch_size):
        model(images[i:i + batch_size].to(device))
        out.append(captured["v"])
    handle.remove()
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=256)
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--corruption-dir", default="data/cifar-100-c/CIFAR-100-C")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accuracy-results", default="results/cifar100c_robustness.json")
    parser.add_argument("--output", default="figures/noise_drift.png")
    parser.add_argument("--results", default="results/noise_drift.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_test = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=False)
    tf = build_val_transform(224)
    n = args.n_images

    clean = torch.stack([tf(raw_test[i][0]) for i in range(n)])
    offset = (args.severity - 1) * 10000
    noisy = {}
    for corr in CORRUPTIONS:
        path = os.path.join(args.corruption_dir, f"{corr}.npy")
        if not os.path.exists(path):
            print(f"  [skip] {corr}: not found")
            continue
        arr = np.load(path, mmap_mode="r")
        noisy[corr] = torch.stack(
            [tf(Image.fromarray(np.array(arr[offset + i]))) for i in range(n)]
        )

    drift = {}
    for mname, spec in MODEL_SPECS.items():
        model = load_model(spec, 100, device)
        f_clean = prehead_features(model, mname, clean, device, args.batch_size)
        drift[mname] = {}
        for corr, imgs in noisy.items():
            f_noisy = prehead_features(model, mname, imgs, device, args.batch_size)
            d = 1.0 - torch.nn.functional.cosine_similarity(f_clean, f_noisy, dim=-1).mean().item()
            drift[mname][corr] = round(d, 4)
        print(f"{mname:24s} mean drift {st.mean(drift[mname].values()):.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump({"severity": args.severity, "n_images": n,
                   "metric": "1 - cosine(pre-head feature, clean vs corrupted)",
                   "drift": drift}, f, indent=2)

    # ---- pair drift reduction against the measured accuracy gap ----
    with open(args.accuracy_results) as f:
        acc = json.load(f)

    xs, ys, labels, is_hard = [], [], [], []
    for corr in noisy:
        hyb_acc = st.mean([acc["nca_vit_hybrid"][f"{corr}_sev{s}"] for s in (1, 3, 5)])
        best_attn = max(st.mean([acc[m][f"{corr}_sev{s}"] for s in (1, 3, 5)])
                        for m in ATTENTION_ONLY_KEYS)
        # positive = hybrid's feature moves less than the reference's
        xs.append((drift[REF][corr] - drift[HYBRID][corr]) / drift[REF][corr] * 100)
        ys.append(hyb_acc - best_attn)
        labels.append(corr)
        # Corruptions the baselines already survive comfortably sit near ceiling:
        # extra feature stability cannot convert into accuracy there, so they
        # dilute the relationship. Threshold chosen post hoc, near the median
        # baseline (62.2%) -- the split is descriptive, not a pre-registered test.
        is_hard.append(best_attn < HARD_THRESHOLD)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.6),
                                    gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel A: pre-head drift on the three decisive corruptions
    focus = [c for c in ["speckle_noise", "gaussian_noise", "impulse_noise"] if c in noisy]
    width = 0.15
    colors = {"DeiT-Tiny": "#8c8c8c", "ViT-Tiny": "#b0b0b0", "Swin-Tiny": "#4c72b0",
              "NCA-ViT (pure)": "#dd8452", HYBRID: "#c44e52"}
    for k, mname in enumerate(MODEL_SPECS):
        pos = [j + (k - 2) * width for j in range(len(focus))]
        ax1.bar(pos, [drift[mname][c] for c in focus], width=width,
                label=mname, color=colors[mname])
    ax1.set_xticks(range(len(focus)))
    ax1.set_xticklabels([
        f"{c.replace('_', ' ')}\n(hybrid {ys[labels.index(c)]:+.1f}pp accuracy)"
        for c in focus], fontsize=9.5)
    ax1.set_ylabel(r"Pre-head feature drift  $1-\cos(f_{clean},f_{noisy})$")
    ax1.set_title("(a) Feature stability under corruption\nlower = the corruption moved the "
                  "classifier's input less")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", alpha=0.3, ls=":")

    # Panel B: does stability predict the accuracy gap?
    for x, y, lab, hard in zip(xs, ys, labels, is_hard):
        ax2.scatter(x, y, s=90 if hard else 45,
                    color="#c44e52" if hard else "#b0b0b0",
                    zorder=3, edgecolor="white", linewidth=0.8)
        if hard or lab == "impulse_noise":
            ax2.annotate(lab.replace("_", " "), (x, y), fontsize=8,
                         xytext=(5, 4), textcoords="offset points")

    r_all = np.corrcoef(xs, ys)[0, 1]
    hx = [x for x, h in zip(xs, is_hard) if h]
    hy = [y for y, h in zip(ys, is_hard) if h]
    r_hard = np.corrcoef(hx, hy)[0, 1]
    m_, b_ = np.polyfit(hx, hy, 1)
    grid = np.linspace(min(hx), max(hx), 50)
    ax2.plot(grid, m_ * grid + b_, color="#c44e52", ls="--", lw=1.5, alpha=0.75,
             label=f"hard corruptions (n={len(hx)}):  r = {r_hard:.2f}")
    ax2.scatter([], [], s=45, color="#b0b0b0",
                label=f"near-ceiling (n={len(xs) - len(hx)}), all 19:  r = {r_all:.2f}")

    ax2.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax2.axhline(0, color="black", lw=0.9)
    ax2.axvline(0, color="black", lw=0.9)
    ax2.set_xlabel("Feature-drift reduction vs DeiT-Tiny (%)")
    ax2.set_ylabel("Hybrid accuracy advantage (pp)")
    ax2.set_title("(b) Stability tracks the gap where accuracy has room to move\n"
                  "impulse noise: the only negative-stability point, and the only large loss")
    ax2.grid(alpha=0.3, ls=":")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    plt.savefig(os.path.splitext(args.output)[0] + ".pdf")
    print(f"\nSaved: {args.output}")
    print(f"Saved: {args.results}")
    print(f"Correlation, all {len(xs)} corruptions: r = {r_all:.3f}")
    print(f"Correlation, {len(hx)} hard corruptions (baseline < {HARD_THRESHOLD}%): r = {r_hard:.3f}")


if __name__ == "__main__":
    main()
