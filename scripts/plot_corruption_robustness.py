"""
Corruption-robustness figures from results/cifar100c_robustness.json.

Panel A: mean accuracy vs corruption severity, one line per architecture.
Panel B: per-corruption-family advantage of the Hybrid over the strongest
         attention-only baseline, which is where the effect actually lives --
         it is concentrated in additive pixel noise, not spread evenly.

Usage:
    python scripts/plot_corruption_robustness.py
"""

import os
import json
import argparse
import statistics as st

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "axes.titlesize": 12.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Clean CIFAR-100 top-1, from each run's own checkpoint (severity-0 anchor).
CLEAN_ACC = {
    "deit_tiny": 72.89, "vit_tiny": 72.33, "swin_tiny": 81.57,
    "nca_vit_tiny": 73.99, "nca_vit_hybrid": 81.29,
}

DISPLAY = {
    "deit_tiny": "DeiT-Tiny", "vit_tiny": "ViT-Tiny", "swin_tiny": "Swin-Tiny",
    "nca_vit_tiny": "NCA-ViT (pure)", "nca_vit_hybrid": "Hybrid NCA-ViT (ours)",
}

STYLE = {
    "deit_tiny": dict(color="#8c8c8c", marker="o", ls="--", lw=1.6),
    "vit_tiny": dict(color="#b0b0b0", marker="^", ls="--", lw=1.6),
    "swin_tiny": dict(color="#4c72b0", marker="s", ls="-", lw=1.8),
    "nca_vit_tiny": dict(color="#dd8452", marker="D", ls="-", lw=1.8),
    "nca_vit_hybrid": dict(color="#c44e52", marker="*", ls="-", lw=2.8, ms=13),
}

# Grouped by what the corruption does to the signal, not by Hendrycks' original
# noise/blur/weather/digital split -- the effect tracks high-frequency additive
# perturbation, which cuts across those categories.
FAMILIES = {
    "Additive pixel noise\n(gaussian, shot, speckle)": ["gaussian_noise", "shot_noise", "speckle_noise"],
    "Compression / resample\n(pixelate, jpeg)": ["pixelate", "jpeg_compression"],
    "Blur\n(defocus, glass, motion, zoom, gaussian)": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "Weather\n(snow, frost, fog, spatter)": ["snow", "frost", "fog", "spatter"],
    "Photometric\n(brightness, contrast, saturate)": ["brightness", "contrast", "saturate"],
    "Geometric\n(elastic transform)": ["elastic_transform"],
    "Impulse noise\n(salt-and-pepper)": ["impulse_noise"],
}

ATTENTION_ONLY = ["deit_tiny", "vit_tiny", "swin_tiny"]


def mean_over(results, model, corruptions, severities):
    vals = [results[model][f"{c}_sev{s}"]
            for c in corruptions for s in severities
            if f"{c}_sev{s}" in results[model]]
    return st.mean(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/cifar100c_robustness.json")
    parser.add_argument("--output", default="figures/corruption_robustness.png")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    models = [m for m in DISPLAY if m in results]
    severities = sorted({int(k.rsplit("_sev", 1)[1]) for k in results[models[0]]})
    all_corruptions = sorted({k.rsplit("_sev", 1)[0] for k in results[models[0]]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                    gridspec_kw={"width_ratios": [1, 1.25]})

    # ---- Panel A: degradation curve ----
    for m in models:
        xs = [0] + severities
        ys = [CLEAN_ACC[m]] + [mean_over(results, m, all_corruptions, [s]) for s in severities]
        ax1.plot(xs, ys, label=DISPLAY[m], **STYLE[m])

    ax1.set_xlabel("Corruption severity  (0 = clean)")
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_title("(a) Degradation under corruption\nmean over 19 CIFAR-100-C corruptions")
    ax1.set_xticks([0] + severities)
    ax1.grid(alpha=0.3, ls=":")
    ax1.legend(frameon=False, loc="lower left")

    # ---- Panel B: where the advantage lives ----
    names, gaps = [], []
    for fam, corrs in FAMILIES.items():
        hyb = mean_over(results, "nca_vit_hybrid", corrs, severities)
        best_attn = max(mean_over(results, m, corrs, severities) for m in ATTENTION_ONLY)
        names.append(fam)
        gaps.append(hyb - best_attn)

    order = sorted(range(len(gaps)), key=lambda i: gaps[i])
    names = [names[i] for i in order]
    gaps = [gaps[i] for i in order]
    colors = ["#c44e52" if g > 0 else "#8c8c8c" for g in gaps]

    bars = ax2.barh(range(len(gaps)), gaps, color=colors, height=0.68)
    ax2.set_yticks(range(len(gaps)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.axvline(0, color="black", lw=1.0)
    ax2.set_xlabel("Hybrid advantage over best attention-only baseline (pp)")
    ax2.set_title("(b) The advantage is not uniform\nit concentrates in additive high-frequency noise")
    ax2.grid(axis="x", alpha=0.3, ls=":")

    for bar, g in zip(bars, gaps):
        ax2.text(g + (0.22 if g >= 0 else -0.22), bar.get_y() + bar.get_height() / 2,
                 f"{g:+.1f}", va="center", ha="left" if g >= 0 else "right",
                 fontsize=9.5, fontweight="bold",
                 color="#c44e52" if g > 0 else "#5c5c5c")
    ax2.set_xlim(min(gaps) - 2.4, max(gaps) + 2.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    plt.savefig(os.path.splitext(args.output)[0] + ".pdf")
    print(f"Saved: {args.output}")
    print(f"Saved: {os.path.splitext(args.output)[0]}.pdf")


if __name__ == "__main__":
    main()
