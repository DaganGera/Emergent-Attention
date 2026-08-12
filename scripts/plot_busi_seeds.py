"""
BUSI, 3-seed comparison: does the Hybrid's advantage survive across seeds?

The first single-seed BUSI run was ambiguous (Hybrid won top-1, lost balanced
accuracy / macro-F1). A recipe change (Mixup/CutMix off, class-weighted CE --
see trainer.py) reversed that on seed 42, which could have been the recipe
change or could have been that one seed. This plots all three seeds together
to show which.

Usage:
    python scripts/plot_busi_seeds.py
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
    "xtick.labelsize": 11, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

RESULTS_DIR = "results/busi_v2"
SEEDS = ["42", "123", "7"]
MODELS = {"nca_vit_hybrid": ("Hybrid NCA-ViT (ours)", "#c44e52", "*"),
          "deit_tiny": ("DeiT-Tiny", "#8c8c8c", "o")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/busi_seeds.png")
    args = parser.parse_args()

    # evaluate_busi.py always writes to the SAME "{model}_busi.json" path, so
    # the per-seed numbers below are transcribed from the eval runs at the
    # point each checkpoint was produced (recorded in conversation / logs),
    # not re-read from a single overwritten file.
    data = {
        "nca_vit_hybrid": {"42": (46.98, 42.70), "123": (58.08, 42.37), "7": (44.42, 39.32)},
        "deit_tiny": {"42": (39.82, 28.56), "123": (35.25, 18.28), "7": (34.10, 16.06)},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    metrics = [("Balanced accuracy (%)", 0, ax1), ("Macro-F1 (%)", 1, ax2)]

    for label, idx, ax in metrics:
        for key, (name, color, marker) in MODELS.items():
            vals = [data[key][s][idx] for s in SEEDS]
            xs = [0.9 if key == "nca_vit_hybrid" else 1.1] * len(vals)
            ax.scatter(xs, vals, s=140, color=color, marker=marker,
                      edgecolor="white", linewidth=1, zorder=3, label=name)
            mean = st.mean(vals)
            ax.hlines(mean, (0.9 if key == "nca_vit_hybrid" else 1.1) - 0.08,
                      (0.9 if key == "nca_vit_hybrid" else 1.1) + 0.08,
                      color=color, lw=2.5, zorder=4)
            # Stack labels top-to-bottom by value rather than placing each at its
            # own y -- close values (e.g. 42.70 vs 42.37) otherwise overprint.
            x0 = 0.9 if key == "nca_vit_hybrid" else 1.1
            for rank, (s, v) in enumerate(sorted(zip(SEEDS, vals), key=lambda t: -t[1])):
                ax.annotate(f"seed {s}", (x0, v), fontsize=7.5,
                           xytext=(7, 6 - rank * 11), textcoords="offset points",
                           color=color, alpha=0.85)
        if idx == 0:
            ax.axhline(33.33, color="gray", ls=":", lw=1, alpha=0.6)
            ax.text(1.55, 33.8, "chance", fontsize=8, color="gray")
        ax.set_xlim(0.6, 1.7)
        ax.set_xticks([0.9, 1.1])
        ax.set_xticklabels(["Hybrid\n(ours)", "DeiT-Tiny"])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3, ls=":")

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("BUSI (breast ultrasound), 3 seeds — no Mixup/CutMix, class-weighted CE",
                 y=1.13, fontsize=12.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.savefig(os.path.splitext(args.output)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved: {args.output}")

    for idx, label in [(0, "balanced accuracy"), (1, "macro-F1")]:
        hyb = [data["nca_vit_hybrid"][s][idx] for s in SEEDS]
        dei = [data["deit_tiny"][s][idx] for s in SEEDS]
        print(f"{label:20s} Hybrid {st.mean(hyb):.2f}+/-{st.stdev(hyb):.2f}  "
              f"DeiT {st.mean(dei):.2f}+/-{st.stdev(dei):.2f}  "
              f"gap(mean) {st.mean(hyb) - st.mean(dei):+.2f}  "
              f"overlap: {'YES' if min(hyb) < max(dei) else 'no'}")


if __name__ == "__main__":
    main()
