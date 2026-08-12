"""
Architecture schematic for the paper: (a) the full Hybrid NCA-ViT pipeline,
(b) the internal K-iteration loop of one Emergent Attention (NCA) block.

Pure matplotlib (boxes + arrows), matching the publication style used by
every other figure in this repo (serif, 300dpi). No external diagram tool.

Usage:
    python scripts/plot_architecture.py
"""

import os
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.size": 10.5, "font.family": "serif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

NCA_COLOR = "#dd8452"
ATTN_COLOR = "#4c72b0"
NEUTRAL = "#5c5c5c"
LIGHT = "#f2f2f2"


def box(ax, xy, w, h, text, color, fontsize=9.5, fontweight="normal", text_color="white"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color, edgecolor="none", zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight, zorder=3)


def arrow(ax, p0, p1, color=NEUTRAL, style="-|>", lw=1.4, connectionstyle=None):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=1,
    ))


def panel_a(ax):
    """Full pipeline, bottom to top: image -> patch embed -> 6 NCA -> 6 MHSA -> head."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20.5)
    ax.axis("off")
    ax.set_title("(a)  Hybrid NCA-ViT pipeline", fontsize=12.5, pad=10)

    cx, w = 5, 6.6
    y = 0.3
    box(ax, (cx - w / 2, y), w, 1.0, "Image  (3 × 224 × 224)", NEUTRAL)
    y += 1.35
    arrow(ax, (cx, y - 0.35), (cx, y))
    box(ax, (cx - w / 2, y), w, 1.0, "Patch embed (Conv 16×16)  →  196 tokens × 192", NEUTRAL)
    y += 1.35
    arrow(ax, (cx, y - 0.35), (cx, y))
    box(ax, (cx - w / 2, y), w, 1.0, "+ CLS token,  + position embedding", NEUTRAL)
    y += 1.55
    arrow(ax, (cx, y - 0.55), (cx, y))

    # 6x NCA blocks (bracketed as one stage)
    stage_y0 = y
    box(ax, (cx - w / 2, y), w, 1.15, "Emergent Attention block\n(NCA, K=4 iterations)", NCA_COLOR)
    y += 1.15
    for _ in range(2):
        arrow(ax, (cx, y), (cx, y + 0.42), color=NCA_COLOR)
        y += 0.42
        box(ax, (cx - w / 2, y), w, 0.5, ".\n.\n.", NCA_COLOR, fontsize=9)
        y += 0.5
    arrow(ax, (cx, y), (cx, y + 0.35), color=NCA_COLOR)
    y += 0.35
    ax.text(cx + w / 2 + 0.25, (stage_y0 + y) / 2, "× 6\nlocal,\nearly", fontsize=8.5,
            color=NCA_COLOR, va="center", ha="left")

    y += 0.55
    arrow(ax, (cx, y - 0.55), (cx, y))

    # 6x MHSA blocks
    stage_y0 = y
    box(ax, (cx - w / 2, y), w, 1.15, "Standard MHSA block\n(3 heads, full attention)", ATTN_COLOR)
    y += 1.15
    for _ in range(2):
        arrow(ax, (cx, y), (cx, y + 0.42), color=ATTN_COLOR)
        y += 0.42
        box(ax, (cx - w / 2, y), w, 0.5, ".\n.\n.", ATTN_COLOR, fontsize=9)
        y += 0.5
    arrow(ax, (cx, y), (cx, y + 0.35), color=ATTN_COLOR)
    y += 0.35
    ax.text(cx + w / 2 + 0.25, (stage_y0 + y) / 2, "× 6\nglobal,\nlate", fontsize=8.5,
            color=ATTN_COLOR, va="center", ha="left")

    y += 0.55
    arrow(ax, (cx, y - 0.55), (cx, y))
    box(ax, (cx - w / 2, y), w, 1.0, "LayerNorm  →  GAP over patch tokens", NEUTRAL)
    y += 1.35
    arrow(ax, (cx, y - 0.35), (cx, y))
    box(ax, (cx - w / 2, y), w, 1.0, "Linear head  →  class logits", "#2c2c2c")

    ax.text(0.1, 20.1, "CLS token: excluded from NCA blocks (fixed delta = 0);\n"
                       "carried through unchanged, never read by the GAP head.",
            fontsize=7.3, color=NEUTRAL, style="italic", va="top")


def panel_b(ax):
    """One NCA block's internal K-iteration update rule."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20.5)
    ax.axis("off")
    ax.set_title("(b)  Emergent Attention: one NCA iteration", fontsize=12.5, pad=10)

    cx, w = 5, 7.4
    y = 17.6
    box(ax, (cx - w / 2, y), w, 1.0, r"cell state  $s_t \in \mathbb{R}^{D \times H_p \times W_p}$", "#2c2c2c")

    # Perception: 3 parallel fixed kernels
    y_top = y - 0.55
    y2 = y_top - 1.7
    kw = 2.1
    xs = [cx - 2.9, cx, cx + 2.9]
    labels = ["Sobel-X\n(fixed)", "Identity\n(fixed)", "Sobel-Y\n(fixed)"]
    for xk, lab in zip(xs, labels):
        arrow(ax, (cx, y_top), (xk, y2 + 1.0), color=NCA_COLOR,
              connectionstyle="arc3,rad=0.0" if xk == cx else ("arc3,rad=-0.15" if xk < cx else "arc3,rad=0.15"))
        box(ax, (xk - kw / 2, y2), kw, 1.0, lab, NCA_COLOR, fontsize=8.5)
    ax.text(cx, y2 - 0.35, "depthwise 3×3 conv, groups = D  →  perception  $(D{\\cdot}M,\\, H_p,\\, W_p)$",
            fontsize=8, ha="center", color=NEUTRAL)

    y3 = y2 - 1.1
    for xk in xs:
        arrow(ax, (xk, y2), (cx, y3 + 0.55), color=NCA_COLOR,
              connectionstyle="arc3,rad=0.0" if xk == cx else ("arc3,rad=0.15" if xk < cx else "arc3,rad=-0.15"))
    y3 -= 0.5
    box(ax, (cx - w / 2, y3), w, 0.85, "Linear → ReLU → Linear   (update MLP, shared across cells)",
        "#b0562f", fontsize=9)

    y4 = y3 - 1.15
    arrow(ax, (cx, y3), (cx, y3 - 0.3))
    box(ax, (cx - w / 2, y4), w, 0.85, r"$\delta_t$   ($W_2, b_2$ zero-init  $\Rightarrow$  $\delta_0{=}0$, identity at init)",
        "#b0562f", fontsize=8.7)

    y5 = y4 - 1.15
    arrow(ax, (cx, y4), (cx, y4 - 0.3))
    box(ax, (cx - w / 2, y5), w, 0.9, r"$s_{t+1} = s_t + m_t \odot \delta_t$,   $m_t \sim \mathrm{Bernoulli}(p{=}0.5)$  (train)",
        "#8c8c8c", fontsize=8.7)

    # loop-back arrow
    loop_x = cx + w / 2 + 0.55
    arrow(ax, (cx + w / 2, y5 + 0.45), (loop_x, y5 + 0.45), color=NEUTRAL, style="-")
    arrow(ax, (loop_x, y5 + 0.45), (loop_x, y + 0.5), color=NEUTRAL, style="-")
    arrow(ax, (loop_x, y + 0.5), (cx + w / 2, y + 0.5), color=NEUTRAL, style="-|>")
    ax.text(loop_x + 0.15, (y5 + y) / 2, "repeat\n$K{=}4$\ntimes", fontsize=8.5, color=NEUTRAL, va="center")

    y6 = y5 - 1.15
    arrow(ax, (cx, y5), (cx, y5 - 0.3))
    box(ax, (cx - w / 2, y6), w, 0.85, r"$\Delta = s_K - s_0$   (additive contribution to residual stream)",
        "#2c2c2c", fontsize=9)

    ax.text(0.1, 0.55,
            "Fixed, non-learned perception by default (learnable_filters=False).\n"
            "CLS token bypasses this entire block (delta forced to 0).",
            fontsize=7.3, color=NEUTRAL, style="italic", va="bottom")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/paper/architecture.png")
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 9.2))
    panel_a(ax1)
    panel_b(ax2)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=NCA_COLOR, markersize=11,
               label="NCA (Emergent Attention) — local, fixed perception"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=ATTN_COLOR, markersize=11,
               label="Standard MHSA — global"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
              fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.savefig(os.path.splitext(args.output)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
