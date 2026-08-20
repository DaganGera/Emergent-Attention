"""
System workflow diagram: the end-to-end pipeline (input -> two-stage
processing -> decision -> explanation), used as the "architecture diagram"
in PATENT_SPECIFICATION.md and as the workflow figure for the demo website.

Deliberately generic/non-technical labelling (no filter names, no layer
counts, no framework-specific terms) -- this figure documents the *system*
(the pipeline of functional modules), not the internal model architecture
(that is figures/paper/architecture.png / scripts/plot_architecture.py).

Pure matplotlib (boxes + arrows), matching the publication style used by
every other figure in this repo (serif, 300dpi). No external diagram tool.

Usage:
    python scripts/plot_system_workflow.py
"""

import os
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

mpl.rcParams.update({
    "font.size": 10.5, "font.family": "serif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

STAGE1_COLOR = "#dd8452"
STAGE2_COLOR = "#4c72b0"
EXPLAIN_COLOR = "#5b9279"
REASON_COLOR = "#9c6bb0"
NEUTRAL = "#5c5c5c"
DARK = "#2c2c2c"


def box(ax, xy, w, h, text, color, fontsize=10, fontweight="normal", text_color="white"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor="none", zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight, zorder=4)


def arrow(ax, p0, p1, color=NEUTRAL, style="-|>", lw=1.5, connectionstyle=None):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=2,
    ))


def ref(ax, x, y, num, color=NEUTRAL):
    ax.text(x, y, f"({num})", fontsize=8, color=color, va="center", ha="left", style="italic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/paper/system_workflow.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.6, 12.8))
    ax.set_xlim(0, 10)
    ax.axis("off")

    cx, w = 5, 7.2
    y = 0.3

    box(ax, (cx - w / 2, y), w, 1.1, "Image acquisition\n(sensor / camera / scan input)", DARK)
    ref(ax, cx + w / 2 + 0.15, y + 0.55, "200")
    y += 1.5
    arrow(ax, (cx, y - 0.4), (cx, y))

    box(ax, (cx - w / 2, y), w, 1.1, "Preprocessing and standardisation module", NEUTRAL)
    ref(ax, cx + w / 2 + 0.15, y + 0.55, "202")
    y += 1.65

    # Processing engine outer dashed frame (encloses stage 1 + stage 2 only)
    frame_y0 = y - 0.3
    arrow(ax, (cx, y - 0.45), (cx, y - 0.05))

    box(ax, (cx - w / 2, y), w, 1.4,
        "Stage 1 — local consistency module\n(reviews each small region on its own terms)",
        STAGE1_COLOR)
    ref(ax, cx + w / 2 + 0.15, y + 0.7, "212")
    y += 1.75
    arrow(ax, (cx, y - 0.35), (cx, y), color=STAGE1_COLOR)

    box(ax, (cx - w / 2, y), w, 1.4,
        "Stage 2 — global context module\n(weighs every region against every other)",
        STAGE2_COLOR)
    ref(ax, cx + w / 2 + 0.15, y + 0.7, "214")

    frame_y1 = y + 1.4 + 0.3
    ax.add_patch(Rectangle((cx - w / 2 - 0.35, frame_y0), w + 0.7, frame_y1 - frame_y0,
                            fill=False, edgecolor=NEUTRAL, linestyle="--", linewidth=1.1, zorder=1))
    ax.text(cx - w / 2 - 0.2, frame_y1 + 0.35, "Two-stage processing engine (210)",
            fontsize=9.5, color=NEUTRAL, style="italic", va="bottom")

    y += 2.6
    arrow(ax, (cx, y - 0.4), (cx, y))

    box(ax, (cx - w / 2, y), w, 1.1, "Decision module\n(produces the classification output)", DARK)
    ref(ax, cx + w / 2 + 0.15, y + 0.55, "220")
    y += 1.6
    arrow(ax, (cx, y - 0.5), (cx, y))

    # Branch into explainability + reasoning
    branch_y = y
    box(ax, (cx - w / 2, y), w, 1.05, "Human oversight and reporting layer (250)",
        "#f2f2f2", text_color=DARK, fontsize=9.5, fontweight="bold")
    y += 1.4

    left_x = cx - w / 2 + 1.7
    right_x = cx + w / 2 - 1.7
    arrow(ax, (cx, branch_y), (left_x, y), color=EXPLAIN_COLOR,
          connectionstyle="arc3,rad=-0.2")
    arrow(ax, (cx, branch_y), (right_x, y), color=REASON_COLOR,
          connectionstyle="arc3,rad=0.2")

    bw, bh = 3.5, 1.5
    box(ax, (left_x - bw / 2, y), bw, bh,
        "Explanation module\nhighlights the regions that\ndrove the decision", EXPLAIN_COLOR, fontsize=9)
    ref(ax, left_x - bw / 2, y - 0.25, "230", color=EXPLAIN_COLOR)

    box(ax, (right_x - bw / 2, y), bw, bh,
        "Reasoning module\nturns the decision into a\nplain-language explanation", REASON_COLOR, fontsize=9)
    ref(ax, right_x - bw / 2, y - 0.25, "240", color=REASON_COLOR)

    y += bh + 0.5
    arrow(ax, (left_x, y - 0.5), (cx, y), color=EXPLAIN_COLOR, connectionstyle="arc3,rad=0.15")
    arrow(ax, (right_x, y - 0.5), (cx, y), color=REASON_COLOR, connectionstyle="arc3,rad=-0.15")

    box(ax, (cx - w / 2, y), w, 1.2,
        "Combined output: decision + visual explanation\n+ plain-language reasoning, shown to the user", DARK, fontsize=9.5)
    ref(ax, cx + w / 2 + 0.15, y + 0.6, "260")
    y += 1.2

    ax.set_ylim(0, y + 1.0)
    ax.set_title("System Workflow", fontsize=14, pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.savefig(os.path.splitext(args.output)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
