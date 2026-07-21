"""
Generate emergence visualizations:
  1. Receptive field growth heatmaps at NCA steps 1, 2, 4, 8, 16
  2. Effective attention maps for a test image

Usage:
    python scripts/visualize_emergence.py --checkpoint checkpoints/nca_vit_tiny/cifar100/seed42/best.pt
    python scripts/visualize_emergence.py --checkpoint <path> --animate  # saves GIF
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT
from src.evaluation.visualization import (
    receptive_field_growth,
    plot_receptive_field_growth,
    effective_attention_map,
    plot_effective_attention,
)
from src.utils.checkpoint import load_checkpoint


def save_emergence_animation(
    nca_block,
    embed_dim: int,
    grid_size: tuple[int, int],
    save_path: str,
    n_frames: int = 16,
    device: torch.device | None = None,
) -> None:
    """
    Save a GIF animating how activations spread over NCA steps.
    """
    if device is None:
        device = next(nca_block.parameters()).device

    Hp, Wp = grid_size
    nca_block.eval()

    grid = torch.zeros(1, embed_dim, Hp, Wp, device=device)
    cy, cx = Hp // 2, Wp // 2
    grid[0, :, cy, cx] = 1.0

    frames = []
    with torch.no_grad():
        for step in range(n_frames):
            heatmap = grid[0].abs().sum(dim=0).cpu().numpy()
            frames.append(heatmap)
            grid = nca_block._nca_step(grid)

    fig, ax = plt.subplots(figsize=(4, 4))
    vmax = max(f.max() for f in frames)
    im = ax.imshow(frames[0], cmap="hot", vmin=0, vmax=vmax)
    ax.axis("off")
    title = ax.set_title("NCA Step 0")

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        title.set_text(f"NCA Step {frame_idx}")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    anim.save(save_path, writer="pillow", fps=5)
    plt.close(fig)
    print(f"Animation saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model", default="nca_vit_hybrid",
                        choices=["nca_vit_tiny", "nca_vit_hybrid"],
                        help="Model type (default: nca_vit_hybrid)")
    parser.add_argument("--output-dir", default="figures", help="Where to save figures")
    parser.add_argument("--animate", action="store_true", help="Also produce emergence GIF")
    parser.add_argument("--block-idx", type=int, default=0,
                        help="Which NCA block to visualise (default: first block)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    if args.model == "nca_vit_hybrid":
        model = HybridNCAViT(
            num_classes=100, embed_dim=192, nca_depth=6, attn_depth=6,
            nca_steps=4, filter_names=["sobel_x", "sobel_y", "identity"],
            nca_hidden_dim=384, stochastic_rate=0.5, mlp_ratio=4.0,
            drop_rate=0.1, drop_path_rate=0.1, learnable_filters=True,
        ).to(device)
    else:
        model = NCAViT().to(device)

    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint}")

    # Get the requested NCA block — hybrid uses nca_blocks, pure NCA uses blocks
    if args.model == "nca_vit_hybrid":
        nca_block = model.nca_blocks[args.block_idx].nca_attn
    else:
        nca_block = model.blocks[args.block_idx].nca_attn
    embed_dim = model.embed_dim
    grid_size = (model.Hp, model.Wp)

    # 1. Receptive field growth
    print("Computing receptive field growth...")
    snapshots = receptive_field_growth(
        nca_block, embed_dim, grid_size,
        record_steps=[1, 2, 4, 8, 16], device=device,
    )
    fig = plot_receptive_field_growth(snapshots)
    path = os.path.join(args.output_dir, f"receptive_field_block{args.block_idx}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # 2. Emergence animation (optional)
    if args.animate:
        print("Generating emergence animation...")
        save_emergence_animation(
            nca_block, embed_dim, grid_size,
            save_path=os.path.join(args.output_dir, f"emergence_block{args.block_idx}.gif"),
            n_frames=16, device=device,
        )

    print("Done.")


if __name__ == "__main__":
    main()
