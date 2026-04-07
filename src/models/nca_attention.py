"""
NCA-Attention Block — drop-in replacement for Multi-Head Self-Attention.

Replaces O(n²) global attention with O(n·K) iterated local NCA message passing.

Forward tensor flow:
    Input:  x = (B, N+1, D)   where N = Hp*Wp (patch tokens) + 1 CLS token

    1. Separate CLS token:  cls  = x[:, :1, :]          → (B, 1, D)
                            patches = x[:, 1:, :]        → (B, N, D)
    2. Pre-norm:            p_norm = layer_norm(patches) → (B, N, D)
    3. Grid reshape:        grid = rearrange(p_norm, 'b (hp wp) d -> b d hp wp')
                                                         → (B, D, Hp, Wp)
    4. K NCA iterations:
        a. perception: perc = conv_groups(grid)          → (B, D*M, Hp, Wp)
        b. permute:    perc = (B, Hp, Wp, D*M)
        c. update MLP: h    = relu(linear1(perc))        → (B, Hp, Wp, H_mlp)
                       delta = linear2(h)                 → (B, Hp, Wp, D)
        d. permute:    delta = (B, D, Hp, Wp)
        e. stochastic: if training:
                           mask = Bernoulli(p) (B, 1, Hp, Wp)
                           grid = grid + mask * delta
                       else: grid = grid + delta
    5. Sequence reshape:    patches_out = rearrange(grid, 'b d hp wp -> b (hp wp) d')
                                                         → (B, N, D)
    6. Residual:            patches_out = patches_out + patches   (skip over pre-norm)
    7. Re-attach CLS:       out = cat([cls, patches_out], dim=1)  → (B, N+1, D)
"""

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.perception import PerceptionModule


class NCAAttention(nn.Module):
    """
    NCA-based local attention block.

    Args:
        dim:            Channel dimension D (e.g. 192)
        nca_steps:      Number of NCA iterations K (e.g. 4)
        filter_names:   Perception filter list (default: sobel_x, sobel_y, identity)
        hidden_dim:     Update MLP hidden dimension H_mlp (e.g. 384)
        stochastic_rate: Bernoulli drop probability during training (e.g. 0.5)
        grid_size:      Spatial grid (Hp, Wp) as a tuple, e.g. (14, 14)
        learnable_filters: Whether perception filters are learnable
    """

    def __init__(
        self,
        dim: int,
        nca_steps: int = 4,
        filter_names: list[str] | None = None,
        hidden_dim: int = 384,
        stochastic_rate: float = 0.5,
        grid_size: tuple[int, int] = (14, 14),
        learnable_filters: bool = False,
    ) -> None:
        super().__init__()

        if filter_names is None:
            filter_names = ["sobel_x", "sobel_y", "identity"]

        self.dim = dim
        self.nca_steps = nca_steps
        self.hidden_dim = hidden_dim
        self.stochastic_rate = stochastic_rate
        self.Hp, self.Wp = grid_size
        self.num_filters = len(filter_names)  # M

        # Pre-norm (applied to patch tokens before NCA)
        self.norm = nn.LayerNorm(dim)

        # Perception: depthwise conv giving (B, D*M, Hp, Wp)
        self.perception = PerceptionModule(
            dim=dim,
            filter_names=filter_names,
            learnable=learnable_filters,
        )

        # Update MLP: operates on perception output (D*M features per cell)
        self.linear1 = nn.Linear(dim * self.num_filters, hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, dim)

        # Zero-init output layer for training stability (delta ≈ 0 at init)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def _nca_step(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Single NCA update step.

        Args:
            grid: (B, D, Hp, Wp)
        Returns:
            updated grid: (B, D, Hp, Wp)
        """
        B = grid.shape[0]

        # Perception: (B, D, Hp, Wp) → (B, D*M, Hp, Wp)
        perc = self.perception(grid)

        # Rearrange for per-cell MLP: (B, D*M, Hp, Wp) → (B, Hp, Wp, D*M)
        perc = rearrange(perc, "b dm hp wp -> b hp wp dm")

        # Update MLP: (B, Hp, Wp, D*M) → (B, Hp, Wp, D)
        h = self.act(self.linear1(perc))
        delta = self.linear2(h)

        # Back to spatial: (B, Hp, Wp, D) → (B, D, Hp, Wp)
        delta = rearrange(delta, "b hp wp d -> b d hp wp")

        # Stochastic cell update (training only)
        if self.training and self.stochastic_rate < 1.0:
            # Mask shape (B, 1, Hp, Wp) — same mask for all channels of a cell
            mask = torch.bernoulli(
                torch.full((B, 1, self.Hp, self.Wp), self.stochastic_rate, device=grid.device)
            )
            grid = grid + mask * delta
        else:
            grid = grid + delta

        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, D) — patch tokens with prepended CLS token
        Returns:
            out: (B, N+1, D)
        """
        # 1. Separate CLS from patch tokens
        cls_token = x[:, :1, :]   # (B, 1, D) — passes through unchanged
        patches = x[:, 1:, :]     # (B, N, D)

        # 2. Pre-norm on patches only
        patches_norm = self.norm(patches)  # (B, N, D)

        # 3. Sequence → spatial grid
        grid = rearrange(patches_norm, "b (hp wp) d -> b d hp wp", hp=self.Hp, wp=self.Wp)
        # grid: (B, D, Hp, Wp)

        # 4. K NCA iterations
        for _ in range(self.nca_steps):
            grid = self._nca_step(grid)

        # 5. Spatial grid → sequence
        patches_out = rearrange(grid, "b d hp wp -> b (hp wp) d")
        # patches_out: (B, N, D)

        # 6. Residual connection (skip over pre-norm, per standard Pre-Norm convention)
        patches_out = patches_out + patches

        # 7. Re-attach CLS token
        out = torch.cat([cls_token, patches_out], dim=1)  # (B, N+1, D)
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, nca_steps={self.nca_steps}, "
            f"hidden_dim={self.hidden_dim}, stochastic_rate={self.stochastic_rate}, "
            f"grid_size=({self.Hp},{self.Wp})"
        )
