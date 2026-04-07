"""
Perception Module for NCA-Attention.

Implements a depthwise grouped convolution that applies a set of fixed
(optionally learnable) filters — Sobel_x, Sobel_y, Identity, Laplacian —
to each channel independently.

Tensor shapes:
    Input:  (B, D, Hp, Wp)
    Output: (B, D*M, Hp, Wp)

where M = number of perception filters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed filter kernels (3×3), shape (1, 1, 3, 3)
FILTER_REGISTRY = {
    "sobel_x": torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ]).unsqueeze(0).unsqueeze(0),  # (1, 1, 3, 3)

    "sobel_y": torch.tensor([
        [-1., -2., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  2.,  1.],
    ]).unsqueeze(0).unsqueeze(0),

    "identity": torch.tensor([
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.],
    ]).unsqueeze(0).unsqueeze(0),

    "laplacian": torch.tensor([
        [ 0.,  1.,  0.],
        [ 1., -4.,  1.],
        [ 0.,  1.,  0.],
    ]).unsqueeze(0).unsqueeze(0),
}


class PerceptionModule(nn.Module):
    """
    Applies M fixed (or learnable) perception filters to a D-channel spatial grid
    via depthwise grouped convolution.

    Weight shape: (D*M, 1, 3, 3)  [groups=D, so each input channel gets M filters]

    Args:
        dim:              Number of channels D (e.g. 192 for NCA-ViT-Tiny)
        filter_names:     List of filter names from FILTER_REGISTRY (default: sobel_x, sobel_y, identity)
        learnable:        If True, initialise from fixed kernels but allow gradient updates.
                          If False, freeze the filters (register as buffer not parameter).
    """

    def __init__(
        self,
        dim: int,
        filter_names: list[str] | None = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        if filter_names is None:
            filter_names = ["sobel_x", "sobel_y", "identity"]

        self.dim = dim
        self.filter_names = filter_names
        self.num_filters = len(filter_names)  # M
        self.learnable = learnable

        # Build weight tensor of shape (D*M, 1, 3, 3)
        # Each filter is tiled D times (one per input channel)
        kernels = []
        for name in filter_names:
            k = FILTER_REGISTRY[name]  # (1, 1, 3, 3)
            k = k.repeat(dim, 1, 1, 1)  # (D, 1, 3, 3)
            kernels.append(k)
        weight = torch.cat(kernels, dim=0)  # (D*M, 1, 3, 3)

        if learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, Hp, Wp)
        Returns:
            out: (B, D*M, Hp, Wp)
        """
        # groups=D: each of the D input channels is convolved independently
        # with M filters, producing D*M output channels
        return F.conv2d(x, self.weight, bias=None, padding=1, groups=self.dim)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, filters={self.filter_names}, "
            f"num_filters={self.num_filters}, learnable={self.learnable}"
        )
