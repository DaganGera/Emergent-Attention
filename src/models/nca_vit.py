"""
NCA-ViT — Vision Transformer with NCA-Attention replacing self-attention.

Architecture (NCA-ViT-Tiny):
    Patch embed:  Conv2d(3, 192, 16, 16)  → (B, 192, 14, 14) → flatten → (B, 196, 192)
    CLS token:    prepend (B, 1, 192)     → (B, 197, 192)
    Pos embed:    learnable (1, 197, 192) added
    12 blocks:    each = NCAViTBlock(NCAAttention + MLP)
    Head:         LayerNorm → Linear(192, num_classes) on CLS token

Total params: ~7.2M
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.models.nca_attention import NCAAttention


class MLP(nn.Module):
    """Standard ViT feed-forward block: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class NCAViTBlock(nn.Module):
    """
    Single transformer block with NCA-Attention.

    Structure:
        x = x + NCAAttention(x)         (pre-norm is inside NCAAttention)
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        dim: int,
        nca_steps: int,
        filter_names: list[str] | None,
        nca_hidden_dim: int,
        stochastic_rate: float,
        grid_size: tuple[int, int],
        mlp_ratio: float = 4.0,
        learnable_filters: bool = False,
    ) -> None:
        super().__init__()
        self.nca_attn = NCAAttention(
            dim=dim,
            nca_steps=nca_steps,
            filter_names=filter_names,
            hidden_dim=nca_hidden_dim,
            stochastic_rate=stochastic_rate,
            grid_size=grid_size,
            learnable_filters=learnable_filters,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCAAttention applies pre-norm internally and adds residual
        x = self.nca_attn(x)
        # Standard MLP block with pre-norm and residual
        x = x + self.mlp(self.norm2(x))
        return x


class NCAViT(nn.Module):
    """
    NCA-based Vision Transformer.

    Args:
        img_size:         Input image size (assumes square), default 224
        patch_size:       Patch size, default 16 (gives 14×14 grid)
        in_chans:         Input channels, default 3
        num_classes:      Number of output classes, default 100
        embed_dim:        Token embedding dimension D, default 192
        depth:            Number of NCAViT blocks, default 12
        nca_steps:        NCA iterations K per block, default 4
        filter_names:     Perception filter list (None → sobel_x, sobel_y, identity)
        nca_hidden_dim:   Update MLP hidden dim, default 384
        stochastic_rate:  Bernoulli mask probability, default 0.5
        mlp_ratio:        MLP hidden/embed ratio, default 4.0
        cls_strategy:     How CLS is handled: "exclude" (default) | future: "broadcast" | "gap"
        learnable_filters: Whether perception filters are learnable
        drop_rate:        MLP dropout rate
        pos_drop_rate:    Positional embedding dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 12,
        nca_steps: int = 4,
        filter_names: list[str] | None = None,
        nca_hidden_dim: int = 384,
        stochastic_rate: float = 0.5,
        mlp_ratio: float = 4.0,
        cls_strategy: str = "exclude",
        learnable_filters: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        assert cls_strategy == "exclude", (
            "Only 'exclude' CLS strategy is implemented. "
            "'broadcast' and 'gap' are ablation variants (A6)."
        )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.cls_strategy = cls_strategy

        # Patch grid dimensions
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.Hp = img_size // patch_size  # 14 for 224/16
        self.Wp = img_size // patch_size
        self.num_patches = self.Hp * self.Wp  # 196

        # 1. Patch embedding: Conv2d projects each patch to embed_dim
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

        # 2. CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional embedding: 1 (CLS) + num_patches tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop_rate)

        # 4. Transformer blocks
        self.blocks = nn.ModuleList([
            NCAViTBlock(
                dim=embed_dim,
                nca_steps=nca_steps,
                filter_names=filter_names,
                nca_hidden_dim=nca_hidden_dim,
                stochastic_rate=stochastic_rate,
                grid_size=(self.Hp, self.Wp),
                mlp_ratio=mlp_ratio,
                learnable_filters=learnable_filters,
            )
            for _ in range(depth)
        ])

        # 5. Final norm + classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        # Positional and CLS embeddings: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Linear layers: truncated normal for weight, zeros for bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W)
        Returns:
            cls_features: (B, D)  — CLS token after all blocks + norm
        """
        B = img.shape[0]

        # Patch embedding: (B, C, H, W) → (B, D, Hp, Wp) → (B, N, D)
        x = self.patch_embed(img)            # (B, D, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)  where N = Hp*Wp

        # Prepend CLS token: (B, N, D) → (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)           # (B, N+1, D)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # NCA-ViT blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x[:, 0]  # CLS token: (B, D)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        return self.head(self.forward_features(img))

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
