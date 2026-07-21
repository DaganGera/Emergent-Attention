"""
Hybrid NCA-ViT — first `nca_depth` blocks use NCA-Attention,
remaining `attn_depth` blocks use standard Multi-Head Self-Attention.

Default split: 6 NCA + 6 MHSA = 12 total blocks.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

from src.models.nca_vit import NCAViTBlock, MLP


class AttentionBlock(nn.Module):
    """
    Standard Pre-Norm transformer block with Multi-Head Self-Attention.

    Structure (matches timm ViT convention):
        x = x + MHSA(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class HybridNCAViT(nn.Module):
    """
    Hybrid NCA + self-attention Vision Transformer.

    Args:
        img_size:         Input image size (square), default 224
        patch_size:       Patch size, default 16
        in_chans:         Input channels, default 3
        num_classes:      Number of output classes, default 100
        embed_dim:        Embedding dimension D, default 192
        nca_depth:        Number of NCA blocks (first N), default 6
        attn_depth:       Number of MHSA blocks (last M), default 6
        nca_steps:        NCA iterations K per NCA block, default 4
        filter_names:     Perception filters for NCA blocks
        nca_hidden_dim:   NCA update MLP hidden dim, default 384
        stochastic_rate:  NCA stochastic update probability, default 0.5
        mlp_ratio:        MLP hidden ratio, default 4.0
        num_heads:        Number of attention heads in MHSA blocks, default 3
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        nca_depth: int = 6,
        attn_depth: int = 6,
        nca_steps: int = 4,
        filter_names: list[str] | None = None,
        nca_hidden_dim: int = 384,
        stochastic_rate: float = 0.5,
        mlp_ratio: float = 4.0,
        num_heads: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        learnable_filters: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.nca_depth = nca_depth
        self.attn_depth = attn_depth

        assert img_size % patch_size == 0
        self.Hp = img_size // patch_size
        self.Wp = img_size // patch_size
        self.num_patches = self.Hp * self.Wp

        # Auto-adjust num_heads to be divisible into embed_dim
        while embed_dim % num_heads != 0:
            num_heads -= 1
        num_heads = max(num_heads, 1)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop_rate)

        # Linearly increasing drop path across all 12 blocks
        total_depth = nca_depth + attn_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # NCA blocks (first nca_depth)
        self.nca_blocks = nn.ModuleList([
            NCAViTBlock(
                dim=embed_dim,
                nca_steps=nca_steps,
                filter_names=filter_names,
                nca_hidden_dim=nca_hidden_dim,
                stochastic_rate=stochastic_rate,
                grid_size=(self.Hp, self.Wp),
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                learnable_filters=learnable_filters,
            )
            for i in range(nca_depth)
        ])

        # MHSA blocks (last attn_depth)
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[nca_depth + i],
            )
            for i in range(attn_depth)
        ])

        # Final norm and head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Preserve zero-init on NCA output layers (trunc_normal_ above overwrites them)
        for block in self.nca_blocks:
            nn.init.zeros_(block.nca_attn.linear2.weight)
            nn.init.zeros_(block.nca_attn.linear2.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = img.shape[0]

        # Patch embed + flatten
        x = self.patch_embed(img).flatten(2).transpose(1, 2)  # (B, N, D)

        # Prepend CLS + positional embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)    # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        # NCA blocks
        for block in self.nca_blocks:
            x = block(x)

        # MHSA blocks
        for block in self.attn_blocks:
            x = block(x)

        # GAP over patch tokens for classification
        # (CLS is excluded from NCA and MLP is token-wise → CLS is image-blind)
        x = self.norm(x)
        return self.head(x[:, 1:].mean(dim=1))
