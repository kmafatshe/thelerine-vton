"""
Multi-Head Self-Attention for ThelerineVTON.

Applies global attention over a 2D feature map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ):

        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # -------------------------
        # Q, K and V projections
        # -------------------------

        self.q_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.k_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.v_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # -------------------------
        # Output projection
        # -------------------------

        self.out_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, x):

        B, C, H, W = x.shape

        N = H * W

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # -------------------------
        # Split into heads
        # -------------------------

        q = q.view(
            B,
            self.num_heads,
            self.head_dim,
            N
        ).permute(0,1,3,2)

        k = k.view(
            B,
            self.num_heads,
            self.head_dim,
            N
        ).permute(0,1,3,2)

        v = v.view(
            B,
            self.num_heads,
            self.head_dim,
            N
        ).permute(0,1,3,2)

        # -------------------------
        # Attention
        # -------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2,-1)
        ) * self.scale

        attention = scores.softmax(dim=-1)

        out = torch.matmul(
            attention,
            v
        )

        # -------------------------
        # Merge heads
        # -------------------------

        out = out.permute(
            0,
            1,
            3,
            2
        ).contiguous()

        out = out.view(
            B,
            C,
            H,
            W
        )

        out = self.out_proj(out)

        # -------------------------
        # Residual connection
        # -------------------------

        return x + out