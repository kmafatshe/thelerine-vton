"""
Multi-scale feature fusion for ThelerineVTON.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .residual_block import ResidualBlock


class MultiScaleFusion(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ):

        super().__init__()

        # ------------------------------------
        # Compress each modality independently
        # ------------------------------------

        self.person_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.garment_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.condition_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # ------------------------------------
        # Fuse
        # ------------------------------------

        self.fuse = ResidualBlock(
            channels * 3,
            channels,
        )

        self.attention = MultiHeadSelfAttention(
            channels,
            num_heads=num_heads,
        )

        self.refine = ResidualBlock(
            channels,
            channels,
        )

    def forward(
        self,
        person,
        garment,
        condition,
    ):

        person = self.person_proj(person)

        garment = self.garment_proj(garment)

        condition = self.condition_proj(condition)

        x = torch.cat(
            [
                person,
                garment,
                condition,
            ],
            dim=1,
        )

        x = self.fuse(x)

        x = self.attention(x)

        x = self.refine(x)

        return x