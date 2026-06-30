"""
Feature encoder for ThelerineVTON.

Encodes an image into multi-scale feature maps.
"""

from __future__ import annotations

import torch.nn as nn

from .blocks.attention import MultiHeadSelfAttention
from .blocks.downsample import Downsample
from .blocks.residual_block import ResidualBlock


class FeatureEncoder(nn.Module):
    """
    Generic encoder used for:

    - Person
    - Garment
    - Condition

    Returns
    -------
    stage1 : (B,64,H,W)

    stage2 : (B,128,H/2,W/2)

    stage3 : (B,256,H/4,W/4)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        attention: bool = True,
    ):

        super().__init__()

        # ---------------------------------------
        # Stage 1
        # 256x256
        # ---------------------------------------

        self.stage1 = ResidualBlock(
            in_channels,
            base_channels,
        )

        # ---------------------------------------
        # Stage 2
        # 128x128
        # ---------------------------------------

        self.stage2 = Downsample(
            base_channels,
            base_channels * 2,
        )

        # ---------------------------------------
        # Stage 3
        # 64x64
        # ---------------------------------------

        self.stage3 = Downsample(
            base_channels * 2,
            base_channels * 4,
        )

        # ---------------------------------------
        # Bottleneck Attention
        # ---------------------------------------

        if attention:

            self.attention = MultiHeadSelfAttention(
                base_channels * 4,
                num_heads=8,
            )

        else:

            self.attention = nn.Identity()

    def forward(self, x):

        # -------------------------
        # Stage 1
        # -------------------------

        s1 = self.stage1(x)

        # -------------------------
        # Stage 2
        # -------------------------

        s2 = self.stage2(s1)

        # -------------------------
        # Stage 3
        # -------------------------

        s3 = self.stage3(s2)

        # -------------------------
        # Attention
        # -------------------------

        s3 = self.attention(s3)

        return (
            s1,
            s2,
            s3,
        )