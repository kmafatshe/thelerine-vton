"""
Feature decoder for ThelerineVTON.

Progressively reconstructs high-resolution feature maps from
multi-scale fused encoder features.
"""

from __future__ import annotations

import torch.nn as nn

from .blocks.attention import MultiHeadSelfAttention
from .blocks.residual_block import ResidualBlock
from .blocks.upsample import Upsample


class FeatureDecoder(nn.Module):
    """
    Progressive decoder.

    Inputs
    ------
    f1 : (B, 64, 256, 256)

    f2 : (B,128,128,128)

    f3 : (B,256,64,64)

    Output
    ------
    (B,64,256,256)
    """

    def __init__(
        self,
        base_channels: int = 64,
    ):

        super().__init__()

        # -------------------------------------------------
        # Stage 3 -> Stage 2
        # -------------------------------------------------

        self.up2 = Upsample(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.attn2 = MultiHeadSelfAttention(
            channels=base_channels * 2,
            num_heads=8,
        )

        # -------------------------------------------------
        # Stage 2 -> Stage 1
        # -------------------------------------------------

        self.up1 = Upsample(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        self.attn1 = MultiHeadSelfAttention(
            channels=base_channels,
            num_heads=8,
        )

        # -------------------------------------------------
        # Final refinement
        # -------------------------------------------------

        self.refine = nn.Sequential(

            ResidualBlock(
                base_channels,
                base_channels,
            ),

            ResidualBlock(
                base_channels,
                base_channels,
            ),
        )

        self.final_attention = MultiHeadSelfAttention(
            channels=base_channels,
            num_heads=8,
        )

    def forward(
        self,
        f1,
        f2,
        f3,
    ):

        # ---------------------------------------------
        # 64x64 -> 128x128
        # ---------------------------------------------

        x = self.up2(
            f3,
            f2,
        )

        x = self.attn2(x)

        # ---------------------------------------------
        # 128x128 -> 256x256
        # ---------------------------------------------

        x = self.up1(
            x,
            f1,
        )

        x = self.attn1(x)

        # ---------------------------------------------
        # Final refinement
        # ---------------------------------------------

        x = self.refine(x)

        x = self.final_attention(x)

        return x