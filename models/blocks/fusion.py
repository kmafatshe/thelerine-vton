"""
Multi-scale feature fusion for ThelerineVTON.

Fuses person, garment and condition features at a
single resolution.
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

        # -------------------------------------------------
        # Individual projections
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Geometry gate
        # Uses condition features to determine where
        # garment information should flow.
        # -------------------------------------------------

        self.gate = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True,
            ),

            nn.Sigmoid(),
        )

        # -------------------------------------------------
        # Fusion
        # -------------------------------------------------

        self.fuse = ResidualBlock(
            channels * 3,
            channels,
        )

        # -------------------------------------------------
        # Global reasoning
        # -------------------------------------------------

        self.attention = MultiHeadSelfAttention(
            channels=channels,
            num_heads=num_heads,
        )

        # -------------------------------------------------
        # Refinement
        # -------------------------------------------------

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

        # ---------------------------------------------
        # Project into common feature space
        # ---------------------------------------------

        person = self.person_proj(person)

        garment = self.garment_proj(garment)

        condition = self.condition_proj(condition)

        # ---------------------------------------------
        # Geometry-guided garment gating
        # ---------------------------------------------

        gate = self.gate(condition)

        garment = garment * (1.0 * gate)

        # ---------------------------------------------
        # Fuse modalities
        # ---------------------------------------------

        x = torch.cat(
            (
                person,
                garment,
                condition,
            ),
            dim=1,
        )

        x = self.fuse(x)

        # ---------------------------------------------
        # Global reasoning
        # ---------------------------------------------

        x = self.attention(x)

        # ---------------------------------------------
        # Final refinement
        # ---------------------------------------------

        x = self.refine(x)

        return x