"""
Upsampling block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualBlock


class Upsample(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.residual = ResidualBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.residual(x)