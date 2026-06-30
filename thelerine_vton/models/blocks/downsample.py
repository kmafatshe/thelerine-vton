"""
Downsampling block.
"""

from __future__ import annotations

import torch.nn as nn

from .residual_block import ResidualBlock


class Downsample(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(2),

            ResidualBlock(
                in_channels,
                out_channels,
            ),
        )

    def forward(self, x):

        return self.block(x)