"""
Residual block for ThelerineVTON.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv_block import ConvBlock


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
        )

        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
        )

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )

        else:

            self.skip = nn.Identity()

    def forward(self, x):

        identity = self.skip(x)

        x = self.conv1(x)

        x = self.conv2(x)

        return x + identity