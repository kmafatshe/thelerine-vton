"""
Basic convolution block for ThelerineVTON.
"""

from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv → GroupNorm → SiLU

    Used throughout the network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 8,
    ):

        super().__init__()

        # -------------------------------------------------
        # Ensure GroupNorm always has a valid number of groups
        # -------------------------------------------------
        num_groups = min(groups, out_channels)

        while out_channels % num_groups != 0:
            num_groups -= 1

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups,
                out_channels,
            ),

            nn.SiLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)