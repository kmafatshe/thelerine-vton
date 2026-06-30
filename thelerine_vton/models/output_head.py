"""
Residual output head for ThelerineVTON.

Predicts:
    - Residual RGB
    - Blend mask
    - Confidence map
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .model_output import ModelOutput


class ResidualOutputHead(nn.Module):
    """
    Converts decoder features into the final try-on image.

    Outputs
    -------
    residual_rgb : (B,3,H,W)
        RGB edits predicted by the network.

    blend_mask : (B,1,H,W)
        Controls how much of the residual replaces the
        original person image.

    confidence : (B,1,H,W)
        Model confidence in the generated result.
    """

    def __init__(
        self,
        in_channels: int = 64,
    ):

        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(8, in_channels),

            nn.SiLU(inplace=True),
        )

        # -----------------------------------------
        # RGB Residual
        # -----------------------------------------

        self.rgb = nn.Conv2d(
            in_channels,
            3,
            kernel_size=1,
        )

        # -----------------------------------------
        # Blend Mask
        # -----------------------------------------

        self.blend = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
        )

        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        self.confidence = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        features: torch.Tensor,
        original_person: torch.Tensor,
    ):

        features = self.head(features)

        # -----------------------------------------
        # Predictions
        # -----------------------------------------

        residual = torch.tanh(
            self.rgb(features)
        )

        blend = torch.sigmoid(
            self.blend(features)
        )

        confidence = torch.sigmoid(
            self.confidence(features)
        )

        # -----------------------------------------
        # Residual editing
        # -----------------------------------------

        edited = original_person + residual

        edited = torch.clamp(
            edited,
            -1.0,
            1.0,
        )

        output = (

            blend * edited +

            (1.0 - blend) * original_person

        )

        output = torch.clamp(
            output,
            -1.0,
            1.0,
        )

        return ModelOutput(
            image=output,
            residual=residual,
            blend=blend,
            confidence=confidence,
        )