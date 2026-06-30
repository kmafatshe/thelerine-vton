"""
ThelerineVTON V2

Main model definition.

Author:
    Keneilwe Mokoka
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .feature_encoder import FeatureEncoder
from .feature_decoder import FeatureDecoder
from .output_head import ResidualOutputHead
from .blocks.fusion import MultiScaleFusion


class VTONGenerator(nn.Module):
    """
    ThelerineVTON V2.

    Inputs
    ------
    person : (B,3,H,W)

    garment : (B,3,H,W)

    condition : (B,7,H,W)
        DensePose + clothing mask.

    Returns
    -------
    output

    residual

    blend

    confidence
    """

    def __init__(
        self,
        base_channels: int = 64,
    ):

        super().__init__()

        # -------------------------------------------------
        # Encoders
        # -------------------------------------------------

        self.person_encoder = FeatureEncoder(
            in_channels=3,
            base_channels=base_channels,
        )

        self.garment_encoder = FeatureEncoder(
            in_channels=3,
            base_channels=base_channels,
        )

        self.condition_encoder = FeatureEncoder(
            in_channels=7,
            base_channels=base_channels,
        )

        # -------------------------------------------------
        # Multi-scale fusion
        # -------------------------------------------------

        self.fusion1 = MultiScaleFusion(
            channels=base_channels,
        )

        self.fusion2 = MultiScaleFusion(
            channels=base_channels * 2,
        )

        self.fusion3 = MultiScaleFusion(
            channels=base_channels * 4,
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.decoder = FeatureDecoder(
            base_channels=base_channels,
        )

        # -------------------------------------------------
        # Output head
        # -------------------------------------------------

        self.output_head = ResidualOutputHead(
            in_channels=base_channels,
        )

    def forward(
        self,
        person: torch.Tensor,
        garment: torch.Tensor,
        condition: torch.Tensor,
    ):

        # -------------------------------------------------
        # Encode person
        # -------------------------------------------------

        p1, p2, p3 = self.person_encoder(person)

        # -------------------------------------------------
        # Encode garment
        # -------------------------------------------------

        g1, g2, g3 = self.garment_encoder(garment)

        # -------------------------------------------------
        # Encode conditioning
        # -------------------------------------------------

        c1, c2, c3 = self.condition_encoder(condition)

        # -------------------------------------------------
        # Multi-scale fusion
        # -------------------------------------------------

        f1 = self.fusion1(
            p1,
            g1,
            c1,
        )

        f2 = self.fusion2(
            p2,
            g2,
            c2,
        )

        f3 = self.fusion3(
            p3,
            g3,
            c3,
        )

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------

        features = self.decoder(
            f1,
            f2,
            f3,
        )

        # -------------------------------------------------
        # Final image
        # -------------------------------------------------

        return self.output_head(
            features,
            person,
        )