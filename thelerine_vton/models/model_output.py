"""
Standard output container for ThelerineVTON.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """
    Standard output returned by VTONGenerator.
    """

    image: torch.Tensor
    residual: torch.Tensor
    blend: torch.Tensor
    confidence: torch.Tensor