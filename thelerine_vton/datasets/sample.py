"""
Standard sample returned by ThelerineVTON datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Sample:
    """
    Standard dataset sample.
    """

    person: torch.Tensor

    garment: torch.Tensor

    condition: torch.Tensor

    target: torch.Tensor

    garment_mask: torch.Tensor