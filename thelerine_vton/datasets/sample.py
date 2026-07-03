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

    person:
        Agnostic person image fed to the model.

    garment:
        Garment image input.

    condition:
        Conditioning tensor, e.g. DensePose + garment mask.

    target:
        Original person image used as reconstruction target.

    garment_mask:
        Binary clothing-region mask used for garment-focused loss.
    """

    person: torch.Tensor
    garment: torch.Tensor
    condition: torch.Tensor
    target: torch.Tensor
    garment_mask: torch.Tensor