
"""
Sample container and collate function for ThelerineVTON datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class Sample:
    person: torch.Tensor
    garment: torch.Tensor
    condition: torch.Tensor
    target: torch.Tensor
    garment_mask: torch.Tensor


def collate_samples(batch: list[Sample]) -> Sample:
    """
    Stack a list of Sample objects into a batched Sample.
    """
    return Sample(
        person=torch.stack([x.person for x in batch], dim=0),
        garment=torch.stack([x.garment for x in batch], dim=0),
        condition=torch.stack([x.condition for x in batch], dim=0),
        target=torch.stack([x.target for x in batch], dim=0),
        garment_mask=torch.stack([x.garment_mask for x in batch], dim=0),
    )
