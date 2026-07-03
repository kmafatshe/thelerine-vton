"""
Create clothing-agnostic person inputs for ThelerineVTON.
"""

from __future__ import annotations

from typing import Iterable

import torch


# ---------------------------------------------------------
# Default clothing labels for CIHP/LIP-style segmentations
# ---------------------------------------------------------
#
# Adjust these if your segmentation label IDs differ.
#
DEFAULT_CLOTHING_LABELS = (
    5,   # upper clothes
    6,   # dress
    7,   # coat / outerwear
    9,   # pants
    10,  # jumpsuit / full-body garment
    12,  # skirt
)


def _ensure_seg_shape(seg: torch.Tensor) -> torch.Tensor:
    """
    Ensure segmentation tensor is shaped [B,1,H,W].

    Accepts:
        [H,W]
        [1,H,W]
        [B,H,W]
        [B,1,H,W]
    """
    if seg.ndim == 2:
        seg = seg.unsqueeze(0).unsqueeze(0)      # [1,1,H,W]
    elif seg.ndim == 3:
        seg = seg.unsqueeze(1)                   # [B,1,H,W]
    elif seg.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Unsupported segmentation shape: {tuple(seg.shape)}"
        )

    return seg


def build_clothing_mask(
    seg: torch.Tensor,
    clothing_labels: Iterable[int] = DEFAULT_CLOTHING_LABELS,
) -> torch.Tensor:
    """
    Build a binary clothing mask from a segmentation label map.

    Args:
        seg: segmentation tensor
        clothing_labels: labels treated as clothing

    Returns:
        mask: float tensor of shape [B,1,H,W] with values in {0,1}
    """
    seg = _ensure_seg_shape(seg)
    seg_long = seg.long()

    mask = torch.zeros_like(seg_long, dtype=torch.bool)

    for label in clothing_labels:
        mask |= (seg_long == int(label))

    return mask.float()


def make_agnostic(
    person: torch.Tensor,
    seg: torch.Tensor,
    clothing_labels: Iterable[int] = DEFAULT_CLOTHING_LABELS,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Remove current-clothing pixels from the person image.

    Args:
        person: [B,3,H,W] normalized image tensor
        seg: segmentation tensor
        clothing_labels: labels considered clothing
        fill_value: value used to replace removed clothing pixels.
                    With image normalization to [-1,1], 0.0 is mid-gray.

    Returns:
        agnostic: [B,3,H,W]
    """
    if person.ndim != 4 or person.shape[1] != 3:
        raise ValueError(
            f"person must have shape [B,3,H,W], got {tuple(person.shape)}"
        )

    clothing_mask = build_clothing_mask(
        seg,
        clothing_labels=clothing_labels,
    )  # [B,1,H,W]

    if clothing_mask.shape[-2:] != person.shape[-2:]:
        raise ValueError(
            "Segmentation and person image spatial sizes do not match: "
            f"{tuple(clothing_mask.shape[-2:])} vs {tuple(person.shape[-2:])}"
        )

    clothing_mask_3 = clothing_mask.expand(-1, 3, -1, -1)

    agnostic = person.clone()
    agnostic = torch.where(
        clothing_mask_3.bool(),
        torch.full_like(agnostic, fill_value),
        agnostic,
    )

    return agnostic