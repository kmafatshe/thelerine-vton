"""
Garment alignment utilities for ThelerineVTON.

This provides a stable baseline alignment strategy for the
current reconstruction-style training setup.

It does NOT do geometric warping. Instead, it optionally
masks the garment using the clothing region from the person's
segmentation so the network receives a cleaner garment input.
"""

from __future__ import annotations

from typing import Iterable

import torch

from thelerine_vton.preprocessing.agnostic import (
    DEFAULT_CLOTHING_LABELS,
    build_clothing_mask,
)


def align_garment_to_body(
    garment: torch.Tensor,
    seg: torch.Tensor,
    clothing_labels: Iterable[int] = DEFAULT_CLOTHING_LABELS,
    keep_background: bool = True,
) -> torch.Tensor:
    """
    Prepare garment input for the model.

    Args:
        garment: [B,3,H,W]
        seg: segmentation tensor from the same sample
        clothing_labels: labels defining clothing region
        keep_background:
            True  -> return garment unchanged
            False -> zero out garment outside clothing region

    Returns:
        aligned_garment: [B,3,H,W]
    """
    if garment.ndim != 4 or garment.shape[1] != 3:
        raise ValueError(
            f"garment must have shape [B,3,H,W], got {tuple(garment.shape)}"
        )

    if keep_background:
        return garment

    clothing_mask = build_clothing_mask(
        seg,
        clothing_labels=clothing_labels,
    )  # [B,1,H,W]

    if clothing_mask.shape[-2:] != garment.shape[-2:]:
        raise ValueError(
            "Segmentation and garment image spatial sizes do not match: "
            f"{tuple(clothing_mask.shape[-2:])} vs {tuple(garment.shape[-2:])}"
        )

    clothing_mask_3 = clothing_mask.expand(-1, 3, -1, -1)

    aligned_garment = garment * clothing_mask_3

    return aligned_garment