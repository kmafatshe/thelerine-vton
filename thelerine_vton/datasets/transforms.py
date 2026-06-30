"""
Image transforms for ThelerineVTON.

This module contains reusable transforms for:

- RGB images
- DensePose conditioning
- Segmentation maps
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F


class ImageTransform:
    """
    Transform RGB images into tensors.
    """

    def __init__(self, image_size: int = 256):

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class ConditionTransform:
    """
    Transform DensePose / conditioning tensors.
    """

    def __init__(self, image_size: int = 256):

        self.image_size = image_size

    def __call__(self, cond: torch.Tensor) -> torch.Tensor:

        if cond.ndim == 3 and cond.shape[0] not in (3, 6):
            cond = cond.permute(2, 0, 1)

        cond = cond.float()

        if cond.shape[1:] != (self.image_size, self.image_size):

            cond = F.interpolate(
                cond.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)

        return cond


class SegmentationTransform:
    """
    Transform segmentation maps.
    """

    def __init__(self, image_size: int = 256):

        self.image_size = image_size

    def __call__(self, seg: torch.Tensor) -> torch.Tensor:

        seg = seg.long()

        seg = F.interpolate(
            seg.unsqueeze(0).unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        return seg.squeeze(0).squeeze(0).long()