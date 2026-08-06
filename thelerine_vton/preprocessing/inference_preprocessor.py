"""Inference preprocessing for ThelerineVTON."""

from __future__ import annotations

import torch
from PIL import Image


class InferencePreprocessor:
    """
    Placeholder inference preprocessor.

    The final implementation should integrate DensePose / parsing.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        person: Image.Image,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DensePose inference is not yet connected."
        )
