"""
Inference preprocessing for ThelerineVTON.

Builds the 7-channel conditioning tensor from a
single person image.
"""

from __future__ import annotations

import torch
from PIL import Image


class InferencePreprocessor:
    """
    Placeholder inference preprocessor.

    The final implementation will integrate
    DensePose and human parsing.
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