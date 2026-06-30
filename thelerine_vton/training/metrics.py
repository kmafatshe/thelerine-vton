"""
Evaluation metrics for ThelerineVTON.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class AverageMeter:
    """
    Tracks the running average of a metric.
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.sum = 0.0
        self.count = 0

    def update(
        self,
        value: float,
        n: int = 1,
    ):

        self.sum += value * n

        self.count += n

    @property
    def average(self):

        if self.count == 0:
            return 0.0

        return self.sum / self.count


class PSNR:
    """
    Peak Signal-to-Noise Ratio.
    """

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> float:

        mse = F.mse_loss(
            prediction,
            target,
        )

        if mse.item() == 0:

            return float("inf")

        return 20 * math.log10(
            2.0 / math.sqrt(mse.item())
        )


class SSIM:
    """
    Lightweight SSIM approximation.
    """

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> float:

        mu_x = prediction.mean()

        mu_y = target.mean()

        sigma_x = prediction.var()

        sigma_y = target.var()

        sigma_xy = (
            (prediction - mu_x)
            * (target - mu_y)
        ).mean()

        c1 = 0.01 ** 2

        c2 = 0.03 ** 2

        ssim = (

            (2 * mu_x * mu_y + c1)

            * (2 * sigma_xy + c2)

        ) / (

            (mu_x ** 2 + mu_y ** 2 + c1)

            * (sigma_x + sigma_y + c2)

        )

        return ssim.item()