"""
Learning rate scheduler factory for ThelerineVTON.
"""

from __future__ import annotations

import torch


def build_scheduler(
    optimizer,
    config: dict,
    steps_per_epoch: int,
):
    """
    Build a learning rate scheduler from the configuration.
    """

    name = config["scheduler"]["name"].lower()

    if name == "none":
        return None

    if name == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=config["training"]["epochs"],

            eta_min=config["scheduler"]["min_lr"],

        )

    if name == "onecycle":

        return torch.optim.lr_scheduler.OneCycleLR(

            optimizer,

            max_lr=config["optimizer"]["learning_rate"],

            epochs=config["training"]["epochs"],

            steps_per_epoch=steps_per_epoch,

        )

    raise ValueError(
        f"Unknown scheduler: {name}"
    )