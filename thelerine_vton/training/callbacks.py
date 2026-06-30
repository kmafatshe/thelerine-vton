"""
Training callbacks for ThelerineVTON.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


class CallbackManager:
    """
    Handles checkpointing, TensorBoard logging
    and image visualization.
    """

    def __init__(

        self,

        output_dir: str | Path,

    ):

        self.output_dir = Path(output_dir)

        self.checkpoint_dir = (
            self.output_dir / "checkpoints"
        )

        self.image_dir = (
            self.output_dir / "images"
        )

        self.log_dir = (
            self.output_dir / "logs"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.writer = SummaryWriter(
            self.log_dir
        )

        self.best_loss = float("inf")

    # ----------------------------------------------------
    # TensorBoard
    # ----------------------------------------------------

    def log_losses(

        self,

        epoch: int,

        losses: dict,

    ):

        for name, value in losses.items():

            self.writer.add_scalar(

                f"Loss/{name}",

                value,

                epoch,

            )

    # ----------------------------------------------------
    # Save checkpoint
    # ----------------------------------------------------

    def save_checkpoint(

        self,

        epoch: int,

        model,

        optimizer,

        loss: float,

    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss": loss,

        }

        latest = (
            self.checkpoint_dir /
            "latest.pt"
        )

        torch.save(
            checkpoint,
            latest,
        )

        if loss < self.best_loss:

            self.best_loss = loss

            best = (
                self.checkpoint_dir /
                "best.pt"
            )

            torch.save(
                checkpoint,
                best,
            )

    # ----------------------------------------------------
    # Save sample images
    # ----------------------------------------------------

    def save_images(

        self,

        epoch: int,

        person,

        garment,

        prediction,

        target,

    ):

        grid = make_grid(

            [

                person.cpu(),

                garment.cpu(),

                prediction.cpu(),

                target.cpu(),

            ],

            nrow=4,

            normalize=True,

            value_range=(-1, 1),

        )

        save_image(

            grid,

            self.image_dir /
            f"epoch_{epoch:04d}.png",

        )

    # ----------------------------------------------------
    # Close TensorBoard
    # ----------------------------------------------------

    def close(self):

        self.writer.close()