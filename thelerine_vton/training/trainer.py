"""
Trainer for ThelerineVTON.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device,
        scheduler=None,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip

        self.use_amp = (
            mixed_precision
            and torch.cuda.is_available()
        )

        self.scaler = GradScaler(
            enabled=self.use_amp
        )

    def train_epoch(self) -> Dict[str, float]:

        self.model.train()

        running = {
            "total": 0.0,
            "image": 0.0,
            "perceptual": 0.0,
            "edge": 0.0,
            "garment": 0.0,
        }

        progress = tqdm(
            self.train_loader,
            leave=False,
        )

        for batch in progress:

            person = batch.person.to(self.device)

            garment = batch.garment.to(self.device)

            condition = batch.condition.to(self.device)

            target = batch.target.to(self.device)

            garment_mask = (

                batch.garment_mask
                .unsqueeze(1)
                .to(self.device)

            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                enabled=self.use_amp
            ):

                prediction = self.model(

                    person,

                    garment,

                    condition,

                )

                losses = self.loss_fn(

                    prediction.image,

                    target,

                    garment_mask,

                )

            self.scaler.scale(
                losses["total"]
            ).backward()

            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.grad_clip,

            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            for key in running:

                running[key] += losses[key].item()

            progress.set_postfix(

                total=f"{losses['total'].item():.4f}",

                image=f"{losses['image'].item():.4f}",

            )

        n = len(self.train_loader)

        if self.scheduler is not None:

            self.scheduler.step()

        return {

            key: value / n

            for key, value in running.items()

        }
    
    def validate_epoch(self):

        if self.val_loader is None:
            return None

        self.model.eval()

        running = {
            "total": 0.0,
            "image": 0.0,
            "perceptual": 0.0,
            "edge": 0.0,
            "garment": 0.0,
        }

        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                leave=False,
                desc="Validation",
            )

            for batch in progress:

                person = batch.person.to(self.device)

                garment = batch.garment.to(self.device)

                condition = batch.condition.to(self.device)

                target = batch.target.to(self.device)

                garment_mask = (
                    batch.garment_mask
                    .unsqueeze(1)
                    .to(self.device)
                )

                with autocast(enabled=self.use_amp):

                    prediction = self.model(

                        person,

                        garment,

                        condition,

                    )

                    losses = self.loss_fn(

                        prediction.image,

                        target,

                        garment_mask,

                    )

                for key in running:

                    running[key] += losses[key].item()

        n = len(self.val_loader)

        return {

            key: value / n

            for key, value in running.items()

        }