"""
Train ThelerineVTON.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from thelerine_vton.datasets.triplet_dataset import TripletDataset
from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.training.callbacks import CallbackManager
from thelerine_vton.training.losses import TotalLoss
from thelerine_vton.training.trainer import Trainer
from thelerine_vton.training.scheduler import build_scheduler
from thelerine_vton.utils.config import load_config
from thelerine_vton.utils.seed import set_seed


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def main():
    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------
    config = load_config("configs/train.yaml")

    set_seed(config["experiment"]["seed"])

    device = get_device()
    print(f"\nUsing device: {device}\n")

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    train_dataset = TripletDataset(
        dataset_root=config["dataset"]["root"],
        manifest_path=config["dataset"]["train_manifest"],
        image_size=config["dataset"]["image_size"],
    )

    val_dataset = TripletDataset(
        dataset_root=config["dataset"]["root"],
        manifest_path=config["dataset"]["val_manifest"],
        image_size=config["dataset"]["image_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = VTONGenerator(
        base_channels=config["model"]["base_channels"]
    ).to(device)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------
    loss_fn = TotalLoss(
        image_weight=config["loss"]["image_weight"],
        perceptual_weight=config["loss"]["perceptual_weight"],
        edge_weight=config["loss"]["edge_weight"],
        garment_weight=config["loss"]["garment_weight"],
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------
    scheduler = build_scheduler(
        optimizer,
        config,
        len(train_loader),
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        grad_clip=config["training"]["gradient_clip"],
        mixed_precision=config["training"]["mixed_precision"],
    )

    # -----------------------------------------------------
    # Callbacks
    # -----------------------------------------------------
    callbacks = CallbackManager(
        config["experiment"]["output_dir"]
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    epochs = config["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_losses = trainer.train_epoch()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning Rate: {current_lr:.8f}")

        val_losses = trainer.validate_epoch()

        callbacks.log_losses(
            epoch,
            train_losses,
        )

        if val_losses is not None:
            callbacks.log_losses(
                epoch,
                {
                    f"val_{k}": v
                    for k, v in val_losses.items()
                },
            )

        callbacks.save_checkpoint(
            epoch,
            model,
            optimizer,
            val_losses["total"]
            if val_losses is not None
            else train_losses["total"],
        )

        print(f"Loss: {train_losses['total']:.4f}")

    callbacks.close()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()