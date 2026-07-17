"""
Train ThelerineVTON.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from thelerine_vton.datasets.triplet_dataset import TripletDataset
from thelerine_vton.datasets.sample import collate_samples
from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.training.callbacks import CallbackManager
from thelerine_vton.training.losses import TotalLoss
from thelerine_vton.training.trainer import Trainer
from thelerine_vton.training.scheduler import build_scheduler
from thelerine_vton.utils.config import load_config
from pathlib import Path
from thelerine_vton.utils.seed import seed_everything


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
    config = load_config("configs/train_personA_overfit.yaml")

    seed_everything(config["experiment"]["seed"])

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
        collate_fn=collate_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        collate_fn=collate_samples,
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
    # Resume / Fine-tune
    # -----------------------------------------------------
    output_dir = Path(config["experiment"]["output_dir"])
    resume_checkpoint = output_dir / "checkpoints" / "latest.pt"
    pretrained_checkpoint = Path(
        config.get("pretrained", {}).get("checkpoint", "")
    ) if config.get("pretrained") else None

    start_epoch = 1

    if resume_checkpoint.exists():
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    elif pretrained_checkpoint and pretrained_checkpoint.exists():
        print(f"\nInitializing from pretrained checkpoint: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Starting fine-tuning from epoch 1.")

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

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_losses = trainer.train_epoch()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning Rate: {current_lr:.8f}")

        if config["validation"]["enabled"]:
            val_losses = trainer.validate_epoch()
        else:
            val_losses = None

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