#!/usr/bin/env python3
"""
Minimal Colab-friendly overfitting script for ThelerineVTON.

Usage in Colab:
    !pip install -q -r requirements.txt
    !python colab_overfit_demo.py --dataset-root /content/drive/MyDrive/thelerine_ai_data --manifest /content/drive/MyDrive/thelerine_ai_data/manifest.jsonl --output-dir /content/drive/MyDrive/thelerine_ai_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from thelerine_vton.datasets.triplet_dataset import TripletDataset
from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.training.losses import TotalLoss
from thelerine_vton.training.trainer import Trainer
from thelerine_vton.utils.seed import seed_everything


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader(dataset_root: Path, manifest_path: Path, image_size: int = 256, batch_size: int = 1):
    dataset = TripletDataset(
        dataset_root=str(dataset_root),
        manifest_path=str(manifest_path),
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def main():
    parser = argparse.ArgumentParser(description="Small overfitting run for ThelerineVTON")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    model = VTONGenerator(base_channels=64).to(device)
    loss_fn = TotalLoss(image_weight=1.0, perceptual_weight=0.3, edge_weight=0.2, garment_weight=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        scheduler=None,
        grad_clip=1.0,
        mixed_precision=torch.cuda.is_available(),
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = trainer.train_epoch()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": losses["total"],
            },
            output_dir / f"epoch_{epoch:03d}.pt",
        )

        print(f"Epoch {epoch}/{args.epochs} total_loss={losses['total']:.6f}")
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                output_dir / "best.pt",
            )

    print("Overfitting run complete.")


if __name__ == "__main__":
    main()
