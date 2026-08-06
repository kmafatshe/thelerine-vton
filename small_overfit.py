#!/usr/bin/env python3
"""Small-data VTON overfit demo.

This script is intended for tiny datasets where you want a working
training pipeline with minimal setup.

Dataset layout:
    root/
      person/
      garments/
      cond/   (optional)
      seg/    (optional)

If `cond/` or `seg/` are missing, the script will use valid dummy
conditioning so the model can still overfit the person+garment pair.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.preprocessing.agnostic import (
    build_clothing_mask,
    make_agnostic,
)
from thelerine_vton.training.losses import TotalLoss
from thelerine_vton.training.trainer import Trainer


ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SmallTripletDataset(Dataset):

    def __init__(
        self,
        root: Path,
        image_size: int = 256,
        person_dir: str = "person",
        garment_dir: str = "garments",
        cond_dir: str | None = None,
        seg_dir: str | None = None,
    ):

        self.root = root
        self.image_size = image_size
        self.person_dir = root / person_dir
        self.garment_dir = root / garment_dir
        self.cond_dir = root / cond_dir if cond_dir is not None else None
        self.seg_dir = root / seg_dir if seg_dir is not None else None

        self.person_paths = self._collect_paths(self.person_dir)
        self.garment_paths = self._collect_paths(self.garment_dir)

        if len(self.person_paths) == 0:
            raise ValueError(f"No person images found in {self.person_dir}")
        if len(self.garment_paths) == 0:
            raise ValueError(f"No garment images found in {self.garment_dir}")

        self.garment_map = {
            p.stem: p for p in self.garment_paths
        }

        self.cond_map = self._build_path_map(self.cond_dir)
        self.seg_map = self._build_path_map(self.seg_dir)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def _collect_paths(self, directory: Path) -> List[Path]:
        if directory is None:
            return []

        if not directory.exists():
            return []

        return sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT
        )

    def _build_path_map(self, directory: Path | None) -> dict[str, Path]:
        if directory is None or not directory.exists():
            return {}

        return {
            p.stem: p
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".npy"}
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        return self.transform(Image.open(path).convert("RGB"))

    def _load_numpy(self, path: Path, is_segmentation: bool) -> torch.Tensor:
        array = np.load(path)

        if array.ndim == 2:
            array = array[None]
        elif array.ndim == 3 and array.shape[-1] <= 10:
            array = np.transpose(array, (2, 0, 1))

        tensor = torch.from_numpy(array).float()

        mode = "nearest" if is_segmentation else "bilinear"
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        ).squeeze(0)

        return tensor

    def _find_garment_path(self, person_path: Path, index: int) -> Path:
        if person_path.stem in self.garment_map:
            return self.garment_map[person_path.stem]

        return self.garment_paths[index % len(self.garment_paths)]

    def __len__(self) -> int:
        return len(self.person_paths)

    def __getitem__(self, index: int) -> dict:
        person_path = self.person_paths[index]
        garment_path = self._find_garment_path(person_path, index)

        person = self._load_image(person_path)
        garment = self._load_image(garment_path)

        cond_path = self.cond_map.get(person_path.stem)
        seg_path = self.seg_map.get(person_path.stem)

        if cond_path is not None:
            cond = self._load_numpy(cond_path, is_segmentation=False)
            if cond.ndim == 2:
                cond = cond.unsqueeze(0)
            if cond.shape[0] == 1:
                cond = cond.repeat(6, 1, 1)
            elif cond.shape[0] == 3:
                cond = cond.repeat(2, 1, 1)
            elif cond.shape[0] > 6:
                cond = cond[:6]
        else:
            cond = torch.zeros(6, self.image_size, self.image_size)

        if seg_path is not None:
            seg = self._load_numpy(seg_path, is_segmentation=True)
            if seg.ndim == 3 and seg.shape[0] != 1:
                seg = seg[0:1]
        else:
            seg = torch.zeros(1, self.image_size, self.image_size)

        garment_mask = build_clothing_mask(seg)
        if garment_mask.ndim == 4 and garment_mask.shape[0] == 1:
            garment_mask = garment_mask.squeeze(0)

        person_agnostic = make_agnostic(
            person.unsqueeze(0),
            seg.unsqueeze(0),
        ).squeeze(0)

        condition = torch.cat([cond, garment_mask], dim=0).contiguous()

        return {
            "person": person_agnostic,
            "garment": garment,
            "condition": condition,
            "target": person,
            "garment_mask": garment_mask,
        }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def save_sample_output(
    model: torch.nn.Module,
    batch: dict,
    path: Path,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        person = batch["person"].to(device)
        garment = batch["garment"].to(device)
        condition = batch["condition"].to(device)
        prediction = model(person, garment, condition)

    save_image(
        prediction.image,
        path,
        normalize=True,
        value_range=(-1, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small data VTON overfit training")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--person-dir", default="person")
    parser.add_argument("--garment-dir", default="garments")
    parser.add_argument("--cond-dir", default=None)
    parser.add_argument("--seg-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    dataset = SmallTripletDataset(
        root=args.dataset_root,
        image_size=args.image_size,
        person_dir=args.person_dir,
        garment_dir=args.garment_dir,
        cond_dir=args.cond_dir,
        seg_dir=args.seg_dir,
    )

    loader = build_loader(dataset, batch_size=args.batch_size)

    model = VTONGenerator(base_channels=args.base_channels).to(device)
    loss_fn = TotalLoss(image_weight=1.0, perceptual_weight=0.3, edge_weight=0.2, garment_weight=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        val_loader=None,
        device=device,
        scheduler=None,
        grad_clip=1.0,
        mixed_precision=False,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        losses = trainer.train_epoch()
        print(f"  loss={losses['total']:.6f}")

        sample = dataset[0]
        save_sample_output(
            model,
            sample,
            output_dir / f"sample_epoch_{epoch:03d}.png",
            device,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": losses["total"],
            },
            output_dir / f"epoch_{epoch:03d}.pt",
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
