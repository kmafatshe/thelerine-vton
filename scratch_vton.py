#!/usr/bin/env python3
"""Standalone small-dataset VTON overfit + inference script.

This script is intentionally self-contained and built from scratch around
ThelerineVTON's model components. It trains on a tiny set of person and
garment images, optionally using provided segmentation/condition .npy files.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.preprocessing.agnostic import build_clothing_mask, make_agnostic
from thelerine_vton.training.losses import TotalLoss

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SmallVTONDataset(Dataset):

    def __init__(
        self,
        root: Path,
        image_size: int = 256,
        person_dir: str = "person",
        garment_dir: str = "garments",
        cond_dir: Optional[str] = None,
        seg_dir: Optional[str] = None,
    ):
        self.root = root
        self.image_size = image_size
        self.person_dir = root / person_dir
        self.garment_dir = root / garment_dir
        self.cond_dir = root / cond_dir if cond_dir is not None else None
        self.seg_dir = root / seg_dir if seg_dir is not None else None

        self.person_paths = self._collect_images(self.person_dir)
        self.garment_paths = self._collect_images(self.garment_dir)

        if len(self.person_paths) == 0:
            raise ValueError(f"No person images found in {self.person_dir}")
        if len(self.garment_paths) == 0:
            raise ValueError(f"No garment images found in {self.garment_dir}")

        self.garment_map = {p.stem: p for p in self.garment_paths}
        self.cond_map = self._build_npy_map(self.cond_dir)
        self.seg_map = self._build_npy_map(self.seg_dir)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _collect_images(self, directory: Path) -> List[Path]:
        if directory is None or not directory.exists():
            return []
        return sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXT
        )

    def _build_npy_map(self, directory: Optional[Path]) -> Dict[str, Path]:
        if directory is None or not directory.exists():
            return {}
        return {
            p.stem: p
            for p in sorted(directory.rglob("*.npy"))
            if p.is_file()
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def _load_npy(self, path: Path, is_segmentation: bool) -> torch.Tensor:
        array = np.load(path)
        if array.ndim == 2:
            array = array[None]
        elif array.ndim == 3 and array.shape[-1] <= 10:
            array = np.transpose(array, (2, 0, 1))

        tensor = torch.from_numpy(array).float()
        if is_segmentation and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _make_dummy_seg(self) -> torch.Tensor:
        return torch.ones(1, self.image_size, self.image_size)

    def _make_dummy_cond(self) -> torch.Tensor:
        return torch.zeros(6, self.image_size, self.image_size)

    def _find_garment(self, person_path: Path, index: int) -> Path:
        return self.garment_map.get(person_path.stem, self.garment_paths[index % len(self.garment_paths)])

    def __len__(self) -> int:
        return len(self.person_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        person_path = self.person_paths[index]
        garment_path = self._find_garment(person_path, index)

        person = self._load_image(person_path)
        garment = self._load_image(garment_path)

        cond_path = self.cond_map.get(person_path.stem)
        seg_path = self.seg_map.get(person_path.stem)

        if cond_path is not None:
            cond = self._load_npy(cond_path, is_segmentation=False)
            if cond.ndim == 2:
                cond = cond.unsqueeze(0)
            if cond.shape[0] == 1:
                cond = cond.repeat(6, 1, 1)
            elif cond.shape[0] == 3:
                cond = cond.repeat(2, 1, 1)
            if cond.shape[0] < 6:
                cond = torch.cat([cond, torch.zeros(6 - cond.shape[0], self.image_size, self.image_size)], dim=0)
            cond = cond[:6]
        else:
            cond = self._make_dummy_cond()

        if seg_path is not None:
            seg = self._load_npy(seg_path, is_segmentation=True)
            if seg.ndim == 3 and seg.shape[0] != 1:
                seg = seg[0:1]
        else:
            seg = self._make_dummy_seg()

        garment_mask = build_clothing_mask(seg)
        if garment_mask.ndim == 4 and garment_mask.shape[0] == 1:
            garment_mask = garment_mask.squeeze(0)

        person_agnostic = make_agnostic(person.unsqueeze(0), seg.unsqueeze(0)).squeeze(0)
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


def build_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running = {"total": 0.0, "image": 0.0, "perceptual": 0.0, "edge": 0.0, "garment": 0.0}
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        person = batch["person"].to(device)
        garment = batch["garment"].to(device)
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        garment_mask = batch["garment_mask"].to(device)

        prediction = model(person, garment, condition)
        losses = loss_fn(prediction.image, target, garment_mask)
        losses["total"].backward()
        optimizer.step()

        for key in running:
            running[key] += losses[key].item()

    return {k: v / len(loader) for k, v in running.items()}


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, epoch: int, loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def save_inference_output(
    model: torch.nn.Module,
    person_path: Path,
    garment_path: Path,
    output_path: Path,
    image_size: int,
    seg_path: Optional[Path] = None,
    cond_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
) -> None:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    person = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path).convert("RGB")
    person_tensor = transform(person).unsqueeze(0).to(device)
    garment_tensor = transform(garment).unsqueeze(0).to(device)

    if cond_path is not None:
        cond = np.load(cond_path)
        cond_tensor = torch.from_numpy(cond).float()
        if cond_tensor.ndim == 2:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim == 3 and cond_tensor.shape[-1] <= 10:
            cond_tensor = cond_tensor.permute(2, 0, 1)
        if cond_tensor.shape[0] == 1:
            cond_tensor = cond_tensor.repeat(6, 1, 1)
        elif cond_tensor.shape[0] == 3:
            cond_tensor = cond_tensor.repeat(2, 1, 1)
        cond_tensor = cond_tensor[:6]
        cond_tensor = cond_tensor.unsqueeze(0)
    else:
        cond_tensor = torch.zeros(1, 6, image_size, image_size)

    if seg_path is not None:
        seg = np.load(seg_path)
        seg_tensor = torch.from_numpy(seg).float()
        if seg_tensor.ndim == 2:
            seg_tensor = seg_tensor.unsqueeze(0)
        elif seg_tensor.ndim == 3 and seg_tensor.shape[-1] != seg_tensor.shape[0]:
            seg_tensor = seg_tensor.permute(2, 0, 1)
        seg_tensor = seg_tensor.unsqueeze(0)
    else:
        seg_tensor = torch.ones(1, 1, image_size, image_size)

    cond_tensor = F.interpolate(cond_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    seg_tensor = F.interpolate(seg_tensor, size=(image_size, image_size), mode="nearest")

    cond_tensor = cond_tensor.to(device)
    seg_tensor = seg_tensor.to(device)

    garment_mask = build_clothing_mask(seg_tensor)
    condition = torch.cat([cond_tensor.squeeze(0), garment_mask.squeeze(0)], dim=0).unsqueeze(0)
    person_agnostic = make_agnostic(person_tensor, seg_tensor)

    model.eval()
    with torch.no_grad():
        prediction = model(person_agnostic, garment_tensor, condition)

    save_image(prediction.image, output_path, normalize=True, value_range=(-1, 1))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, mask.shape[0], mask.shape[1]
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _expand_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    expanded = mask.copy()
    for _ in range(iterations):
        expanded = np.logical_or(expanded, np.roll(expanded, 1, axis=0))
        expanded = np.logical_or(expanded, np.roll(expanded, -1, axis=0))
        expanded = np.logical_or(expanded, np.roll(expanded, 1, axis=1))
        expanded = np.logical_or(expanded, np.roll(expanded, -1, axis=1))
    return expanded


def _foreground_mask_from_garment(garment_np: np.ndarray) -> np.ndarray:
    # Determine the garment region using a color/saturation heuristic.
    # This preserves the actual garment shape rather than the source person clothing.
    r = garment_np[..., 0]
    g = garment_np[..., 1]
    b = garment_np[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    saturation = np.zeros_like(maxc)
    nonzero = maxc > 0
    saturation[nonzero] = delta[nonzero] / maxc[nonzero]
    brightness = maxc

    mask = (brightness > 0.08) & (saturation > 0.10)
    mask = np.logical_or(mask, (saturation > 0.15) & (brightness > 0.05))

    if mask.sum() < 500:
        mask = (brightness > 0.05) & (saturation > 0.08)

    # Keep only the largest connected component.
    label = np.zeros_like(mask, dtype=np.int32)
    current_label = 0
    h, w = mask.shape
    largest_label = 0
    largest_size = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or label[y, x] != 0:
                continue
            current_label += 1
            queue = deque([(y, x)])
            label[y, x] = current_label
            size = 0
            while queue:
                yy, xx = queue.popleft()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and label[ny, nx] == 0:
                        label[ny, nx] = current_label
                        queue.append((ny, nx))
            if size > largest_size:
                largest_size = size
                largest_label = current_label

    cleaned = label == largest_label

    # Close small gaps in the shape.
    for _ in range(2):
        dilated = np.logical_or(cleaned, np.roll(cleaned, 1, axis=0))
        dilated = np.logical_or(dilated, np.roll(dilated, -1, axis=0))
        dilated = np.logical_or(dilated, np.roll(dilated, 1, axis=1))
        dilated = np.logical_or(dilated, np.roll(dilated, -1, axis=1))
        cleaned = np.logical_and(
            np.logical_and(dilated, np.roll(dilated, 1, axis=0)),
            np.logical_and(np.roll(dilated, -1, axis=0), np.logical_and(np.roll(dilated, 1, axis=1), np.roll(dilated, -1, axis=1))),
        )

    return cleaned.astype(np.bool_)


def _resize_image_np(image_np: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((image_np * 255).astype(np.uint8))
    resized = np.array(image.resize(size, resample=Image.LANCZOS)).astype(np.float32) / 255.0
    return resized


def _resize_mask_np(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = np.array(image.resize(size, resample=Image.NEAREST)).astype(np.uint8) > 0
    return resized


def overlay_person_garment(
    person_path: Path,
    garment_path: Path,
    output_path: Path,
    image_size: int = 256,
    seg_path: Optional[Path] = None,
) -> None:
    person = Image.open(person_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    garment = Image.open(garment_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)

    garment_np = np.array(garment).astype(np.float32) / 255.0
    garment_mask = _foreground_mask_from_garment(garment_np)

    person_np = np.array(person).astype(np.float32) / 255.0
    out_np = person_np.copy()

    if seg_path is not None and seg_path.exists():
        seg_arr = np.load(seg_path)
        seg_tensor = torch.from_numpy(seg_arr).float()
        if seg_tensor.ndim == 2:
            seg_tensor = seg_tensor.unsqueeze(0)
        elif seg_tensor.ndim == 3 and seg_tensor.shape[-1] == 1:
            seg_tensor = seg_tensor.permute(2, 0, 1)
        seg_tensor = seg_tensor.unsqueeze(0)
        seg_tensor = torch.nn.functional.interpolate(
            seg_tensor,
            size=(image_size, image_size),
            mode="nearest",
        )
        clothing_mask = build_clothing_mask(seg_tensor).squeeze(0).squeeze(0).bool().cpu().numpy()

        if clothing_mask.sum() > 0:
            removed = person_np.copy()

            # Fit the green dress to the clothing bbox from segmentation.
            gy1, gx1, gy2, gx2 = _bbox_from_mask(garment_mask)
            cy1, cx1, cy2, cx2 = _bbox_from_mask(clothing_mask)
            if gy1 < gy2 and gx1 < gx2 and cy1 < cy2 and cx1 < cx2:
                garment_crop = garment_np[gy1:gy2, gx1:gx2]
                garment_mask_crop = garment_mask[gy1:gy2, gx1:gx2]

                total_height = cy2 - cy1
                total_width = cx2 - cx1
                target_width = int(total_width * 1.4)
                target_height = int(total_height * 1.25)
                target_width = min(target_width, image_size)
                target_height = min(target_height, image_size)

                resized_garment = _resize_image_np(garment_crop, (target_width, target_height))
                resized_mask = _resize_mask_np(garment_mask_crop, (target_width, target_height))

                # Allow the green dress to occupy the full expanded clothing area.
                clothing_mask_expanded = _expand_mask(clothing_mask, iterations=4)
                clothing_region = clothing_mask_expanded[cy1:cy2, cx1:cx2]
                resized_mask = np.logical_or(resized_mask, clothing_region)

                final_mask_img = Image.fromarray((resized_mask.astype(np.uint8) * 255))
                final_mask_img = final_mask_img.filter(ImageFilter.GaussianBlur(radius=4))
                final_mask = np.array(final_mask_img).astype(np.float32) / 255.0
                final_mask = np.clip(final_mask, 0.0, 1.0)

                clothing_center_x = (cx1 + cx2) // 2
                place_x = max(0, min(image_size - target_width, clothing_center_x - target_width // 2))
                place_y = max(0, cy1 - int(total_height * 0.12))
                if place_y + target_height > image_size:
                    place_y = image_size - target_height

                crop_person = removed[place_y:place_y + target_height, place_x:place_x + target_width]
                crop_out = (
                    crop_person * (1.0 - final_mask[..., None])
                    + resized_garment * final_mask[..., None]
                )
                out_np = removed.copy()
                out_np[place_y:place_y + target_height, place_x:place_x + target_width] = crop_out
                out = torch.from_numpy(out_np).permute(2, 0, 1)
                save_image(out, output_path)
                return

    # Fallback: overlay garment shape directly when no valid segmentation or fit is available.
    out_np = person_np * (1.0 - garment_mask[..., None].astype(np.float32)) + garment_np * garment_mask[..., None].astype(np.float32)
    out = torch.from_numpy(out_np).permute(2, 0, 1)
    save_image(out, output_path)


def train(args: argparse.Namespace) -> None:
    device = get_device()
    dataset = SmallVTONDataset(
        root=args.dataset_root,
        image_size=args.image_size,
        person_dir=args.person_dir,
        garment_dir=args.garment_dir,
        cond_dir=args.cond_dir,
        seg_dir=args.seg_dir,
    )

    loader = build_dataloader(dataset, args.batch_size)
    model = VTONGenerator(base_channels=args.base_channels).to(device)
    loss_fn = TotalLoss(
        image_weight=args.image_weight,
        perceptual_weight=args.perceptual_weight,
        edge_weight=args.edge_weight,
        garment_weight=args.garment_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs} total={losses['total']:.6f} image={losses['image']:.6f}")

        save_model_checkpoint(
            model,
            optimizer,
            output_dir / f"epoch_{epoch:03d}.pt",
            epoch,
            losses["total"],
        )

        sample = dataset[0]
        save_inference_output(
            model,
            dataset.person_paths[0],
            dataset._find_garment(dataset.person_paths[0], 0),
            output_dir / f"epoch_{epoch:03d}_sample.png",
            args.image_size,
            seg_path=dataset.seg_map.get(dataset.person_paths[0].stem),
            cond_path=dataset.cond_map.get(dataset.person_paths[0].stem),
            device=device,
        )

    print("Training finished.")


def infer(args: argparse.Namespace) -> None:
    if args.overlay:
        overlay_person_garment(
            Path(args.person),
            Path(args.garment),
            Path(args.output),
            image_size=args.image_size,
            seg_path=Path(args.seg) if args.seg else None,
        )
        print(f"Saved overlay fallback output to {args.output}")
        return

    device = get_device()
    model = VTONGenerator(base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_inference_output(
        model,
        Path(args.person),
        Path(args.garment),
        Path(args.output),
        args.image_size,
        seg_path=Path(args.seg) if args.seg else None,
        cond_path=Path(args.cond) if args.cond else None,
        device=device,
    )
    print(f"Saved inference output to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scratch VTON small-data training and inference")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train a small VTON model")
    train_parser.add_argument("--dataset-root", required=True, type=Path)
    train_parser.add_argument("--output-dir", required=True, type=Path)
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--person-dir", default="person")
    train_parser.add_argument("--garment-dir", default="garments")
    train_parser.add_argument("--cond-dir", default=None)
    train_parser.add_argument("--seg-dir", default=None)
    train_parser.add_argument("--image-weight", type=float, default=1.0)
    train_parser.add_argument("--perceptual-weight", type=float, default=0.3)
    train_parser.add_argument("--edge-weight", type=float, default=0.2)
    train_parser.add_argument("--garment-weight", type=float, default=2.0)

    infer_parser = sub.add_parser("infer", help="Run inference on a person+garment pair")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--person", required=True)
    infer_parser.add_argument("--garment", required=True)
    infer_parser.add_argument("--output", default="output.png")
    infer_parser.add_argument("--image-size", type=int, default=256)
    infer_parser.add_argument("--base-channels", type=int, default=32)
    infer_parser.add_argument("--seg", default=None)
    infer_parser.add_argument("--cond", default=None)
    infer_parser.add_argument("--overlay", action="store_true", help="Use simple overlay fallback instead of the model")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)


if __name__ == "__main__":
    main()
