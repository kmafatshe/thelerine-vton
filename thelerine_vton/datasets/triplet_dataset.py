
"""
datasets/triplet_dataset.py

PyTorch dataset for ThelerineVTON V2 using swap-manifest training pairs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from .manifest import load_manifest
from .sample import Sample
from thelerine_vton.preprocessing.agnostic import (
    build_clothing_mask,
    make_agnostic,
)


class TripletDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        manifest_path=None,
        image_size=256,
    ):

        self.root = Path(dataset_root)

        if manifest_path is None:
            manifest_path = self.root / "swap_manifest.jsonl"

        self.manifest = load_manifest(manifest_path)
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=Image.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    # --------------------------------------------------------

    def __len__(self):
        return len(self.manifest)

    # --------------------------------------------------------

    def _check_exists(self, path: Path):

        if not path.exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------------

    def _load_image(self, relative_path):

        path = self.root / relative_path
        self._check_exists(path)

        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    # --------------------------------------------------------

    def _resize_tensor(
        self,
        tensor,
        mode="bilinear",
    ):

        tensor = tensor.unsqueeze(0)

        if mode == "nearest":
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="nearest",
            )
        else:
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return tensor.squeeze(0)

    # --------------------------------------------------------

    def _load_numpy(
        self,
        relative_path,
        is_segmentation=False,
    ):

        path = self.root / relative_path
        self._check_exists(path)

        array = np.load(path)

        if array.ndim == 2:
            array = array[None]  # H,W -> 1,H,W

        elif array.ndim == 3:
            if array.shape[-1] <= 10:
                array = np.transpose(array, (2, 0, 1))

        else:
            raise ValueError(
                f"Unsupported array shape {array.shape}"
            )

        tensor = torch.from_numpy(array).float()

        if is_segmentation:
            tensor = self._resize_tensor(
                tensor,
                mode="nearest",
            )
        else:
            tensor = self._resize_tensor(
                tensor,
                mode="bilinear",
            )

        return tensor.contiguous()

    # --------------------------------------------------------

    def _normalize_condition(self, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.float()

        if cond.max() > 2.0:
            cond = cond / 255.0

        return cond

    # --------------------------------------------------------

    def _crop_garment_foreground(
        self,
        img_tensor: torch.Tensor,
        threshold: float = -0.95,
        margin: int = 4,
    ) -> torch.Tensor:
        """
        Crop away large black background around garment.

        img_tensor: [3,H,W] in [-1,1]
        threshold: pixels above this are considered foreground.
                   Since padded/black background is near -1, this works
                   for your extracted garment images.
        """
        # foreground if any channel is above threshold
        fg = (img_tensor > threshold).any(dim=0)   # [H,W]

        if fg.sum() == 0:
            return img_tensor

        ys, xs = torch.where(fg)
        y0 = max(0, ys.min().item() - margin)
        y1 = min(img_tensor.shape[1], ys.max().item() + 1 + margin)
        x0 = max(0, xs.min().item() - margin)
        x1 = min(img_tensor.shape[2], xs.max().item() + 1 + margin)

        cropped = img_tensor[:, y0:y1, x0:x1]
        return cropped.contiguous()

    # --------------------------------------------------------

    def _resize_garment_keep_aspect(
        self,
        img_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Crop garment foreground, then resize to fit inside a square canvas
        while preserving aspect ratio.

        Input:
            img_tensor: [3,H,W] in [-1,1]

        Output:
            [3,image_size,image_size] in [-1,1]
        """
        img_tensor = self._crop_garment_foreground(img_tensor)

        _, h, w = img_tensor.shape
        target_size = self.image_size

        if h == target_size and w == target_size:
            return img_tensor

        scale = target_size / float(max(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        img = img_tensor.unsqueeze(0)
        img = F.interpolate(
            img,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_h = target_size - new_h
        pad_w = target_size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img = F.pad(
            img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=-1.0,
        )

        return img.contiguous()

    # --------------------------------------------------------

    def __getitem__(self, index):

        sample = self.manifest[index]

        source = self._load_image(sample.source_person)
        target = self._load_image(sample.target_person)

        garment_raw = self._load_image(sample.garment)
        garment = self._resize_garment_keep_aspect(garment_raw)

        cond = self._load_numpy(
            sample.source_cond,
            is_segmentation=False,
        )
        cond = self._normalize_condition(cond)

        seg = self._load_numpy(
            sample.source_seg,
            is_segmentation=True,
        )

        garment_mask = build_clothing_mask(seg)

        if garment_mask.ndim == 4:
            garment_mask = garment_mask.squeeze(0)

        person = make_agnostic(
            source.unsqueeze(0),
            seg.unsqueeze(0),
        ).squeeze(0)

        condition = torch.cat(
            [cond, garment_mask],
            dim=0,
        ).contiguous()

        return Sample(
            person=person.contiguous(),
            garment=garment.contiguous(),
            condition=condition,
            target=target.contiguous(),
            garment_mask=garment_mask.contiguous(),
        )
