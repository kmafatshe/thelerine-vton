"""
datasets/triplet_dataset.py

PyTorch dataset for ThelerineVTON V2.
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
from thelerine_vton.preprocessing.garment_alignment import (
    align_garment_to_body,
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
            manifest_path = self.root / "manifest.jsonl"

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

        # -----------------------------------------
        # Convert to CHW
        # -----------------------------------------

        if array.ndim == 2:
            # H,W -> 1,H,W
            array = array[None]

        elif array.ndim == 3:
            # If last dim looks like channel count, assume HWC -> CHW
            if array.shape[-1] <= 10:
                array = np.transpose(array, (2, 0, 1))
            # otherwise assume already CHW

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

    def __getitem__(self, index):

        sample = self.manifest[index]

        # -----------------------------------------
        # Raw inputs
        # -----------------------------------------

        target = self._load_image(sample.person)     # original person image
        garment = self._load_image(sample.garment)
        cond = self._load_numpy(
            sample.cond,
            is_segmentation=False,
        )
        seg = self._load_numpy(
            sample.seg,
            is_segmentation=True,
        )

        # -----------------------------------------
        # Derived training inputs
        # -----------------------------------------

        garment_mask = build_clothing_mask(seg)      # [1,H,W]
        person = make_agnostic(target, seg)          # agnostic person
        garment = align_garment_to_body(
            garment.unsqueeze(0),
            seg.unsqueeze(0),
            keep_background=True,
        ).squeeze(0)

        # condition = cond + garment_mask
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