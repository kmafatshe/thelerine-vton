"""
Triplet dataset for ThelerineVTON.

Reads samples directly from manifest.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from thelerine_vton.datasets.manifest import (
    ConditionAsset,
    GarmentAsset,
    ImageAsset,
    ManifestRecord,
    SegmentationAsset,
)

from thelerine_vton.datasets.sample import Sample

from thelerine_vton.datasets.transforms import (
    ConditionTransform,
    ImageTransform,
    SegmentationTransform,
)


class TripletDataset(Dataset):

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_file: str | Path,
        image_size: int = 256,
    ):

        self.root = Path(dataset_root)

        self.records = []

        with open(manifest_file, "r") as f:

            for line in f:

                d = json.loads(line)

                record = ManifestRecord(

                    identity=d["identity"],

                    split=d["split"],

                    source=d["source"],

                    target=d["target"],

                    person=ImageAsset(**d["person"]),

                    target_person=ImageAsset(
                        **d["target_person"]
                    ),

                    condition=ConditionAsset(
                        **d["condition"]
                    ),

                    source_segmentation=SegmentationAsset(
                        **d["source_segmentation"]
                    ),

                    target_segmentation=SegmentationAsset(
                        **d["target_segmentation"]
                    ),

                    garments=[
                        GarmentAsset(**g)
                        for g in d["garments"]
                    ],

                )

                self.records.append(record)

        self.image_tf = ImageTransform(image_size)

        self.cond_tf = ConditionTransform(image_size)

        self.seg_tf = SegmentationTransform(image_size)

    def __len__(self):

        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        # -------------------------------------------------
        # Load images
        # -------------------------------------------------

        person = Image.open(
            self.root / record.person.path
        ).convert("RGB")

        target = Image.open(
            self.root / record.target_person.path
        ).convert("RGB")

        garment = Image.open(
            self.root / record.garments[0].path
        ).convert("RGB")

        # -------------------------------------------------
        # Load conditioning
        # -------------------------------------------------

        cond = np.load(
            self.root / record.condition.path
        )

        source_seg = np.load(
            self.root /
            record.source_segmentation.path
        )

        target_seg = np.load(
            self.root /
            record.target_segmentation.path
        )

        # -------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------

        person = self.image_tf(person)

        target = self.image_tf(target)

        garment = self.image_tf(garment)

        cond = self.cond_tf(
            torch.from_numpy(cond)
        )

        source_seg = self.seg_tf(
            torch.from_numpy(source_seg)
        )

        target_seg = self.seg_tf(
            torch.from_numpy(target_seg)
        )

        # -------------------------------------------------
        # Sanity check
        # -------------------------------------------------

        if cond.shape[0] != 6:

            raise ValueError(

                f"Expected DensePose with 6 channels, "
                f"got {cond.shape}"

            )

        # -------------------------------------------------
        # Binary garment masks
        # -------------------------------------------------

        source_garment_mask = extract_garment_mask(
            source_seg
        )

        target_garment_mask = extract_garment_mask(
            target_seg
        )

        # -------------------------------------------------
        # Build condition tensor
        # -------------------------------------------------

        condition = torch.cat(

            (

                cond,

                source_garment_mask.unsqueeze(0),

            ),

            dim=0,

        )

        # -------------------------------------------------
        # Return sample
        # -------------------------------------------------

        return Sample(

            person=person,

            garment=garment,

            condition=condition,

            target=target,

            garment_mask=target_garment_mask,

        )


def extract_garment_mask(
    seg_map: torch.Tensor,
) -> torch.Tensor:
    """
    Extract a binary garment mask from a CIHP segmentation map.

    CIHP clothing classes:

        5  Upper clothes
        6  Dress
        7  Coat
        9  Pants
        10 Jumpsuit
    """

    garment_classes = torch.tensor(

        [5, 6, 7, 9, 10],

        device=seg_map.device,

    )

    mask = (

        seg_map.unsqueeze(0)

        == garment_classes[:, None, None]

    )

    return mask.any(dim=0).float()