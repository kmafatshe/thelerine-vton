"""
Manifest data structures for ThelerineVTON.
"""

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class ImageAsset:
    path: str


@dataclass(slots=True)
class ConditionAsset:
    path: str
    channels: int


@dataclass(slots=True)
class SegmentationAsset:
    path: str
    classes: int


@dataclass(slots=True)
class GarmentAsset:
    category: str
    path: str


@dataclass(slots=True)
class ManifestRecord:

    identity: str

    split: str

    source: str

    target: str

    person: ImageAsset

    target_person: ImageAsset

    condition: ConditionAsset

    source_segmentation: SegmentationAsset

    target_segmentation: SegmentationAsset

    garments: List[GarmentAsset]