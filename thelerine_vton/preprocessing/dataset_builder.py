"""
Dataset builder for ThelerineVTON.

Builds a manifest.jsonl file describing every
training sample in the dataset.

Author:
    Keneilwe Mokoka
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Dict, List


class DatasetBuilder:

    def __init__(self, dataset_root: str):

        self.root = Path(dataset_root)

        self.person_dir = self.root / "person"
        self.cond_dir = self.root / "cond"
        self.seg_dir = self.root / "seg"
        self.garment_dir = self.root / "garments"

        self.records: List[dict] = []

    # ----------------------------------------------------
    # GARMENT LOOKUP
    # ----------------------------------------------------

    def build_garment_lookup(self) -> Dict[str, List[str]]:

        lookup = {}

        suffixes = [
            "_upper.jpg",
            "_lower.jpg",
            "_dress.jpg",
            "_pants.jpg",
            "_jumpsuit.jpg",
            "_set.jpg",
        ]

        for garment in self.garment_dir.glob("*.jpg"):

            stem = garment.name

            base = stem

            for suffix in suffixes:
                base = base.replace(suffix, "")

            lookup.setdefault(base, []).append(garment.name)

        return lookup

    # ----------------------------------------------------
    # IDENTITIES
    # ----------------------------------------------------

    def discover_identities(self):

        identities = {}

        for identity_dir in sorted(self.person_dir.iterdir()):

            if not identity_dir.is_dir():
                continue

            looks = []

            for image in identity_dir.glob("*_person.jpg"):

                look = image.stem.replace("_person", "")

                looks.append(look)

            if len(looks) >= 2:

                identities[identity_dir.name] = sorted(looks)

        return identities

    # ----------------------------------------------------
    # BUILD
    # ----------------------------------------------------

    def build(self):

        garment_lookup = self.build_garment_lookup()

        identities = self.discover_identities()

        for identity, looks in identities.items():

            for source, target in permutations(looks, 2):

                garments = garment_lookup.get(target, [])

                categories = []

                for g in garments:

                    if "_upper" in g:
                        categories.append("upper")

                    elif "_lower" in g:
                        categories.append("lower")

                    elif "_pants" in g:
                        categories.append("pants")

                    elif "_dress" in g:
                        categories.append("dress")

                    elif "_jumpsuit" in g:
                        categories.append("jumpsuit")

                    elif "_set" in g:
                        categories.append("set")

                person_path = (
                    self.person_dir
                    / identity
                    / f"{source}_person.jpg"
                )

                target_path = (
                    self.person_dir
                    / identity
                    / f"{target}_person.jpg"
                )

                record = {

                    "identity": identity,

                    "split": "train",

                    "source": source,

                    "target": target,

                    "person":
                        str(person_path.relative_to(self.root)),

                    "target_person":
                        str(target_path.relative_to(self.root)),

                    "condition":
                        str(
                            (
                                self.cond_dir
                                / identity
                                / f"{source}_cond_input.npy"
                            ).relative_to(self.root)
                        ),

                    "source_seg":
                        str(
                            (
                                self.seg_dir
                                / identity
                                / f"{source}_seg.npy"
                            ).relative_to(self.root)
                        ),

                    "target_seg":
                        str(
                            (
                                self.seg_dir
                                / identity
                                / f"{target}_seg.npy"
                            ).relative_to(self.root)
                        ),

                    "garments": garments,

                    "categories": categories,

                    "num_garments": len(garments),

                }

                self.records.append(record)

        return self.records

    # ----------------------------------------------------
    # SAVE
    # ----------------------------------------------------

    def save(self, output_file: str):

        output = Path(output_file)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:

            for record in self.records:

                f.write(
                    json.dumps(record)
                    + "\n"
                )

        print(f"Saved {len(self.records)} records.")