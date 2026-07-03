"""
datasets/manifest.py

Manifest utilities for ThelerineVTON V2.

Each training sample contains:

- person image
- conditioning input (.npy)
- segmentation (.npy)
- garment image

Example:

{
    "id": "20240127_153016",
    "person": "person/personA/20240127_153016_person.jpg",
    "cond": "cond/personA/20240127_153016_cond_input.npy",
    "seg": "seg/personA/20240127_153016_seg.npy",
    "garment": "garments/20240127_153016_dress.jpg"
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


# ------------------------------------------------------------
# Manifest Entry
# ------------------------------------------------------------

@dataclass
class ManifestEntry:
    """
    Represents one training sample.
    """

    sample_id: str

    person: str
    cond: str
    seg: str
    garment: str

    def to_dict(self):
        return {
            "id": self.sample_id,
            "person": self.person,
            "cond": self.cond,
            "seg": self.seg,
            "garment": self.garment,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            sample_id=d["id"],
            person=d["person"],
            cond=d["cond"],
            seg=d["seg"],
            garment=d["garment"],
        )


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

class Manifest:
    """
    Container for all samples.

    Provides:

    - save()
    - load()
    - iteration
    - indexing
    """

    def __init__(self, entries: List[ManifestEntry]):

        self.entries = entries

    def __len__(self):

        return len(self.entries)

    def __getitem__(self, idx):

        return self.entries[idx]

    def __iter__(self):

        return iter(self.entries)

    # --------------------------------------------------------

    def save(self, path: str | Path):

        path = Path(path)

        with open(path, "w") as f:

            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()))
                f.write("\n")

    # --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path):

        path = Path(path)

        entries = []

        with open(path, "r") as f:

            for line in f:

                if not line.strip():
                    continue

                obj = json.loads(line)

                entries.append(
                    ManifestEntry.from_dict(obj)
                )

        return cls(entries)


# ------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------

def load_manifest(path: str | Path) -> Manifest:

    return Manifest.load(path)