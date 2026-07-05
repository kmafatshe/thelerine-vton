
"""
datasets/manifest.py

Manifest loading utilities for ThelerineVTON.
Supports both:
1) original reconstruction rows
2) swap-manifest rows
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManifestSample:
    # common
    sample_id: str
    garment: str

    # original reconstruction schema
    person: str | None = None
    cond: str | None = None
    seg: str | None = None

    # swap schema
    source_person: str | None = None
    source_cond: str | None = None
    source_seg: str | None = None
    target_person: str | None = None
    garment_type: str | None = None

    @classmethod
    def from_dict(cls, d):
        # --------------------------------------------------
        # Swap-manifest row
        # --------------------------------------------------
        if "source_person" in d:
            return cls(
                sample_id=d["id"],
                garment=d["garment"],
                source_person=d["source_person"],
                source_cond=d["source_cond"],
                source_seg=d["source_seg"],
                target_person=d["target_person"],
                garment_type=d.get("garment_type"),
            )

        # --------------------------------------------------
        # Original reconstruction row
        # --------------------------------------------------
        return cls(
            sample_id=d["id"],
            person=d["person"],
            cond=d["cond"],
            seg=d["seg"],
            garment=d["garment"],
        )


def load_manifest(path: str | Path):
    path = Path(path)

    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(ManifestSample.from_dict(row))

    return samples
