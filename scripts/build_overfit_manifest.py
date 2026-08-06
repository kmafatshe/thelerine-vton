#!/usr/bin/env python3
"""
Build a triplet manifest for a small overfitting dataset.

Expected folder layout inside your dataset root:

    dataset_root/
      person/
        sample_001.jpg
      garment/
        sample_001.jpg
      cond/
        sample_001.npy
      seg/
        sample_001.npy

This script writes a JSONL manifest compatible with
`thelerine_vton.datasets.triplet_dataset.TripletDataset`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(
    dataset_root: Path,
    output_path: Path,
    person_dir_name: str = "person",
    garment_dir_name: str = "garment",
    cond_dir_name: str = "cond",
    seg_dir_name: str = "seg",
):
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()

    person_dir = dataset_root / person_dir_name
    garment_dir = dataset_root / garment_dir_name
    cond_dir = dataset_root / cond_dir_name
    seg_dir = dataset_root / seg_dir_name

    required_dirs = [person_dir, garment_dir, cond_dir, seg_dir]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Missing required directory: {directory}")

    entries = []
    person_files = sorted(person_dir.rglob("*"))

    def resolve_path(base_dir: Path, candidate: Path, filename: str) -> Path:
        if candidate.exists():
            return candidate
        fallback = base_dir / filename
        return fallback

    for person_path in person_files:
        if not person_path.is_file():
            continue

        if person_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        rel_path = person_path.relative_to(person_dir)
        stem = person_path.stem
        garment_path = resolve_path(
            garment_dir,
            garment_dir / rel_path,
            person_path.name,
        )
        cond_path = resolve_path(
            cond_dir,
            cond_dir / rel_path.with_suffix(".npy"),
            f"{person_path.name}.npy",
        )
        seg_path = resolve_path(
            seg_dir,
            seg_dir / rel_path.with_suffix(".npy"),
            f"{person_path.name}.npy",
        )

        if not garment_path.exists():
            raise FileNotFoundError(f"Missing garment image for {stem}: {garment_path}")
        if not cond_path.exists():
            raise FileNotFoundError(f"Missing condition tensor for {stem}: {cond_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation tensor for {stem}: {seg_path}")

        entries.append(
            {
                "id": stem,
                "person": str(person_path.relative_to(dataset_root)),
                "cond": str(cond_path.relative_to(dataset_root)),
                "seg": str(seg_path.relative_to(dataset_root)),
                "garment": str(garment_path.relative_to(dataset_root)),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    print(f"Wrote {len(entries)} manifest entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build a small overfitting triplet dataset manifest")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Dataset root directory")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL manifest path")
    parser.add_argument("--person-dir", default="person", help="Folder containing person images")
    parser.add_argument("--garment-dir", default="garments", help="Folder containing garment images")
    parser.add_argument("--cond-dir", default="cond", help="Folder containing conditioning .npy files")
    parser.add_argument("--seg-dir", default="seg", help="Folder containing segmentation .npy files")
    args = parser.parse_args()

    build_manifest(
        dataset_root=args.dataset_root,
        output_path=args.output,
        person_dir_name=args.person_dir,
        garment_dir_name=args.garment_dir,
        cond_dir_name=args.cond_dir,
        seg_dir_name=args.seg_dir,
    )


if __name__ == "__main__":
    main()
