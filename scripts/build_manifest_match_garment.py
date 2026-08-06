#!/usr/bin/env python3
"""
Build manifest tolerant to garment filename suffixes (e.g. _dress, _pants, _upper).

Scans `person/` (including nested `personA`/`personB`), finds garments in `garments/`
that contain the person base id, and locates matching `cond` and `seg` files (tries
in the same subfolder first, then globally).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def find_first(glob_iter):
    for p in glob_iter:
        if p.exists():
            return p
    return None


def find_garment(garment_dir: Path, base: str) -> Optional[Path]:
    # try exact-ish matches in root
    candidates = list(garment_dir.glob(f"{base}*")) + list(garment_dir.glob(f"*{base}*"))
    if candidates:
        return candidates[0]
    # try recursive search
    for p in garment_dir.rglob(f"{base}*"):
        return p
    return None


def find_cond_or_seg(base_dir: Path, subdir: Path, base: str, pattern: str):
    # look in subdir first
    try_paths = []
    if subdir:
        try_paths.extend(list((base_dir / subdir).glob(f"{base}*{pattern}*")))
    # then global
    try_paths.extend(list(base_dir.glob(f"{base}*{pattern}*")))
    if try_paths:
        return try_paths[0]
    # finally try recursive
    for p in base_dir.rglob(f"{base}*{pattern}*"):
        return p
    return None


def build_manifest(dataset_root: Path, output_path: Path, person_dir_name: str = "person", garment_dir_name: str = "garments", cond_dir_name: str = "cond", seg_dir_name: str = "seg"):
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()

    person_dir = dataset_root / person_dir_name
    garment_dir = dataset_root / garment_dir_name
    cond_dir = dataset_root / cond_dir_name
    seg_dir = dataset_root / seg_dir_name

    for d in (person_dir, garment_dir, cond_dir, seg_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing required directory: {d}")

    entries = []

    for person_path in sorted(person_dir.rglob("*")):
        if not person_path.is_file():
            continue
        if person_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        rel = person_path.relative_to(person_dir)
        subdir = rel.parent if len(rel.parts) > 1 else Path("")
        stem = person_path.stem

        # derive base id by removing trailing token like _person
        base = stem
        if base.endswith("_person"):
            base = base[: -len("_person")]

        # find garment
        garment_path = find_garment(garment_dir, base)

        # find cond and seg
        cond_path = find_cond_or_seg(cond_dir, subdir, base, "cond")
        seg_path = find_cond_or_seg(seg_dir, subdir, base, "seg")

        if garment_path is None:
            raise FileNotFoundError(f"Missing garment image for {stem}: searched for base '{base}' in {garment_dir}")
        if cond_path is None:
            raise FileNotFoundError(f"Missing condition tensor for {stem}: searched for base '{base}' in {cond_dir}")
        if seg_path is None:
            raise FileNotFoundError(f"Missing segmentation tensor for {stem}: searched for base '{base}' in {seg_dir}")

        entries.append({
            "id": f"{base}",
            "person": str(person_path.relative_to(dataset_root)),
            "cond": str(cond_path.relative_to(dataset_root)),
            "seg": str(seg_path.relative_to(dataset_root)),
            "garment": str(garment_path.relative_to(dataset_root)),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e))
            f.write("\n")

    print(f"Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build manifest matching garments by base ids")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--person-dir", default="person")
    parser.add_argument("--garment-dir", default="garments")
    parser.add_argument("--cond-dir", default="cond")
    parser.add_argument("--seg-dir", default="seg")
    args = parser.parse_args()

    build_manifest(
        args.dataset_root,
        args.output,
        person_dir_name=args.person_dir,
        garment_dir_name=args.garment_dir,
        cond_dir_name=args.cond_dir,
        seg_dir_name=args.seg_dir,
    )


if __name__ == "__main__":
    main()
