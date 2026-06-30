"""
Dataset validation test for ThelerineVTON.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from thelerine_vton.datasets.triplet_dataset import TripletDataset


def validate_sample(sample):

    # ---------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------

    assert sample.person.shape == (3, 256, 256)

    assert sample.garment.shape == (3, 256, 256)

    assert sample.condition.shape == (7, 256, 256)

    assert sample.target.shape == (3, 256, 256)

    assert sample.garment_mask.shape == (256, 256)

    # ---------------------------------------------------------
    # Image ranges
    # ---------------------------------------------------------

    assert sample.person.min() >= -1.0
    assert sample.person.max() <= 1.0

    assert sample.garment.min() >= -1.0
    assert sample.garment.max() <= 1.0

    assert sample.target.min() >= -1.0
    assert sample.target.max() <= 1.0

    # ---------------------------------------------------------
    # Condition tensor
    # ---------------------------------------------------------

    assert sample.condition.shape[0] == 7

    # ---------------------------------------------------------
    # Garment mask should be binary
    # ---------------------------------------------------------

    unique = torch.unique(sample.garment_mask)

    assert torch.all(
        (unique == 0) | (unique == 1)
    )


def test_dataset(dataset_root: Path, manifest_file: Path):

    dataset = TripletDataset(
        dataset_root=dataset_root,
        manifest_file=manifest_file,
        image_size=256,
    )

    print(f"\nDataset size: {len(dataset)} samples")

    assert len(dataset) > 0

    # ---------------------------------------------------------
    # Test 5 random samples
    # ---------------------------------------------------------

    num_samples = min(5, len(dataset))

    indices = random.sample(
        range(len(dataset)),
        num_samples,
    )

    for idx in indices:

        print(f"\nTesting sample {idx}")

        sample = dataset[idx]

        validate_sample(sample)

        print(f"Person        : {sample.person.shape}")
        print(f"Garment       : {sample.garment.shape}")
        print(f"Condition     : {sample.condition.shape}")
        print(f"Target        : {sample.target.shape}")
        print(f"Garment Mask  : {sample.garment_mask.shape}")

    print("\n✓ Dataset validation successful.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root directory of the dataset.",
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to manifest.jsonl.",
    )

    args = parser.parse_args()

    test_dataset(
        args.dataset_root,
        args.manifest,
    )