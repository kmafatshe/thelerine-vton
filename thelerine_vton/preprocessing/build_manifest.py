"""
Build a manifest.jsonl file for ThelerineVTON.

Author:
    Keneilwe Mokoka
"""

from pathlib import Path

from thelerine_vton.preprocessing.dataset_builder import DatasetBuilder


def main():

    # Update this path to point to your dataset
    dataset_root = Path("data")

    builder = DatasetBuilder(dataset_root)

    builder.build()

    output_path = dataset_root / "manifest.jsonl"

    builder.save(output_path)

    print(f"\nManifest written to: {output_path}")
    print(f"Total samples: {len(builder.records)}")


if __name__ == "__main__":
    main()