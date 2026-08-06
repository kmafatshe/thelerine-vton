"""
Run inference using a trained ThelerineVTON model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.datasets.transforms import ImageTransform


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
):

    model = VTONGenerator()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--person",
        required=True,
    )

    parser.add_argument(
        "--garment",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="output.png",
    )

    args = parser.parse_args()

    device = get_device()

    transform = ImageTransform(256)

    person = Image.open(
        args.person
    ).convert("RGB")

    garment = Image.open(
        args.garment
    ).convert("RGB")

    person_tensor = transform(person).unsqueeze(0).to(device)

    garment_tensor = transform(garment).unsqueeze(0).to(device)

    # Placeholder condition tensor: zeros with 7 channels.
    # This is compatible with the model input shape, but does not
    # provide real DensePose/segmentation conditioning.
    condition = torch.zeros(
        1,
        7,
        256,
        256,
        device=device,
    )

    model = load_model(
        Path(args.checkpoint),
        device,
    )

    with torch.no_grad():

        prediction = model(

            person_tensor,

            garment_tensor,

            condition,

        )

    save_image(

        prediction.image,

        args.output,

        normalize=True,

        value_range=(-1, 1),

    )

    print(f"\nSaved output to {args.output}")


if __name__ == "__main__":

    main()