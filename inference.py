"""
Run inference using a trained ThelerineVTON model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from thelerine_vton.models.vton_generator import VTONGenerator
from thelerine_vton.datasets.transforms import ImageTransform
from thelerine_vton.preprocessing.agnostic import (
    build_clothing_mask,
    make_agnostic,
)


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
        help="Image of the person to dress",
    )

    parser.add_argument(
        "--garment",
        required=True,
        help="Garment image to transfer onto the person",
    )

    parser.add_argument(
        "--seg",
        required=False,
        help="Optional person segmentation .npy file for clothing mask",
    )

    parser.add_argument(
        "--cond",
        required=False,
        help="Optional condition file (.npy) with DensePose/pose encoding",
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

    if args.cond is not None:
        cond_tensor = torch.from_numpy(
            np.load(args.cond)
        ).float()

        if cond_tensor.ndim == 2:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim == 3 and cond_tensor.shape[-1] <= 10:
            cond_tensor = cond_tensor.permute(2, 0, 1)

        cond_tensor = cond_tensor.unsqueeze(0).to(device)
    else:
        cond_tensor = None

    if args.seg is not None:
        seg_tensor = torch.from_numpy(
            np.load(args.seg)
        ).float()

        if seg_tensor.ndim == 2:
            seg_tensor = seg_tensor.unsqueeze(0)
        elif seg_tensor.ndim == 3 and seg_tensor.shape[-1] != seg_tensor.shape[0]:
            seg_tensor = seg_tensor.permute(2, 0, 1)

        seg_tensor = seg_tensor.unsqueeze(0).to(device)
    else:
        seg_tensor = None

    if cond_tensor is None or seg_tensor is None:
        # Fallback dummy condition if real conditioning is unavailable.
        cond_tensor = cond_tensor or torch.zeros(
            1,
            6,
            256,
            256,
            device=device,
        )

        seg_tensor = seg_tensor or torch.zeros(
            1,
            1,
            256,
            256,
            device=device,
        )

    garment_mask = build_clothing_mask(seg_tensor)
    garment_mask = garment_mask.to(device)

    condition = torch.cat([cond_tensor.squeeze(0), garment_mask.squeeze(0)], dim=0)
    condition = condition.unsqueeze(0)

    person_agnostic = make_agnostic(person_tensor, seg_tensor)

    model = load_model(
        Path(args.checkpoint),
        device,
    )

    with torch.no_grad():

        prediction = model(

            person_agnostic,

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