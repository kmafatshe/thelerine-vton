"""
Unit tests for ThelerineVTON.

Tests:
    - Model construction
    - Forward pass
    - Tensor shapes
    - Output ranges
"""

from __future__ import annotations

import torch

from thelerine_vton.models.vton_generator import VTONGenerator


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def test_model_forward():

    device = torch.device("cpu")

    model = VTONGenerator(
        base_channels=64
    ).to(device)
    
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    model.eval()

    batch_size = 1

    person = (torch.rand(
        batch_size,
        3,
        128,
        128,
        device=device,
        )* 2 
    )- 1

    garment = (torch.rand(
        batch_size,
        3,
        128,
        128,
        device=device,
         )* 2 
    )- 1
   

    # DensePose (6) + Clothing Mask (1)
    condition = torch.randn(
        batch_size,
        7,
        128,
        128,
        device=device,
    )

    with torch.no_grad():

        output, residual, blend, confidence = model(
            person,
            garment,
            condition,
        )

    # -------------------------------------------------
    # Shape checks
    # -------------------------------------------------

    assert output.shape == (
        batch_size,
        3,
        128,
        128,
    )

    assert residual.shape == (
        batch_size,
        3,
        128,
        128,
    )

    assert blend.shape == (
        batch_size,
        1,
        128,
        128,
    )

    assert confidence.shape == (
        batch_size,
        1,
        128,
        128,
    )

    # -------------------------------------------------
    # Value checks
    # -------------------------------------------------

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)

    assert torch.all(blend >= 0.0)
    assert torch.all(blend <= 1.0)

    assert torch.all(confidence >= 0.0)
    assert torch.all(confidence <= 1.0)

    print("\nModel forward pass successful.\n")

    print(f"Output      : {output.shape}")
    print(f"Residual    : {residual.shape}")
    print(f"Blend       : {blend.shape}")
    print(f"Confidence  : {confidence.shape}")


if __name__ == "__main__":

    test_model_forward()