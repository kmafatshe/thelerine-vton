"""
Loss functions for ThelerineVTON.
"""

from __future__ import annotations

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageLoss(nn.Module):
    """
    Standard L1 image reconstruction loss.
    """

    def forward(
        self,
        prediction,
        target,
    ):
        return F.l1_loss(
            prediction,
            target,
        )
    
class PerceptualLoss(nn.Module):
    """
    LPIPS perceptual similarity loss.
    """

    def __init__(self):

        super().__init__()

        self.loss = lpips.LPIPS(
            net="vgg"
        )

        self.loss.eval()

        for p in self.loss.parameters():
            p.requires_grad = False

    def forward(
        self,
        prediction,
        target,
    ):
        return self.loss(
            prediction,
            target,
        ).mean()

class EdgeLoss(nn.Module):
    """
    Encourages sharp garment boundaries.
    """

    def gradient_x(self, img):

        return img[:, :, :, 1:] - img[:, :, :, :-1]

    def gradient_y(self, img):

        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def forward(
        self,
        prediction,
        target,
    ):

        loss_x = F.l1_loss(
            self.gradient_x(prediction),
            self.gradient_x(target),
        )

        loss_y = F.l1_loss(
            self.gradient_y(prediction),
            self.gradient_y(target),
        )

        return loss_x + loss_y    
    
class GarmentConsistencyLoss(nn.Module):
    """
    Computes loss only inside the clothing region.
    """

    def forward(
        self,
        prediction,
        target,
        garment_mask,
    ):

        mask = garment_mask.float()

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        mask = mask.repeat(
            1,
            3,
            1,
            1,
        )

        diff = torch.abs(
            prediction - target
        )

        diff = diff * mask

        return diff.sum() / (
            mask.sum() + 1e-6
        )

class TotalLoss(nn.Module):

    def __init__(

        self,

        image_weight=1.0,

        perceptual_weight=0.3,

        edge_weight=0.2,

        garment_weight=2.0,

    ):

        super().__init__()

        self.image = ImageLoss()

        self.perceptual = PerceptualLoss()

        self.edge = EdgeLoss()

        self.garment = GarmentConsistencyLoss()

        self.w_image = image_weight

        self.w_perc = perceptual_weight

        self.w_edge = edge_weight

        self.w_garment = garment_weight

    def forward(

        self,

        prediction,

        target,

        garment_mask,

    ):

        image_loss = self.image(
            prediction,
            target,
        )

        perceptual_loss = self.perceptual(
            prediction,
            target,
        )

        edge_loss = self.edge(
            prediction,
            target,
        )

        garment_loss = self.garment(
            prediction,
            target,
            garment_mask,
        )

        total = (

            self.w_image * image_loss

            + self.w_perc * perceptual_loss

            + self.w_edge * edge_loss

            + self.w_garment * garment_loss

        )

        losses = {

            "total": total,

            "image": image_loss,

            "perceptual": perceptual_loss,

            "edge": edge_loss,

            "garment": garment_loss,

        }

        return losses        