
"""
Attention block for ThelerineVTON.

Temporary identity implementation for the swap-training baseline.
This keeps the model API intact while removing the O(N^2) memory cost
of spatial self-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_tokens_side: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.max_tokens_side = max_tokens_side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
