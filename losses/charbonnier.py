"""
Charbonnier Loss for robust image restoration.
sqrt((x - y)^2 + eps^2)
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a smooth L1 loss variant)."""

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
