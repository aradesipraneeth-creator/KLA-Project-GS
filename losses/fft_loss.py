"""
Normalized 2D Frequency (FFT) Domain Loss for KLA Semiconductor Restoration.
Uses orthonormal FFT ('ortho' normalization) to ensure the loss is scale-invariant
and numerically balanced with spatial domain losses (Charbonnier, SSIM).
"""

import torch
import torch.nn as nn
import torch.fft


class FFTLoss(nn.Module):
    """
    Numerically Stable Normalized 2D Real FFT Loss.
    Measures L1 distance between orthonormal frequency representations of prediction and ground truth.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        Returns:
            Scalar FFT L1 loss
        """
        # Orthonormal 2D RFFT ensures scale-invariance: energy is preserved without blowing up with H*W
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")

        pred_real = torch.real(pred_fft)
        pred_imag = torch.imag(pred_fft)
        target_real = torch.real(target_fft)
        target_imag = torch.imag(target_fft)

        loss_real = self.criterion(pred_real, target_real)
        loss_imag = self.criterion(pred_imag, target_imag)

        return self.loss_weight * (loss_real + loss_imag)
