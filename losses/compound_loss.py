"""
Compound Multi-Objective Loss Module for KLA Semiconductor Image Restoration (V2).
Authoritative Compound Loss:
0.60 * Charbonnier + 0.25 * SSIM + 0.15 * FFT

Provides individual loss tracking, weighted components, dynamic range control,
and clean training curriculum support.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from .charbonnier import CharbonnierLoss
from .ssim import SSIMLoss
from .fft_loss import FFTLoss


class CompoundLoss(nn.Module):
    """
    Authoritative Compound Loss for KLA V2.
    L_total = w_charb * L_charb + w_ssim * L_ssim + w_fft * L_fft
    """

    def __init__(
        self,
        weight_charbonnier: float = 0.60,
        weight_ssim: float = 0.25,
        weight_fft: float = 0.15,
        charbonnier_eps: float = 1e-3,
        ssim_window_size: int = 11,
        ssim_data_range: float = 1.0,
    ):
        super().__init__()
        self.w_charb = float(weight_charbonnier)
        self.w_ssim = float(weight_ssim)
        self.w_fft = float(weight_fft)

        self.charbonnier_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size, data_range=ssim_data_range)
        self.fft_loss = FFTLoss()

    def set_weights(
        self,
        weight_charbonnier: Optional[float] = None,
        weight_ssim: Optional[float] = None,
        weight_fft: Optional[float] = None,
    ):
        """Allows dynamic curriculum adjustment of loss weights."""
        if weight_charbonnier is not None:
            self.w_charb = float(weight_charbonnier)
        if weight_ssim is not None:
            self.w_ssim = float(weight_ssim)
        if weight_fft is not None:
            self.w_fft = float(weight_fft)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, data_range: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the compound loss with full diagnostic breakdown.
        
        Args:
            pred: Restored prediction tensor [B, C, H, W]
            target: Ground truth target tensor [B, C, H, W]
            data_range: Optional dynamic range override for SSIM
            
        Returns:
            Tuple of (total_loss_tensor, loss_dict_with_all_components)
        """
        # 1. Individual Raw Losses
        raw_charb = self.charbonnier_loss(pred, target)
        
        if data_range is not None:
            self.ssim_loss.data_range = data_range
        raw_ssim = self.ssim_loss(pred, target)
        
        raw_fft = self.fft_loss(pred, target)

        # 2. Weighted Losses
        weighted_charb = self.w_charb * raw_charb
        weighted_ssim = self.w_ssim * raw_ssim
        weighted_fft = self.w_fft * raw_fft

        # 3. Total Loss
        total_loss = weighted_charb + weighted_ssim + weighted_fft

        # 4. Diagnostic Dictionary
        loss_dict = {
            "total_loss": total_loss.item(),
            "charbonnier": raw_charb.item(),
            "ssim": raw_ssim.item(),
            "fft": raw_fft.item(),
            "weighted_charbonnier": weighted_charb.item(),
            "weighted_ssim": weighted_ssim.item(),
            "weighted_fft": weighted_fft.item(),
            "w_charb": self.w_charb,
            "w_ssim": self.w_ssim,
            "w_fft": self.w_fft,
        }

        return total_loss, loss_dict
