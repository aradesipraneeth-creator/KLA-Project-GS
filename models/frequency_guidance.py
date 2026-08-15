"""
Structure-Frequency Guidance (SFG) Module for Semiconductor Image Restoration.
Distinguishes genuine directional semiconductor structures from isotropic speckle/Gaussian noise
using lightweight frequency-domain gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class StructureFrequencyGuidance(nn.Module):
    """
    Lightweight Structure-Frequency Guidance module.
    Modulates feature spectra to attenuate isotropic noise while preserving
    directional semiconductor line and edge frequencies.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        hidden_dim = max(8, channels // reduction)

        # Spectral gating network
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable residual scale initialized near zero for seamless ablation
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. 2D Real FFT
        fft_feat = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        mag = torch.abs(fft_feat) + 1e-8

        # 2. Learn structure mask in frequency domain
        gate = self.freq_gate(mag)  # [B, C, H, W//2 + 1]

        # 3. Modulate spectral representation
        fft_modulated = fft_feat * (1.0 + gate * self.scale)

        # 4. Inverse 2D Real FFT
        out = torch.fft.irfft2(fft_modulated, s=(H, W), dim=(-2, -1), norm="ortho")

        return x + out
