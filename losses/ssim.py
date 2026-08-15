"""
Differentiable SSIM and SSIM Loss module for PyTorch.
Computes Structural Similarity Index for 2D single/multi-channel images,
supporting configurable dynamic range for normalized or unbounded float32 inputs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    _1D_window = _gaussian_window(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Computes SSIM between img1 and img2.
    img1, img2: [B, C, H, W]
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(device=img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """1.0 - SSIM Loss."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Determine dynamic range if not fixed
        dr = self.data_range
        return 1.0 - ssim(pred, target, window_size=self.window_size, data_range=dr)
