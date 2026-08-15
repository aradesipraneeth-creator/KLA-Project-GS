"""
Evaluation Metrics for Image Restoration: PSNR and SSIM.
Operates on float32 image tensors without synthetic assumptions.
"""

import math
import torch
import numpy as np
from typing import Union
from losses.ssim import ssim as ssim_fn


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    pred, target: [B, C, H, W] or [C, H, W]
    """
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse < eps:
        return 100.0
    return 10.0 * math.log10((data_range**2) / mse)


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> float:
    """
    Computes Structural Similarity Index (SSIM).
    pred, target: [B, C, H, W]
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)

    val = ssim_fn(pred.float(), target.float(), window_size=window_size, data_range=data_range)
    return val.item()
