"""
NAFBlock (Nonlinear Activation Free Block) Implementation.
Includes LayerNorm2D, Pointwise Conv, Depthwise Conv, SimpleGate,
Simplified Channel Attention (SCA), Residual Scaling, and DropPath.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2D(nn.Module):
    """Channels-first 2D Layer Normalization for [B, C, H, W] tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class SimpleGate(nn.Module):
    """SimpleGate: splits channels in half and computes element-wise multiplication."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA)."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.fc(self.pool(x))
        return x * attn


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block.
    Consists of Spatial Mixing (DWConv + SimpleGate + SCA) and Channel Mixing (FFN + SimpleGate).
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        drop_path_rate: float = 0.0,
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.res_scale = res_scale
        expanded_channels = channels * expansion

        # Spatial Mixing Block
        self.norm1 = LayerNorm2D(channels)
        self.conv1 = nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            padding=1,
            groups=expanded_channels,
            bias=True,
        )
        self.sg1 = SimpleGate()
        self.sca = SimpleChannelAttention(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # Channel Mixing Block (FFN)
        self.norm2 = LayerNorm2D(channels)
        self.conv4 = nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Learnable channel-wise scale parameters (residual scaling)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Spatial Mixing
        res = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = self.drop_path(y)
        x = res + y * (1.0 + self.beta) * self.res_scale

        # 2. Channel Mixing
        res = x
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.drop_path(y)
        x = res + y * (1.0 + self.gamma) * self.res_scale

        return x
