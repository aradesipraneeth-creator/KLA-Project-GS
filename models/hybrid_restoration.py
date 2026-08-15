"""
KLA Semiconductor Image Restoration — Hybrid Restoration Architecture (V2).
Combines Shallow Feature Extraction, 6 Hybrid Restoration Groups
(each containing 6 NAF Blocks + 2 Swin Transformer Blocks + Local Residual),
Reconstruction with 2x PixelShuffle, and Global Bicubic Residual Connection.
"""

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .naf_block import NAFBlock, LayerNorm2D
from .swin_block import SwinTransformerBlock
from .frequency_guidance import StructureFrequencyGuidance


class ResConvBlock(nn.Module):
    """Residual Convolutional Block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class HybridRestorationGroup(nn.Module):
    """
    Hybrid Restoration Group.
    Contains:
    - naf_blocks_per_group (default 6) NAF Blocks
    - swin_blocks_per_group (default 2: 1 W-MSA, 1 SW-MSA) Swin Transformer Blocks
    - Optional StructureFrequencyGuidance module (default False)
    - 3x3 Conv
    - Local Residual Connection
    """

    def __init__(
        self,
        channels: int,
        num_naf: int = 6,
        num_swin: int = 2,
        window_size: int = 8,
        num_heads: int = 4,
        naf_expansion: int = 2,
        drop_path_rate: float = 0.0,
        use_frequency_guidance: bool = False,
    ):
        super().__init__()
        self.naf_blocks = nn.ModuleList(
            [
                NAFBlock(
                    channels=channels,
                    expansion=naf_expansion,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(num_naf)
            ]
        )

        self.swin_blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    drop_path=drop_path_rate,
                )
                for i in range(num_swin)
            ]
        )

        self.sfg = (
            StructureFrequencyGuidance(channels=channels)
            if use_frequency_guidance
            else None
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for block in self.naf_blocks:
            x = block(x)
        for block in self.swin_blocks:
            x = block(x)
        if self.sfg is not None:
            x = self.sfg(x)
        x = self.conv(x)
        return res + x


class HybridRestorationNet(nn.Module):
    """
    KLA Hybrid Restoration Architecture V2.
    Input -> 3x3 Conv -> ResConvBlock -> 6 Hybrid Restoration Groups
    -> Reconstruction Conv -> PixelShuffle x2 -> Final Conv -> Global Bicubic Residual -> Output
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 32,
        num_channels: Optional[int] = None,
        num_groups: int = 6,
        naf_blocks_per_group: int = 6,
        num_naf_per_group: Optional[int] = None,
        swin_blocks_per_group: int = 2,
        num_swin_per_group: Optional[int] = None,
        window_size: int = 8,
        swin_heads: int = 4,
        num_heads: Optional[int] = None,
        naf_expansion: int = 2,
        drop_path_rate: float = 0.0,
        scale: int = 2,
        use_frequency_guidance: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        # Handle parameter aliases
        dim = num_channels if num_channels is not None else base_dim
        n_naf = num_naf_per_group if num_naf_per_group is not None else naf_blocks_per_group
        n_swin = num_swin_per_group if num_swin_per_group is not None else swin_blocks_per_group
        n_heads = num_heads if num_heads is not None else swin_heads

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_dim = dim
        self.num_groups = num_groups
        self.naf_blocks_per_group = n_naf
        self.swin_blocks_per_group = n_swin
        self.window_size = window_size
        self.swin_heads = n_heads
        self.use_frequency_guidance = use_frequency_guidance

        # 1. Shallow Feature Extraction
        self.intro_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=True)
        self.res_conv = ResConvBlock(dim)

        # 2. Deep Feature Extraction (6 Hybrid Restoration Groups)
        self.groups = nn.ModuleList(
            [
                HybridRestorationGroup(
                    channels=dim,
                    num_naf=n_naf,
                    num_swin=n_swin,
                    window_size=window_size,
                    num_heads=n_heads,
                    naf_expansion=naf_expansion,
                    drop_path_rate=drop_path_rate,
                    use_frequency_guidance=use_frequency_guidance,
                )
                for _ in range(num_groups)
            ]
        )
        self.group_agg = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

        # 3. Reconstruction & PixelShuffle 2x Upsampling
        self.reconstruction_conv = nn.Conv2d(
            dim, dim * (scale**2), kernel_size=3, padding=1, bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # 4. Final Convolution
        self.final_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Global Bicubic Residual Connection.
        
        Args:
            x: Input tensor [B, 1, 128, 128]
        Returns:
            Restored output tensor [B, 1, 256, 256]
        """
        # Global Bicubic Baseline (computed in same domain as x)
        bicubic_baseline = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )

        # Shallow feature extraction
        fea = self.intro_conv(x)
        fea = self.res_conv(fea)

        # Deep hybrid group feature extraction
        res_fea = fea
        for group in self.groups:
            fea = group(fea)
        fea = self.group_agg(fea) + res_fea

        # Reconstruction & PixelShuffle 2x
        fea = self.reconstruction_conv(fea)
        fea = self.pixel_shuffle(fea)
        res_image = self.final_conv(fea)

        # Global Residual Addition
        out = bicubic_baseline + res_image
        return out
