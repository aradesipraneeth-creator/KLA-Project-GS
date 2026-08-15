"""
Swin Transformer Block for Image Restoration.
Includes Window Partitioning, Window Reverse, Window Attention (W-MSA & SW-MSA)
with Relative Position Bias, LayerNorm, and MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .naf_block import DropPath


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: [B, H, W, C]
        window_size: (int) window size
    Returns:
        windows: [B * num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: [B * num_windows, window_size, window_size, C]
        window_size: (int) Window size
        H: (int) Height of image
        W: (int) Width of image
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, nH, N, d]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, nH, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )  # [Wh*Ww, Wh*Ww, nH]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Wh*Ww, Wh*Ww]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer and Swin."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block supporting spatial 2D feature maps [B, C, H, W]."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is [B, C, H, W]
        B, C, H, W = x.shape
        # Permute to channels-last: [B, H, W, C]
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x_perm

        x_norm = self.norm1(x_perm)

        # Pad feature maps if H, W not divisible by window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x_norm.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # Calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # [nW, ws, ws, 1]
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            shifted_x = x_norm
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_norm = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x_norm = shifted_x

        if pad_r > 0 or pad_b > 0:
            x_norm = x_norm[:, :H, :W, :].contiguous()

        # FFN & Residual
        x_perm = shortcut + self.drop_path(x_norm)
        x_perm = x_perm + self.drop_path(self.mlp(self.norm2(x_perm)))

        # Permute back to channels-first [B, C, H, W]
        return x_perm.permute(0, 3, 1, 2).contiguous()
