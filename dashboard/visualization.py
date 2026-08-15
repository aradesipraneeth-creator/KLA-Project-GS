"""
KLA Semiconductor Image Restoration — Dashboard Visualization Utilities.
Generates publication-quality figures, difference heatmaps, histograms,
and frequency spectra for display without modifying original float32 arrays.
"""

import io
from typing import Optional, Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from PIL import Image


def normalize_for_display(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile_clip: bool = False,
) -> np.ndarray:
    """
    Normalizes a float32 image array to [0, 1] for UI visualization only.
    Preserves original float32 data unchanged.
    """
    arr = arr.astype(np.float32)
    if percentile_clip:
        low = np.percentile(arr, 1) if vmin is None else vmin
        high = np.percentile(arr, 99) if vmax is None else vmax
    else:
        low = np.min(arr) if vmin is None else vmin
        high = np.max(arr) if vmax is None else vmax

    denom = high - low
    if denom < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)

    norm = np.clip((arr - low) / denom, 0.0, 1.0)
    return norm


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Converts a float32 2D array to PNG bytes for browser download."""
    disp = (normalize_for_display(arr) * 255.0).astype(np.uint8)
    img = Image.fromarray(disp, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def array_to_npy_bytes(arr: np.ndarray) -> bytes:
    """Serializes a float32 array to .npy binary bytes for browser download."""
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def plot_comparison_panels(
    lq: np.ndarray,
    bicubic: np.ndarray,
    restored: np.ndarray,
    gt: Optional[np.ndarray] = None,
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
    bic_psnr: Optional[float] = None,
    bic_ssim: Optional[float] = None,
) -> plt.Figure:
    """
    Creates a 3-panel (Input, Bicubic, Restored) or 4-panel (+ GT) figure.
    """
    num_panels = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, num_panels, figsize=(4.5 * num_panels, 4.5), dpi=150)

    # 1. Input Noisy LQ (128x128)
    axes[0].imshow(lq, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Input (Noisy LQ)\n{lq.shape[0]}×{lq.shape[1]} float32", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Bicubic Baseline (256x256)
    bic_title = "Bicubic Baseline\n256×256 float32"
    if bic_psnr is not None and bic_ssim is not None:
        bic_title += f"\nPSNR: {bic_psnr:.2f}dB | SSIM: {bic_ssim:.4f}"
    axes[1].imshow(bicubic, cmap="gray", interpolation="nearest")
    axes[1].set_title(bic_title, fontsize=11)
    axes[1].axis("off")

    # 3. KLA-HYBRID-V2 Restored (256x256)
    rest_title = "KLA-HYBRID-V2 Restored\n256×256 float32"
    if psnr is not None and ssim is not None:
        rest_title += f"\nPSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}"
    axes[2].imshow(restored, cmap="gray", interpolation="nearest")
    axes[2].set_title(rest_title, fontsize=11, fontweight="bold", color="#0066cc")
    axes[2].axis("off")

    # 4. Ground Truth if available (256x256)
    if gt is not None:
        axes[3].imshow(gt, cmap="gray", interpolation="nearest")
        axes[3].set_title(f"Ground Truth (Reference)\n{gt.shape[0]}×{gt.shape[1]} float32", fontsize=11, fontweight="bold")
        axes[3].axis("off")

    plt.tight_layout()
    return fig


def plot_error_heatmap(
    abs_error: np.ndarray,
    title: str = "Absolute Error Map (|Restored - Ground Truth|)",
) -> plt.Figure:
    """Creates a high-contrast heatmap for residual and reconstruction error."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(abs_error, cmap="inferno", interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def plot_histograms(
    lq: np.ndarray,
    restored: np.ndarray,
    gt: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plots pixel intensity distribution histograms using actual float32 values."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    ax.hist(lq.ravel(), bins=60, alpha=0.5, label="Noisy LQ (128×128)", color="#ff7f0e", density=True)
    ax.hist(restored.ravel(), bins=80, alpha=0.6, label="Restored V2 (256×256)", color="#1f77b4", density=True)
    if gt is not None:
        ax.hist(gt.ravel(), bins=80, alpha=0.4, label="Ground Truth (256×256)", color="#2ca02c", density=True)

    ax.set_title("Intensity Distribution Histogram (Original Float32 Values)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Pixel Intensity Value", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_fft_spectrum_comparison(
    lq_mag: np.ndarray,
    restored_mag: np.ndarray,
    gt_mag: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plots 2D FFT log-magnitude spectra."""
    num_cols = 3 if gt_mag is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(4.5 * num_cols, 4.5), dpi=150)

    axes[0].imshow(lq_mag, cmap="viridis", interpolation="nearest")
    axes[0].set_title("Input LQ Spectrum\n(Log FFT Magnitude)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(restored_mag, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Restored Spectrum\n(Log FFT Magnitude)", fontsize=11)
    axes[1].axis("off")

    if gt_mag is not None:
        axes[2].imshow(gt_mag, cmap="viridis", interpolation="nearest")
        axes[2].set_title("Ground Truth Spectrum\n(Log FFT Magnitude)", fontsize=11)
        axes[2].axis("off")

    plt.tight_layout()
    return fig
