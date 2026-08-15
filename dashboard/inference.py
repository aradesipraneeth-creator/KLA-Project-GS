"""
KLA Semiconductor Image Restoration — Dashboard Inference Engine.
Handles model loading with caching, single-image restoration, batch inference,
metric evaluation, and difference calculations without modifying core files.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hybrid_restoration import HybridRestorationNet
from utils.normalization import normalize, denormalize, KLA_MEAN, KLA_STD
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from trainer.trainer import MODEL_VERSION, get_autocast_context


def find_available_checkpoints() -> List[Path]:
    """Discovers all available .pth checkpoint files in the project."""
    search_dirs = [
        REPO_ROOT / "outputs" / "v2" / "checkpoints",
        REPO_ROOT / "outputs" / "checkpoints",
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "outputs" / "v2",
    ]
    checkpoints = []
    for s_dir in search_dirs:
        if s_dir.exists() and s_dir.is_dir():
            for p in s_dir.glob("*.pth"):
                if p.is_file() and not p.name.startswith("."):
                    checkpoints.append(p)
    return sorted(list(set(checkpoints)), key=lambda x: str(x))


def load_model_from_checkpoint(
    weights_path: Union[str, Path],
    device: str = "cpu",
    use_sfg: bool = False,
) -> Tuple[nn.Module, Dict]:
    """
    Loads HybridRestorationNet from checkpoint with fallback parameter discovery.
    
    Args:
        weights_path: Path to checkpoint .pth
        device: 'cpu' or 'cuda'
        use_sfg: Whether to activate Structure-Frequency Guidance module
        
    Returns:
        Tuple of (model, checkpoint_metadata_dict)
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    checkpoint = torch.load(str(weights_path), map_location=dev)

    # Extract state dict (prefer EMA weights)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    # Infer architecture dimensions from state dict
    if "intro_conv.weight" in cleaned_state_dict:
        num_channels = cleaned_state_dict["intro_conv.weight"].shape[0]
    else:
        num_channels = 32

    # Group count
    group_indices = set()
    for k in cleaned_state_dict.keys():
        if k.startswith("groups."):
            g_idx = int(k.split(".")[1])
            group_indices.add(g_idx)
    num_groups = len(group_indices) if group_indices else 6

    # NAF blocks count
    naf_indices = set()
    for k in cleaned_state_dict.keys():
        if k.startswith("groups.0.naf_blocks."):
            n_idx = int(k.split(".")[3])
            naf_indices.add(n_idx)
    num_naf_per_group = len(naf_indices) if naf_indices else 6

    # Swin blocks count
    swin_indices = set()
    for k in cleaned_state_dict.keys():
        if k.startswith("groups.0.swin_blocks."):
            s_idx = int(k.split(".")[3])
            swin_indices.add(s_idx)
    num_swin_per_group = len(swin_indices) if swin_indices else 2

    model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        base_dim=num_channels,
        num_groups=num_groups,
        naf_blocks_per_group=num_naf_per_group,
        swin_blocks_per_group=num_swin_per_group,
        window_size=8,
        swin_heads=4,
        scale=2,
        use_frequency_guidance=use_sfg,
    )

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(dev)
    model.eval()

    meta = {
        "model_version": checkpoint.get("model_version", "KLA-HYBRID-V2"),
        "epoch": checkpoint.get("epoch", "N/A"),
        "best_psnr": checkpoint.get("best_psnr", None),
        "best_ssim": checkpoint.get("best_ssim", None),
        "num_channels": num_channels,
        "num_groups": num_groups,
        "num_naf_per_group": num_naf_per_group,
        "num_swin_per_group": num_swin_per_group,
        "has_ema": "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None,
    }

    return model, meta


def restore_single_image(
    model: nn.Module,
    lq_arr: np.ndarray,
    device: str = "cpu",
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs end-to-end inference on a single float32 128x128 image.
    
    Args:
        model: PyTorch model
        lq_arr: [128, 128] float32 input array
        device: 'cpu' or 'cuda'
        amp: Enable mixed precision on CUDA
        
    Returns:
        Tuple of (restored_256x256_float32, bicubic_256x256_float32, latency_ms)
    """
    # 1. Validation
    if not isinstance(lq_arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(lq_arr)}")
    if lq_arr.shape != (128, 128):
        raise ValueError(f"Expected shape (128, 128), got {lq_arr.shape}")
    if not np.all(np.isfinite(lq_arr)):
        raise ValueError("Input array contains NaN or Inf values!")

    lq_float32 = lq_arr.astype(np.float32)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(dev)
    model.eval()

    # 2. Compute Bicubic baseline in original domain
    lq_raw_tensor = torch.from_numpy(lq_float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128]
    bicubic_tensor = F.interpolate(
        lq_raw_tensor, scale_factor=2, mode="bicubic", align_corners=False
    )
    bicubic_arr = bicubic_tensor.squeeze().numpy().astype(np.float32)

    # 3. Normalize for neural network feature extraction
    lq_norm = normalize(lq_float32, KLA_MEAN, KLA_STD)
    tensor_in = torch.from_numpy(lq_norm).unsqueeze(0).unsqueeze(0).to(dev)

    # 4. Timed Inference
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.inference_mode():
        with get_autocast_context(dev.type, enabled=(amp and dev.type == "cuda")):
            pred_norm = model(tensor_in)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    # 5. Denormalize to original float32 intensity range
    pred_raw = denormalize(pred_norm, KLA_MEAN, KLA_STD)
    restored_arr = pred_raw.squeeze().cpu().numpy().astype(np.float32)

    return restored_arr, bicubic_arr, latency_ms


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """Computes PSNR and SSIM between prediction and ground truth float32 arrays."""
    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    dr = max(0.1, float(gt.max() - gt.min()))
    psnr_val = calculate_psnr(pred_t, gt_t, data_range=dr)
    ssim_val = calculate_ssim(pred_t, gt_t, data_range=dr)
    
    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "data_range": dr,
    }


def compute_diff_maps(pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes spatial absolute error and signed residual maps."""
    abs_error = np.abs(pred - gt).astype(np.float32)
    residual = (pred - gt).astype(np.float32)
    return {
        "abs_error": abs_error,
        "residual": residual,
    }


def compute_fft_magnitude(arr: np.ndarray) -> np.ndarray:
    """Computes 2D FFT log-magnitude spectrum for frequency inspection."""
    fft_2d = np.fft.fftshift(np.fft.fft2(arr.astype(np.float32)))
    magnitude = np.abs(fft_2d)
    log_mag = np.log1p(magnitude)
    return log_mag.astype(np.float32)


def run_batch_test(
    model: nn.Module,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "cpu",
    amp: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Union[int, float, str]]:
    """
    Executes batch restoration on all .npy files in input_dir and saves to output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in input_dir.iterdir()
        if f.suffix.lower() == ".npy" and not f.name.startswith(".") and not f.name.startswith("._")
    ]
    files.sort(key=lambda x: x.name)

    if not files:
        raise ValueError(f"No valid .npy files found in {input_dir}")

    latencies = []
    failed_count = 0

    for idx, f_path in enumerate(files, 1):
        try:
            arr = np.load(str(f_path)).astype(np.float32)
            restored, _, lat_ms = restore_single_image(model, arr, device=device, amp=amp)
            np.save(str(output_dir / f_path.name), restored)
            latencies.append(lat_ms)
        except Exception:
            failed_count += 1

        if progress_callback is not None:
            progress_callback(idx, len(files), f_path.name)

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "total_files": len(files),
        "success_count": len(files) - failed_count,
        "failed_count": failed_count,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": med_lat,
        "p95_latency_ms": p95_lat,
        "throughput_fps": fps,
        "output_dir": str(output_dir),
    }
