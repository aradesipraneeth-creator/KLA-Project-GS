"""
Official Evaluation and Test Set Inference Script for KLA Semiconductor Restoration (V2).
Processes all 400 official KLA test images from Test_NoisyLR/NoisyLR/,
generates 2x super-resolved restored 256x256 float32 arrays in original intensity domain,
saves them to outputs/v2/test_outputs/<filename>.npy, and outputs detailed latency & throughput benchmarks.

Command:
    python evaluate.py --input_dir ./Test_NoisyLR/NoisyLR --output_dir ./outputs/v2/test_outputs --weights ./outputs/v2/checkpoints/best_psnr.pth --device cuda --amp
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from models.hybrid_restoration import HybridRestorationNet
from trainer.trainer import get_autocast_context, MODEL_VERSION
from datasets.kla_dataset import (
    get_valid_npy_files,
    resolve_dataset_dir,
)
from utils.normalization import normalize, denormalize, KLA_MEAN, KLA_STD

# Inference Configuration
USE_TTA: bool = False  # Full-image fast inference default


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KLA Test Set (V2)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./Test_NoisyLR/NoisyLR",
        help="Path to Test_NoisyLR/NoisyLR directory containing 400 .npy files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/v2/test_outputs",
        help="Path to directory where 256x256 restored float32 .npy files will be saved",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./outputs/v2/checkpoints/best_psnr.pth",
        help="Path to model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device ('cuda', 'cuda:0', 'cpu', etc.)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable FP16 automatic mixed precision for maximum throughput",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations for GPU benchmark",
    )
    return parser.parse_args()


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Loads model weights with version verification."""
    print(f"[INFO] Loading checkpoint from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    # Validate version tag if present
    ckpt_ver = checkpoint.get("model_version")
    if ckpt_ver is not None and ckpt_ver != MODEL_VERSION:
        raise ValueError(
            f"Incompatible checkpoint version '{ckpt_ver}', expected '{MODEL_VERSION}'."
        )

    # Extract state dict (prefer EMA weights)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    # Extract architecture dimensions from state dict
    if "intro_conv.weight" in cleaned_state_dict:
        num_channels = cleaned_state_dict["intro_conv.weight"].shape[0]
    else:
        num_channels = 32

    # Count number of groups
    group_indices = set()
    for k in cleaned_state_dict.keys():
        if k.startswith("groups."):
            g_idx = int(k.split(".")[1])
            group_indices.add(g_idx)
    num_groups = len(group_indices) if group_indices else 6

    # Count NAF blocks per group
    naf_indices = set()
    for k in cleaned_state_dict.keys():
        if k.startswith("groups.0.naf_blocks."):
            n_idx = int(k.split(".")[3])
            naf_indices.add(n_idx)
    num_naf_per_group = len(naf_indices) if naf_indices else 6

    # Count Swin blocks per group
    swin_indices = set()
    for k in cleaned_state_dict.keys():
        if k.startswith("groups.0.swin_blocks."):
            s_idx = int(k.split(".")[3])
            swin_indices.add(s_idx)
    num_swin_per_group = len(swin_indices) if swin_indices else 2

    print(
        f"[INFO] Inferred Architecture: Channels={num_channels}, Groups={num_groups}, "
        f"NAF/Group={num_naf_per_group}, Swin/Group={num_swin_per_group}"
    )

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
    )

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    input_dir = resolve_dataset_dir(args.input_dir, target_type="test")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("=" * 65)
    print(" KLA OFFICIAL TEST SET EVALUATION & INFERENCE (V2)")
    print(f" Input Directory: {input_dir}")
    print(f" Output Directory: {output_dir}")
    print(f" Checkpoint: {args.weights}")
    print(f" Device: {device} (AMP={args.amp}, USE_TTA={USE_TTA})")
    print("=" * 65)

    test_files = get_valid_npy_files(input_dir)
    print(f"[INFO] Found {len(test_files)} valid real test files to process.")

    # Load model
    model = load_model(args.weights, device)

    # Warmup GPU
    if device.type == "cuda" and args.warmup > 0 and len(test_files) > 0:
        print(f"[INFO] Running {args.warmup} GPU warmup iterations...")
        dummy_in = torch.randn(1, 1, 128, 128, device=device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model(dummy_in)
        torch.cuda.synchronize()

    model_latencies = []
    e2e_latencies = []
    failed_files = []

    print("[INFO] Beginning evaluation...")

    for idx, file_path in enumerate(test_files, 1):
        t_e2e_start = time.perf_counter()

        try:
            # 1. Load array float32
            arr = np.load(str(file_path)).astype(np.float32)
            if arr.shape != (128, 128):
                raise ValueError(f"Invalid input shape {arr.shape}, expected (128, 128)")

            # 2. Normalize and convert to tensor [1, 1, 128, 128]
            arr_norm = normalize(arr, KLA_MEAN, KLA_STD)
            tensor_in = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(device)

            # 3. Model Inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model_start = time.perf_counter()

            with torch.inference_mode():
                with get_autocast_context(device.type, enabled=(args.amp and device.type == "cuda")):
                    pred_norm = model(tensor_in)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model_end = time.perf_counter()

            # 4. Denormalize & save in original float32 domain
            pred_raw = denormalize(pred_norm, KLA_MEAN, KLA_STD)
            out_arr = pred_raw.squeeze().cpu().numpy().astype(np.float32)

            if out_arr.shape != (256, 256):
                raise ValueError(f"Output shape error: {out_arr.shape}, expected (256, 256)")

            save_dest = output_dir / file_path.name
            np.save(str(save_dest), out_arr)

            t_e2e_end = time.perf_counter()

            model_latencies.append((t_model_end - t_model_start) * 1000.0)  # ms
            e2e_latencies.append((t_e2e_end - t_e2e_start) * 1000.0)  # ms

            if idx % 50 == 0 or idx == len(test_files):
                print(
                    f"[{idx}/{len(test_files)}] Processed {file_path.name} | "
                    f"Model: {model_latencies[-1]:.2f}ms | E2E: {e2e_latencies[-1]:.2f}ms"
                )

        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            print(f"[ERROR] Failed processing {file_path.name}: {e}")

    # Summary Statistics
    print("=" * 65)
    print(" EVALUATION SUMMARY REPORT")
    print("=" * 65)
    print(f" Total Files Processed: {len(test_files)}")
    print(f" Successfully Saved: {len(test_files) - len(failed_files)}")
    print(f" Failed Files: {len(failed_files)}")

    if failed_files:
        print("[FAILURES]")
        for fname, err in failed_files:
            print(f"  - {fname}: {err}")

    if model_latencies:
        mean_model = float(np.mean(model_latencies))
        median_model = float(np.median(model_latencies))
        p95_model = float(np.percentile(model_latencies, 95))

        mean_e2e = float(np.mean(e2e_latencies))
        median_e2e = float(np.median(e2e_latencies))
        p95_e2e = float(np.percentile(e2e_latencies, 95))
        throughput = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0

        print("-" * 65)
        print(" LATENCY & PERFORMANCE BENCHMARKS")
        print("-" * 65)
        print(f" Model-Only Latency: Mean={mean_model:.2f} ms | Median={median_model:.2f} ms | P95={p95_model:.2f} ms")
        print(f" End-to-End Latency: Mean={mean_e2e:.2f} ms | Median={median_e2e:.2f} ms | P95={p95_e2e:.2f} ms")
        print(f" Throughput: {throughput:.2f} images/sec (FPS)")
        print("=" * 65)


if __name__ == "__main__":
    main()
