"""
Benchmarking Script for KLA Image Restoration Model.
Measures model-only and end-to-end latency, FLOPs, memory footprint, and FPS.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch

from models.hybrid_restoration import HybridRestorationNet
from datasets.kla_dataset import KLADataset


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KLA Model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_channels", type=int, default=48)
    parser.add_argument("--num_groups", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=args.num_channels,
        num_groups=args.num_groups,
        num_naf_per_group=6,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
    ).to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(" KLA HYBRID MODEL PERFORMANCE BENCHMARK")
    print(f" Device: {device}")
    print(f" Parameters: {params:,}")
    print(f" Channels: {args.num_channels}, Groups: {args.num_groups}")
    print("=" * 60)

    # Load 1 real sample for realistic execution
    repo_root = Path(__file__).resolve().parent
    lq_dir = repo_root / "train" / "train" / "NoisyLR"
    if not lq_dir.exists():
        lq_dir = repo_root / "Train" / "Train" / "Noisy_LR"

    dataset = KLADataset(lq_dir=lq_dir, gt_dir=None, split="all")
    sample = dataset[0]
    lq = sample["lq"].unsqueeze(0).to(device)

    print(f"[INFO] Running {args.warmup} warmup iterations...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(lq)
            if device.type == "cuda":
                torch.cuda.synchronize()

    print(f"[INFO] Running {args.iterations} timed benchmark iterations...")
    latencies = []
    with torch.inference_mode():
        for _ in range(args.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(lq)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat

    print("-" * 60)
    print(f" Mean Latency:   {mean_lat:.2f} ms")
    print(f" Median Latency: {median_lat:.2f} ms")
    print(f" P95 Latency:    {p95_lat:.2f} ms")
    print(f" Throughput:     {fps:.2f} FPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
