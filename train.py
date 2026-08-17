"""
Training Entry Point for KLA Semiconductor Image Restoration (V2).
Configured for NVIDIA H100/A100 remote GPU training and local smoke validation.

Remote GPU execution command:
    python train.py --config configs/h100.yaml --device cuda
"""

import argparse
import os
from pathlib import Path
import yaml

# Optimize CUDA allocator to prevent OOM from fragmentation on shared DGX/GPU environments
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train KLA Image Restoration Network V2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v2.yaml",
        help="Path to YAML configuration file (e.g. configs/h100.yaml or configs/v2.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cuda:0', 'cpu', etc.)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run short development smoke check (1-2 real samples, 1 epoch)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (e.g. --epochs 30)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (e.g. --batch_size 16)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps (e.g. --grad_accum 2)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=None,
        help="Run validation every N epochs (e.g. --val_interval 1)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["resume_checkpoint"] = args.resume

    if args.epochs is not None:
        if "train" not in config:
            config["train"] = {}
        config["train"]["epochs"] = args.epochs

    if args.batch_size is not None:
        if "data" not in config:
            config["data"] = {}
        config["data"]["batch_size"] = args.batch_size

    if args.grad_accum is not None:
        if "train" not in config:
            config["train"] = {}
        config["train"]["gradient_accumulation"] = args.grad_accum

    if args.val_interval is not None:
        if "train" not in config:
            config["train"] = {}
        config["train"]["val_interval"] = args.val_interval

    device = args.device or ("cpu" if args.dev else config.get("device", "cuda"))

    print("=" * 65)
    print(" KLA SEMICONDUCTOR IMAGE RESTORATION — VERSION 2")
    print(f" Config: {config_path}")
    print(f" Target Device: {device}")
    print(f" Dev Mode: {args.dev}")
    print("=" * 65)

    trainer = Trainer(config=config, dev_mode=args.dev, device_override=device)
    trainer.fit()


if __name__ == "__main__":
    main()
