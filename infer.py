"""
Single / Batch Inference Utility for KLA Semiconductor Image Restoration.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from evaluate import load_model
from datasets.kla_dataset import normalize_tensor, denormalize_tensor, DEFAULT_MEAN, DEFAULT_STD


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single file or directory")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy file or folder")
    parser.add_argument("--output", type=str, default="./outputs", help="Output file or folder")
    parser.add_argument("--weights", type=str, default="./checkpoints/best_psnr.pth", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def restore_array(arr: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    norm_arr = normalize_tensor(arr, DEFAULT_MEAN, DEFAULT_STD)
    tensor_in = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred_norm = model(tensor_in)
    pred_raw = denormalize_tensor(pred_norm, DEFAULT_MEAN, DEFAULT_STD)
    return pred_raw.squeeze().cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    input_path = Path(args.input)
    device = torch.device(args.device)

    model = load_model(args.weights, device)

    if input_path.is_file():
        arr = np.load(str(input_path)).astype(np.float32)
        out_arr = restore_array(arr, model, device)
        out_path = Path(args.output)
        if out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / input_path.name
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), out_arr)
        print(f"[SUCCESS] Restored {input_path.name} -> {out_path} (shape: {out_arr.shape})")
    elif input_path.is_dir():
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in input_path.iterdir() if f.suffix == ".npy" and not f.name.startswith(".")]
        for f in files:
            arr = np.load(str(f)).astype(np.float32)
            out_arr = restore_array(arr, model, device)
            np.save(str(out_dir / f.name), out_arr)
        print(f"[SUCCESS] Processed {len(files)} files to {out_dir}")


if __name__ == "__main__":
    main()
