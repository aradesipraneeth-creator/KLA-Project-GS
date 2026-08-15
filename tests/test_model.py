"""
Model Forward Test using ACTUAL KLA Data Samples.
Strictly zero synthetic images.
Loads 1 real Noisy_LR file, passes through HybridRestorationNet,
and validates 2x super-resolved output tensor dimensions and values.
"""

import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datasets.kla_dataset import KLADataset
from models.hybrid_restoration import HybridRestorationNet


def test_hybrid_model_real_kla_sample():
    lq_dir = repo_root / "train" / "train" / "NoisyLR"
    gt_dir = repo_root / "train" / "train" / "GT"
    if not lq_dir.exists():
        lq_dir = repo_root / "Train" / "Train" / "Noisy_LR"
    if not gt_dir.exists():
        gt_dir = repo_root / "Train" / "Train" / "GT"

    print(f"[TEST] Loading real sample from {lq_dir}...")
    dataset = KLADataset(lq_dir=lq_dir, gt_dir=gt_dir, split="train", augment=False, normalize=True)
    sample = dataset[0]
    lq = sample["lq"].unsqueeze(0)  # [1, 1, 128, 128]
    gt = sample["gt"].unsqueeze(0)  # [1, 1, 256, 256]
    fname = sample["filename"]

    print(f"[TEST] Loaded real sample '{fname}': LQ shape {lq.shape}, GT shape {gt.shape}")

    # Build model (for CPU test, channels=32 or 48)
    model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=32,
        num_groups=4,
        num_naf_per_group=4,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TEST] Model parameters: {num_params:,}")

    with torch.no_grad():
        out = model(lq)

    print(f"[TEST] Output shape: {out.shape}, dtype: {out.dtype}")
    assert out.shape == (1, 1, 256, 256), f"Expected output (1, 1, 256, 256), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf!"
    assert out.dtype == torch.float32

    # Also test full standard architecture (num_channels=48, num_groups=6, num_naf_per_group=6)
    full_model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=48,
        num_groups=6,
        num_naf_per_group=6,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
    )
    full_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"[TEST] Full Model parameters: {full_params:,}")

    with torch.no_grad():
        full_out = full_model(lq)
    assert full_out.shape == (1, 1, 256, 256)
    assert torch.isfinite(full_out).all()

    print("[SUCCESS] Phase 3, 4, 5, 6 Model forward tests passed on REAL KLA data!")


if __name__ == "__main__":
    test_hybrid_model_real_kla_sample()
