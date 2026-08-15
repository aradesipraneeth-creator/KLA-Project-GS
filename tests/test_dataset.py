"""
Unit test for KLADataset using actual KLA samples.
Verifies shapes, dtypes, pairing, and normalization with zero synthetic data.
"""

import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datasets.kla_dataset import KLADataset, DEFAULT_MEAN, DEFAULT_STD, denormalize_tensor


def test_kla_dataset_real_samples():
    lq_dir = repo_root / "train" / "train" / "NoisyLR"
    gt_dir = repo_root / "train" / "train" / "GT"

    if not lq_dir.exists():
        lq_dir = repo_root / "Train" / "Train" / "Noisy_LR"
    if not gt_dir.exists():
        gt_dir = repo_root / "Train" / "Train" / "GT"

    print(f"[TEST] Using LQ dir: {lq_dir}")
    print(f"[TEST] Using GT dir: {gt_dir}")

    # Initialize train and val datasets
    train_ds = KLADataset(lq_dir=lq_dir, gt_dir=gt_dir, split="train", val_ratio=0.1, seed=42, augment=True)
    val_ds = KLADataset(lq_dir=lq_dir, gt_dir=gt_dir, split="val", val_ratio=0.1, seed=42, augment=False)

    print(f"[TEST] Total train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    assert len(train_ds) > 0, "Train dataset is empty!"
    assert len(val_ds) > 0, "Validation dataset is empty!"
    assert set(train_ds.filenames).isdisjoint(set(val_ds.filenames)), "Train and Val split have overlap!"

    # Test loading first 5 samples from train_ds
    for i in range(min(5, len(train_ds))):
        sample = train_ds[i]
        lq = sample["lq"]
        gt = sample["gt"]
        fname = sample["filename"]

        assert isinstance(lq, torch.Tensor), f"LQ is not a torch.Tensor for {fname}"
        assert isinstance(gt, torch.Tensor), f"GT is not a torch.Tensor for {fname}"
        assert lq.shape == (1, 128, 128), f"Expected LQ shape (1, 128, 128), got {lq.shape}"
        assert gt.shape == (1, 256, 256), f"Expected GT shape (1, 256, 256), got {gt.shape}"
        assert lq.dtype == torch.float32, f"Expected float32, got {lq.dtype}"
        assert gt.dtype == torch.float32, f"Expected float32, got {gt.dtype}"
        assert torch.isfinite(lq).all(), f"NaN or Inf found in LQ {fname}"
        assert torch.isfinite(gt).all(), f"NaN or Inf found in GT {fname}"

        # Check denormalization consistency
        lq_raw = denormalize_tensor(lq, DEFAULT_MEAN, DEFAULT_STD)
        gt_raw = denormalize_tensor(gt, DEFAULT_MEAN, DEFAULT_STD)
        assert torch.isfinite(lq_raw).all()
        assert torch.isfinite(gt_raw).all()

        print(f"[TEST] Verified sample {i} ({fname}): LQ min={lq.min():.4f}, max={lq.max():.4f} | GT min={gt.min():.4f}, max={gt.max():.4f}")

    # Test Test_NoisyLR dataset loading
    test_dir = repo_root / "Test_NoisyLR" / "NoisyLR"
    test_ds = KLADataset(lq_dir=test_dir, gt_dir=None, split="test", normalize=True)
    print(f"[TEST] Total official test samples: {len(test_ds)}")
    assert len(test_ds) == 400, f"Expected 400 test samples, found {len(test_ds)}"

    test_sample = test_ds[0]
    assert test_sample["lq"].shape == (1, 128, 128)
    assert "gt" not in test_sample

    print("[SUCCESS] Phase 1 & 2 Dataset tests passed on REAL KLA data!")


if __name__ == "__main__":
    test_kla_dataset_real_samples()
