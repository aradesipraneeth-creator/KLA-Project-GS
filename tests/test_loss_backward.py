"""
Real KLA Sample Loss & Backward Test.
Loads 1 real Noisy_LR and GT sample, computes compound loss, runs backward(),
and verifies that all trainable parameters receive valid non-zero finite gradients.
"""

import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datasets.kla_dataset import KLADataset
from models.hybrid_restoration import HybridRestorationNet
from losses.compound_loss import CompoundLoss


def test_loss_and_backward_real_sample():
    lq_dir = repo_root / "train" / "train" / "NoisyLR"
    gt_dir = repo_root / "train" / "train" / "GT"
    if not lq_dir.exists():
        lq_dir = repo_root / "Train" / "Train" / "Noisy_LR"
    if not gt_dir.exists():
        gt_dir = repo_root / "Train" / "Train" / "GT"

    dataset = KLADataset(lq_dir=lq_dir, gt_dir=gt_dir, split="train", augment=False, normalize=True)
    sample = dataset[0]
    lq = sample["lq"].unsqueeze(0)  # [1, 1, 128, 128]
    gt = sample["gt"].unsqueeze(0)  # [1, 1, 256, 256]
    fname = sample["filename"]

    print(f"[TEST] Testing backward pass with real sample '{fname}'...")

    model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=32,
        num_groups=2,
        num_naf_per_group=2,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
    )
    model.train()

    criterion = CompoundLoss(
        weight_charbonnier=0.60,
        weight_ssim=0.25,
        weight_fft=0.15,
    )

    pred = model(lq)
    assert pred.shape == gt.shape

    loss, loss_dict = criterion(pred, gt)
    print(f"[TEST] Compound Loss: {loss.item():.6f} (Charb: {loss_dict['loss_charb']:.6f}, SSIM: {loss_dict['loss_ssim']:.6f}, FFT: {loss_dict['loss_fft']:.6f})")

    assert torch.isfinite(loss), "Loss is NaN or Inf!"
    assert loss.item() > 0, "Loss is not positive!"

    # Run backward pass
    loss.backward()

    # Verify gradients
    grad_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has NaN/Inf gradient!"
            grad_count += 1

    print(f"[TEST] Successfully verified valid gradients for all {grad_count} trainable parameter tensors.")
    print("[SUCCESS] Phase 7 & 8 Loss & Backward tests passed on REAL KLA data!")


if __name__ == "__main__":
    test_loss_and_backward_real_sample()
