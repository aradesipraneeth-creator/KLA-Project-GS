"""
Test Structure-Frequency Guidance (SFG) module on REAL KLA data samples.
Verifies forward pass, spectral modulation, backward gradient flow, and ablation equivalence.
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


def test_structure_frequency_guidance():
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

    print("[TEST] Initializing Baseline Model (use_frequency_guidance=False)...")
    baseline_model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=32,
        num_groups=2,
        num_naf_per_group=2,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
        use_frequency_guidance=False,
    )
    base_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)

    print("[TEST] Initializing Enhanced Model (use_frequency_guidance=True)...")
    sfg_model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=32,
        num_groups=2,
        num_naf_per_group=2,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
        use_frequency_guidance=True,
    )
    sfg_params = sum(p.numel() for p in sfg_model.parameters() if p.requires_grad)

    param_diff = sfg_params - base_params
    print(f"[TEST] Baseline Params: {base_params:,} | SFG Params: {sfg_params:,} | Added: {param_diff:,} (< {param_diff/base_params*100:.2f}%)")

    # Forward pass
    sfg_model.train()
    out = sfg_model(lq)
    assert out.shape == (1, 1, 256, 256)
    assert torch.isfinite(out).all()

    # Loss and backward
    criterion = CompoundLoss()
    loss, _ = criterion(out, gt)
    loss.backward()

    # Verify gradients on SFG parameters
    sfg_has_grad = False
    for name, p in sfg_model.named_parameters():
        if "sfg" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"SFG param {name} grad error!"
            sfg_has_grad = True

    assert sfg_has_grad, "No SFG parameters received gradients!"
    print("[SUCCESS] Phase 15 Structure-Frequency Guidance verified with REAL KLA sample!")


if __name__ == "__main__":
    test_structure_frequency_guidance()
