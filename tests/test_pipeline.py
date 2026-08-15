"""
Complete End-to-End Pipeline Integration Test.
Runs on CPU using REAL KLA samples:
1. Load real LQ and GT samples
2. Normalize
3. Forward pass through HybridRestorationNet
4. Compute CompoundLoss
5. Backward pass and Optimizer Step
6. ModelEMA update
7. Save checkpoint
8. Load checkpoint and run evaluation inference
9. Denormalize and check 256x256 float32 output
"""

import sys
import tempfile
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datasets.kla_dataset import KLADataset, denormalize_tensor, DEFAULT_MEAN, DEFAULT_STD
from models.hybrid_restoration import HybridRestorationNet
from losses.compound_loss import CompoundLoss
from utils.ema import ModelEMA
from metrics.psnr_ssim import calculate_psnr, calculate_ssim


def test_full_pipeline_real_sample():
    print("[TEST] Starting Full Pipeline Integration Test...")
    lq_dir = repo_root / "train" / "train" / "NoisyLR"
    gt_dir = repo_root / "train" / "train" / "GT"
    if not lq_dir.exists():
        lq_dir = repo_root / "Train" / "Train" / "Noisy_LR"
    if not gt_dir.exists():
        gt_dir = repo_root / "Train" / "Train" / "GT"

    dataset = KLADataset(lq_dir=lq_dir, gt_dir=gt_dir, split="train", augment=True, normalize=True)
    sample = dataset[0]
    lq = sample["lq"].unsqueeze(0)  # [1, 1, 128, 128]
    gt = sample["gt"].unsqueeze(0)  # [1, 1, 256, 256]

    # 1. Instantiate Model
    model = HybridRestorationNet(
        in_channels=1,
        out_channels=1,
        num_channels=24,
        num_groups=2,
        num_naf_per_group=2,
        num_swin_per_group=2,
        window_size=8,
        num_heads=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = ModelEMA(model, decay=0.99)
    criterion = CompoundLoss()

    # 2. Forward pass
    pred = model(lq)
    assert pred.shape == (1, 1, 256, 256)

    # 3. Loss & Backward
    loss, loss_dict = criterion(pred, gt)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    ema.update(model)

    print(f"[TEST] Forward & Backward step passed. Loss: {loss.item():.4f}")

    # 4. Checkpoint Save & Load
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "epoch": 1,
                "best_psnr": 28.5,
            },
            str(ckpt_path),
        )

        loaded_ckpt = torch.load(str(ckpt_path))
        eval_model = HybridRestorationNet(
            in_channels=1,
            out_channels=1,
            num_channels=24,
            num_groups=2,
            num_naf_per_group=2,
            num_swin_per_group=2,
            window_size=8,
            num_heads=4,
        )
        eval_model.load_state_dict(loaded_ckpt["ema_state_dict"])
        eval_model.eval()

        # 5. Inference
        with torch.inference_mode():
            eval_pred = eval_model(lq)
            pred_raw = denormalize_tensor(eval_pred, DEFAULT_MEAN, DEFAULT_STD)
            gt_raw = denormalize_tensor(gt, DEFAULT_MEAN, DEFAULT_STD)

            dr = max(0.1, (gt_raw.max() - gt_raw.min()).item())
            psnr = calculate_psnr(pred_raw, gt_raw, data_range=dr)
            ssim_val = calculate_ssim(pred_raw, gt_raw, data_range=dr)

            print(f"[TEST] Checkpoint evaluation: PSNR={psnr:.4f} dB, SSIM={ssim_val:.4f}")
            assert pred_raw.shape == (1, 1, 256, 256)
            assert torch.isfinite(pred_raw).all()

    print("[SUCCESS] Complete End-to-End Pipeline Integration Test passed on REAL KLA data!")


if __name__ == "__main__":
    test_full_pipeline_real_sample()
