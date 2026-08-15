"""
PyTorch Trainer for KLA Semiconductor Image Restoration (V2).
Optimized for remote NVIDIA GPU training (H100/A100).
Handles ModelEMA, AMP, AdamW, Cosine Annealing, Curriculum Compound Loss,
Checkpoint safety with version tagging ('KLA-HYBRID-V2'),
Bicubic Baseline comparisons, and 5-sample visual diagnostics.
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.hybrid_restoration import HybridRestorationNet
from datasets.kla_dataset import KLADataset
from losses.compound_loss import CompoundLoss
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from utils.normalization import normalize, denormalize, KLA_MEAN, KLA_STD
from utils.ema import ModelEMA
from utils.logger import setup_logger, CSVLogger

MODEL_VERSION = "KLA-HYBRID-V2"


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts any string/number to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val_str)
    if match:
        try:
            return float(match.group(0))
        except Exception:
            return default
    return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely converts any string/number to integer."""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    val_str = str(val).strip()
    match = re.search(r"[-+]?\d+", val_str)
    if match:
        try:
            return int(match.group(0))
        except Exception:
            return default
    return default


def get_grad_scaler(device_type: str, enabled: bool):
    """Initializes GradScaler using modern PyTorch torch.amp API with fallback."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except Exception:
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


def get_autocast_context(device_type: str, enabled: bool):
    """Returns autocast context manager using modern PyTorch torch.amp API with fallback."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device_type, enabled=enabled)
        except Exception:
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)
    import contextlib
    return contextlib.nullcontext()


class Trainer:
    """End-to-End Trainer for KLA Hybrid Restoration Network V2."""

    def __init__(self, config: Dict[str, Any], dev_mode: bool = False, device_override: Optional[str] = None):
        self.config = config
        self.dev_mode = dev_mode

        # 1. Device configuration
        if device_override is not None:
            self.device = torch.device(device_override)
        elif torch.cuda.is_available() and str(config.get("device", "cuda")).lower().startswith("cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 2. Output and Checkpoint Directories (outputs/v2)
        self.output_dir = Path(config.get("output_dir", "./outputs/v2"))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./outputs/v2/checkpoints"))
        self.val_results_dir = Path(config.get("val_results_dir", "./outputs/v2/results"))
        self.visual_dir = Path(config.get("visual_diagnostics_dir", "./outputs/v2/visual_diagnostics"))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.val_results_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("KLA_Trainer_V2", self.output_dir / "train.log")
        self.logger.info(f"Initialized V2 Trainer on device: {self.device} (dev_mode={self.dev_mode})")

        # 3. Datasets & Dataloaders (Using ONLY official real KLA data)
        self._build_dataloaders()

        # 4. Model Construction
        model_cfg = config.get("model", {})
        base_dim = safe_int(model_cfg.get("base_dim", model_cfg.get("num_channels", 32)), default=32)
        num_groups = safe_int(model_cfg.get("num_groups", 6), default=6)
        num_naf = safe_int(model_cfg.get("naf_blocks_per_group", model_cfg.get("num_naf_per_group", 6)), default=6)
        num_swin = safe_int(model_cfg.get("swin_blocks_per_group", model_cfg.get("num_swin_per_group", 2)), default=2)
        window_size = safe_int(model_cfg.get("window_size", 8), default=8)
        swin_heads = safe_int(model_cfg.get("swin_heads", model_cfg.get("num_heads", 4)), default=4)
        naf_exp = safe_int(model_cfg.get("naf_expansion", 2), default=2)
        drop_path = safe_float(model_cfg.get("drop_path_rate", 0.0), default=0.0)
        use_sfg = bool(model_cfg.get("use_frequency_guidance", False))

        self.model = HybridRestorationNet(
            in_channels=1,
            out_channels=1,
            base_dim=base_dim,
            num_groups=num_groups,
            naf_blocks_per_group=num_naf,
            swin_blocks_per_group=num_swin,
            window_size=window_size,
            swin_heads=swin_heads,
            naf_expansion=naf_exp,
            drop_path_rate=drop_path,
            scale=2,
            use_frequency_guidance=use_sfg,
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(
            f"V2 Model built: {num_groups} groups, {num_naf} NAF/grp, {num_swin} Swin/grp, "
            f"base_dim={base_dim} | {param_count:,} trainable parameters."
        )

        # 5. Model EMA
        self.use_ema = bool(config.get("use_ema", True))
        if self.use_ema:
            self.ema = ModelEMA(
                self.model,
                decay=safe_float(config.get("ema_decay", 0.999), default=0.999),
                device=self.device,
            )
        else:
            self.ema = None

        # 6. Compound Loss & Curriculum Configuration
        loss_cfg = config.get("loss", {})
        self.w_charb = safe_float(loss_cfg.get("weight_charbonnier", 0.60), default=0.60)
        self.w_ssim = safe_float(loss_cfg.get("weight_ssim", 0.25), default=0.25)
        self.w_fft = safe_float(loss_cfg.get("weight_fft", 0.15), default=0.15)
        
        self.curriculum_epochs = safe_int(loss_cfg.get("curriculum_epochs", 2), default=2)
        self.curriculum_w_charb = safe_float(loss_cfg.get("curriculum_warmup_charbonnier", 0.80), default=0.80)
        self.curriculum_w_ssim = safe_float(loss_cfg.get("curriculum_warmup_ssim", 0.15), default=0.15)
        self.curriculum_w_fft = safe_float(loss_cfg.get("curriculum_warmup_fft", 0.05), default=0.05)

        self.criterion = CompoundLoss(
            weight_charbonnier=self.curriculum_w_charb if self.curriculum_epochs > 0 else self.w_charb,
            weight_ssim=self.curriculum_w_ssim if self.curriculum_epochs > 0 else self.w_ssim,
            weight_fft=self.curriculum_w_fft if self.curriculum_epochs > 0 else self.w_fft,
            charbonnier_eps=safe_float(loss_cfg.get("charbonnier_eps", 1e-3), default=1e-3),
            ssim_window_size=safe_int(loss_cfg.get("ssim_window_size", 11), default=11),
        ).to(self.device)

        # 7. Optimizer & Scheduler
        train_cfg = config.get("train", {})
        self.lr = safe_float(train_cfg.get("lr", 2e-4), default=2e-4)
        self.weight_decay = safe_float(train_cfg.get("weight_decay", 1e-4), default=1e-4)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.epochs = 1 if self.dev_mode else safe_int(train_cfg.get("epochs", 30), default=30)
        self.grad_accum = safe_int(train_cfg.get("gradient_accumulation", 1), default=1)
        self.val_interval = 1 if self.dev_mode else safe_int(train_cfg.get("val_interval", 1), default=1)
        self.clip_grad = safe_float(train_cfg.get("clip_grad", 1.0), default=1.0)

        effective_batch_size = self.batch_size * self.grad_accum
        self.logger.info(
            f"Training Parameters: Batch Size={self.batch_size}, Grad Accum={self.grad_accum} "
            f"=> Effective Batch Size={effective_batch_size} | Epochs={self.epochs} | Base LR={self.lr:.2e}"
        )

        self.amp_enabled = bool(config.get("amp", True)) and (self.device.type == "cuda")
        self.scaler = get_grad_scaler(self.device.type, enabled=self.amp_enabled)

        # Cosine Annealing with Warmup
        warmup_epochs = 0 if self.dev_mode else safe_int(train_cfg.get("warmup_epochs", 2), default=2)
        min_lr = safe_float(train_cfg.get("min_lr", 1e-6), default=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.epochs - warmup_epochs), eta_min=min_lr
        )
        self.warmup_epochs = warmup_epochs

        # Tracking state
        self.start_epoch = 1
        self.best_psnr = 0.0
        self.best_ssim = 0.0

        # CSV Logging
        csv_fields = [
            "epoch",
            "train_loss",
            "loss_charb",
            "loss_ssim",
            "loss_fft",
            "val_psnr",
            "val_ssim",
            "ema_psnr",
            "ema_ssim",
            "lr",
            "epoch_time_sec",
            "gpu_mem_mb",
        ]
        self.csv_logger = CSVLogger(self.output_dir / "metrics.csv", fieldnames=csv_fields)

        # 8. Resume if requested
        resume_path = config.get("resume_checkpoint")
        if resume_path:
            self._load_checkpoint(resume_path)

    def _build_dataloaders(self):
        data_cfg = self.config.get("data", {})

        lq_train_dir = data_cfg.get("lq_train_dir")
        gt_train_dir = data_cfg.get("gt_train_dir")

        val_ratio = safe_float(data_cfg.get("val_ratio", 0.1), default=0.1)
        seed = safe_int(data_cfg.get("seed", 42), default=42)
        self.batch_size = 2 if self.dev_mode else safe_int(data_cfg.get("batch_size", 32), default=32)
        num_workers = 0 if (self.dev_mode or os.name == "nt") else safe_int(data_cfg.get("num_workers", 4), default=4)
        cache_mem = bool(data_cfg.get("cache_in_memory", True)) and not self.dev_mode
        pin_memory = bool(data_cfg.get("pin_memory", True)) and (self.device.type == "cuda")

        train_ds = KLADataset(
            lq_dir=lq_train_dir,
            gt_dir=gt_train_dir,
            split="train",
            val_ratio=val_ratio,
            seed=seed,
            augment=True,
            patch_size_lq=data_cfg.get("patch_size_lq", None),
            cache_in_memory=cache_mem,
            max_samples=4 if self.dev_mode else None,
        )

        val_ds = KLADataset(
            lq_dir=lq_train_dir,
            gt_dir=gt_train_dir,
            split="val",
            val_ratio=val_ratio,
            seed=seed,
            augment=False,
            cache_in_memory=cache_mem,
            max_samples=2 if self.dev_mode else None,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=1 if self.dev_mode else min(32, len(val_ds)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.logger.info(
            f"Dataset loaded: {len(train_ds)} train samples ({len(self.train_loader)} batches), "
            f"{len(val_ds)} validation samples."
        )

    def _update_curriculum(self, epoch: int):
        """Applies loss curriculum transition."""
        if epoch <= self.curriculum_epochs:
            self.criterion.set_weights(
                weight_charbonnier=self.curriculum_w_charb,
                weight_ssim=self.curriculum_w_ssim,
                weight_fft=self.curriculum_w_fft,
            )
            stage_name = f"Stage 1 (Warmup: Charb={self.curriculum_w_charb}, SSIM={self.curriculum_w_ssim}, FFT={self.curriculum_w_fft})"
        else:
            self.criterion.set_weights(
                weight_charbonnier=self.w_charb,
                weight_ssim=self.w_ssim,
                weight_fft=self.w_fft,
            )
            stage_name = f"Stage 2 (Full Compound: Charb={self.w_charb}, SSIM={self.w_ssim}, FFT={self.w_fft})"

        if epoch == 1 or epoch == self.curriculum_epochs + 1:
            self.logger.info(f">> Active Loss Curriculum at Epoch {epoch}: {stage_name}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._update_curriculum(epoch)

        total_loss = 0.0
        total_charb = 0.0
        total_ssim = 0.0
        total_fft = 0.0
        start_time = time.time()

        # Warmup LR schedule
        if epoch <= self.warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, self.warmup_epochs))
            current_lr = self.lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            current_lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader, 1):
            lq = batch["lq"].to(self.device, non_blocking=True)
            gt = batch["gt"].to(self.device, non_blocking=True)

            with get_autocast_context(self.device.type, enabled=self.amp_enabled):
                pred = self.model(lq)
                loss, loss_dict = self.criterion(pred, gt)
                loss = loss / self.grad_accum

            if self.scaler is not None and self.amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grad_accum == 0 or step == len(self.train_loader):
                if self.scaler is not None and self.amp_enabled:
                    if self.clip_grad > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.clip_grad > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update(self.model)

            total_loss += loss_dict["total_loss"]
            total_charb += loss_dict["charbonnier"]
            total_ssim += loss_dict["ssim"]
            total_fft += loss_dict["fft"]

        if epoch > self.warmup_epochs:
            self.scheduler.step()

        num_batches = max(1, len(self.train_loader))
        avg_loss = total_loss / num_batches
        avg_charb = total_charb / num_batches
        avg_ssim = total_ssim / num_batches
        avg_fft = total_fft / num_batches

        elapsed = time.time() - start_time
        gpu_mem = f"{torch.cuda.max_memory_allocated() / (1024**2):.1f}MB" if torch.cuda.is_available() else "N/A"

        self.logger.info(
            f"Epoch [{epoch:02d}/{self.epochs:02d}] "
            f"Loss: {avg_loss:.6f} [Charb: {avg_charb:.6f} | SSIM: {avg_ssim:.6f} | FFT: {avg_fft:.6f}] | "
            f"LR: {current_lr:.2e} | Mem: {gpu_mem} | Time: {elapsed:.2f}s"
        )

        return {
            "train_loss": avg_loss,
            "loss_charb": avg_charb,
            "loss_ssim": avg_ssim,
            "loss_fft": avg_fft,
        }

    def _eval_single_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluates model on validation set in original float32 intensity domain."""
        model.eval()
        psnr_list = []
        ssim_list = []

        with torch.inference_mode():
            for batch in self.val_loader:
                lq = batch["lq"].to(self.device, non_blocking=True)
                gt = batch["gt"].to(self.device, non_blocking=True)

                with get_autocast_context(self.device.type, enabled=self.amp_enabled):
                    pred = model(lq)

                # Denormalize both prediction and target to original float32 domain
                pred_raw = denormalize(pred)
                gt_raw = denormalize(gt)

                # Compute metrics on original intensity domain per sample
                for b in range(pred_raw.shape[0]):
                    p_img = pred_raw[b : b + 1]
                    g_img = gt_raw[b : b + 1]
                    dr = max(0.1, (g_img.max() - g_img.min()).item())

                    psnr = calculate_psnr(p_img, g_img, data_range=dr)
                    ssim = calculate_ssim(p_img, g_img, data_range=dr)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

        return float(np.mean(psnr_list)), float(np.mean(ssim_list))

    def validate(self, epoch: int) -> Dict[str, float]:
        """Evaluates both standard and EMA model on the validation split."""
        # Standard model evaluation
        norm_psnr, norm_ssim = self._eval_single_model(self.model)

        # EMA model evaluation
        if self.ema is not None:
            ema_psnr, ema_ssim = self._eval_single_model(self.ema.module)
        else:
            ema_psnr, ema_ssim = norm_psnr, norm_ssim

        self.logger.info(
            f"--- Val Epoch {epoch:02d} --- "
            f"Model PSNR: {norm_psnr:.4f} dB, SSIM: {norm_ssim:.4f} | "
            f"EMA PSNR: {ema_psnr:.4f} dB, SSIM: {ema_ssim:.4f}"
        )

        # Generate visual diagnostics for 5 fixed real validation samples
        self._generate_visual_diagnostics(epoch)

        return {
            "val_psnr": norm_psnr,
            "val_ssim": norm_ssim,
            "ema_psnr": ema_psnr,
            "ema_ssim": ema_ssim,
        }

    def _generate_visual_diagnostics(self, epoch: int, num_samples: int = 5):
        """Saves visual diagnostics (LQ, Bicubic, V2 Output, GT + stats) for fixed samples."""
        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.eval()

        diagnostics_records = []
        count = 0

        with torch.inference_mode():
            for batch in self.val_loader:
                lq = batch["lq"].to(self.device)
                gt = batch["gt"].to(self.device)
                filenames = batch["filename"]

                with get_autocast_context(self.device.type, enabled=self.amp_enabled):
                    pred = eval_model(lq)

                pred_raw = denormalize(pred)
                gt_raw = denormalize(gt)
                lq_raw = denormalize(lq)

                bicubic_raw = F.interpolate(
                    lq_raw, scale_factor=2, mode="bicubic", align_corners=False
                )

                for b in range(lq.shape[0]):
                    if count >= num_samples:
                        break
                    fname = filenames[b]
                    p_img = pred_raw[b : b + 1]
                    g_img = gt_raw[b : b + 1]
                    b_img = bicubic_raw[b : b + 1]
                    l_img = lq_raw[b : b + 1]

                    dr = max(0.1, (g_img.max() - g_img.min()).item())
                    model_psnr = calculate_psnr(p_img, g_img, data_range=dr)
                    model_ssim = calculate_ssim(p_img, g_img, data_range=dr)
                    bic_psnr = calculate_psnr(b_img, g_img, data_range=dr)
                    bic_ssim = calculate_ssim(b_img, g_img, data_range=dr)

                    p_np = p_img.squeeze().cpu().numpy().astype(np.float32)
                    g_np = g_img.squeeze().cpu().numpy().astype(np.float32)
                    b_np = b_img.squeeze().cpu().numpy().astype(np.float32)
                    l_np = l_img.squeeze().cpu().numpy().astype(np.float32)

                    # Save actual arrays for epoch inspection
                    sample_dir = self.visual_dir / f"epoch_{epoch:03d}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(fname).stem
                    np.save(str(sample_dir / f"{stem}_pred.npy"), p_np)
                    np.save(str(sample_dir / f"{stem}_bicubic.npy"), b_np)
                    np.save(str(sample_dir / f"{stem}_gt.npy"), g_np)
                    np.save(str(sample_dir / f"{stem}_lq.npy"), l_np)

                    diagnostics_records.append(
                        {
                            "filename": fname,
                            "pred_mean": float(np.mean(p_np)),
                            "pred_std": float(np.std(p_np)),
                            "pred_min": float(np.min(p_np)),
                            "pred_max": float(np.max(p_np)),
                            "gt_mean": float(np.mean(g_np)),
                            "gt_std": float(np.std(g_np)),
                            "gt_min": float(np.min(g_np)),
                            "gt_max": float(np.max(g_np)),
                            "model_psnr": model_psnr,
                            "model_ssim": model_ssim,
                            "bicubic_psnr": bic_psnr,
                            "bicubic_ssim": bic_ssim,
                        }
                    )
                    count += 1

                if count >= num_samples:
                    break

        # Save diagnostic stats JSON
        diag_path = self.visual_dir / f"diagnostics_epoch_{epoch:03d}.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics_records, f, indent=2)

    def calculate_bicubic_baseline(self) -> Dict[str, float]:
        """Calculates exact Bicubic baseline on the entire validation split."""
        self.logger.info("Evaluating Bicubic baseline on validation split...")
        psnr_list = []
        ssim_list = []

        with torch.inference_mode():
            for batch in self.val_loader:
                lq = batch["lq"].to(self.device, non_blocking=True)
                gt = batch["gt"].to(self.device, non_blocking=True)

                # Denormalize to original domain
                lq_raw = denormalize(lq)
                gt_raw = denormalize(gt)

                bicubic_raw = F.interpolate(
                    lq_raw, scale_factor=2, mode="bicubic", align_corners=False
                )

                for b in range(lq_raw.shape[0]):
                    b_img = bicubic_raw[b : b + 1]
                    g_img = gt_raw[b : b + 1]
                    dr = max(0.1, (g_img.max() - g_img.min()).item())

                    psnr = calculate_psnr(b_img, g_img, data_range=dr)
                    ssim = calculate_ssim(b_img, g_img, data_range=dr)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

        bic_psnr = float(np.mean(psnr_list))
        bic_ssim = float(np.mean(ssim_list))

        self.logger.info(f">> Bicubic Validation Baseline: PSNR = {bic_psnr:.6f} dB | SSIM = {bic_ssim:.6f}")

        # Save to baselines.csv
        baseline_csv = self.val_results_dir / "baselines.csv"
        with open(baseline_csv, "w") as f:
            f.write("model,psnr,ssim\n")
            f.write(f"bicubic,{bic_psnr:.6f},{bic_ssim:.6f}\n")

        return {"bicubic_psnr": bic_psnr, "bicubic_ssim": bic_ssim}

    def _save_checkpoint(self, epoch: int, psnr: float, ssim: float, is_best: bool = False, is_best_ssim: bool = False):
        """Saves checkpoint containing full state dict and model_version tag."""
        state = {
            "model_version": MODEL_VERSION,
            "epoch": epoch,
            "best_psnr": self.best_psnr,
            "best_ssim": self.best_ssim,
            "current_psnr": psnr,
            "current_ssim": ssim,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }

        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(state, str(latest_path))

        if is_best:
            best_psnr_path = self.checkpoint_dir / "best_psnr.pth"
            torch.save(state, str(best_psnr_path))
            self.logger.info(f">> Saved new BEST PSNR checkpoint to {best_psnr_path} (PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f})")

        if is_best_ssim:
            best_ssim_path = self.checkpoint_dir / "best_ssim.pth"
            torch.save(state, str(best_ssim_path))
            self.logger.info(f">> Saved new BEST SSIM checkpoint to {best_ssim_path} (PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f})")

    def _load_checkpoint(self, checkpoint_path: str):
        """Loads checkpoint with strict version validation."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        ckpt_version = ckpt.get("model_version")
        if ckpt_version != MODEL_VERSION:
            raise ValueError(
                f"Incompatible checkpoint version: '{ckpt_version}', expected '{MODEL_VERSION}'. "
                f"Loading legacy V1 or non-V2 checkpoints is strictly prohibited."
            )

        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.ema is not None and ckpt.get("ema_state_dict") is not None:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_psnr = ckpt.get("best_psnr", 0.0)
        self.best_ssim = ckpt.get("best_ssim", 0.0)
        self.logger.info(f"Resumed from epoch {self.start_epoch} (Best PSNR: {self.best_psnr:.4f} dB)")

    def fit(self):
        """Main training loop."""
        self.logger.info(f"Starting KLA V2 Training for {self.epochs} epochs...")

        # Calculate initial Bicubic baseline
        self.calculate_bicubic_baseline()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)

            should_val = (epoch % self.val_interval == 0) or (epoch == self.epochs) or self.dev_mode
            val_psnr = None
            val_ssim = None
            ema_psnr = None
            ema_ssim = None

            if should_val:
                val_metrics = self.validate(epoch)
                val_psnr = val_metrics["val_psnr"]
                val_ssim = val_metrics["val_ssim"]
                ema_psnr = val_metrics["ema_psnr"]
                ema_ssim = val_metrics["ema_ssim"]

                # Use better of EMA or model for best checkpoint
                eval_psnr = max(val_psnr, ema_psnr)
                eval_ssim = max(val_ssim, ema_ssim)

                is_best_psnr = eval_psnr > self.best_psnr
                is_best_ssim = eval_ssim > self.best_ssim

                if is_best_psnr:
                    self.best_psnr = eval_psnr
                if is_best_ssim:
                    self.best_ssim = eval_ssim

                self._save_checkpoint(epoch, eval_psnr, eval_ssim, is_best=is_best_psnr, is_best_ssim=is_best_ssim)
            else:
                self._save_checkpoint(epoch, self.best_psnr, self.best_ssim, is_best=False)

            epoch_time = time.time() - epoch_start
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

            self.csv_logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["train_loss"],
                    "loss_charb": train_metrics["loss_charb"],
                    "loss_ssim": train_metrics["loss_ssim"],
                    "loss_fft": train_metrics["loss_fft"],
                    "val_psnr": val_psnr if val_psnr is not None else "",
                    "val_ssim": val_ssim if val_ssim is not None else "",
                    "ema_psnr": ema_psnr if ema_psnr is not None else "",
                    "ema_ssim": ema_ssim if ema_ssim is not None else "",
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_time_sec": epoch_time,
                    "gpu_mem_mb": f"{gpu_mem:.1f}",
                }
            )

        # Record final V2 baseline in baselines.csv
        baseline_csv = self.val_results_dir / "baselines.csv"
        with open(baseline_csv, "a") as f:
            f.write(f"v2_model,{self.best_psnr:.6f},{self.best_ssim:.6f}\n")

        self.logger.info(
            f"V2 Training Complete! Best PSNR: {self.best_psnr:.4f} dB | Best SSIM: {self.best_ssim:.4f}"
        )
