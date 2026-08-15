# KLA Semiconductor Image Restoration — Version 2 (KLA-HYBRID-V2)

An end-to-end deep learning framework designed specifically for KLA semiconductor wafer image restoration, simultaneously tackling **Speckle Noise**, **Gaussian Noise**, and **2× Super-Resolution** ($128 \times 128 \rightarrow 256 \times 256$) while preserving full float32 dynamic range and delicate semiconductor circuit patterns.

---

## 🏛️ Architecture Overview

The system employs a **Hybrid Restoration Network (V2)** combining convolution-based Nonlinear Activation Free (NAF) blocks and attention-based Swin Transformer blocks with PixelShuffle upsampling and global bicubic residual guidance.

```
Input LQ (128×128, float32)
  │
  ├──► [Bicubic Upsample ×2 (Original Domain)] ────────────────────────┐
  │                                                                    │ (Global Residual)
  └──► [Normalization: mean=0.4335, std=0.2848]                        │
         │                                                             │
       [3×3 Intro Conv]                                                │
         │                                                             │
       [Residual Conv Block]                                           │
         │                                                             │
       [6× Hybrid Restoration Groups]                                  │
         │  Each Group:                                                │
         │  ├── 6× NAF Blocks (LayerNorm2D + DWConv + SimpleGate + SCA)│
         │  ├── 2× Swin Blocks (W-MSA + SW-MSA + RelPosBias + MLP)     │
         │  ├── (Optional) Structure-Frequency Guidance (SFG)          │
         │  └── Local Residual Connection                              │
         │                                                             │
       [Reconstruction Conv (C -> 4C)]                                 │
         │                                                             │
       [PixelShuffle ×2] (128×128 -> 256×256)                          │
         │                                                             │
       [Final 3×3 Conv]                                                │
         │                                                             │
         ▼                                                             ▼
     [Network Residual] ──────────────────────────────────────────► [+] ──► Restored Output (256×256, float32)
```

---

## 📂 Dataset Setup & Local Placement

> [!IMPORTANT]
> **Proprietary Dataset Exclusion**: In strict accordance with competition rules, the official KLA dataset files (`.npy`) are **NOT** committed or distributed via GitHub.

To run training or evaluation locally or on a remote compute instance, place the official dataset into the project root following this directory layout:

```
KLA FINAL/
├── Train/
│   └── train/ (or Train/)
│       ├── NoisyLR/ (or Noisy_LR/)   # 3,200 float32 .npy arrays [128×128]
│       └── GT/                       # 3,200 float32 .npy arrays [256×256]
└── Test_NoisyLR/
    └── NoisyLR/                      # 400 float32 .npy arrays [128×128]
```

### Verified Dataset Statistics:
- **Input (NoisyLR)**: $128 \times 128$, float32 grayscale
- **Target (GT)**: $256 \times 256$, float32 grayscale
- **Dataset Mean**: `0.43353602`
- **Dataset Std**: `0.28478748`
- **Split**: 90% Train (2,880 pairs) / 10% Validation (320 pairs), Seed = 42
- **Zero Synthetic Data Policy**: Float32 values are never clamped to $[0, 1]$, never converted to uint8, and never artificially fabricated.

---

## 🎯 Authoritative Compound Loss (V2)

$$\mathcal{L}_{\text{total}} = 0.60 \times \mathcal{L}_{\text{Charbonnier}} + 0.25 \times \mathcal{L}_{\text{SSIM}} + 0.15 \times \mathcal{L}_{\text{FFT}}$$

- **Charbonnier Loss**: Robust smooth L1 pixel reconstruction.
- **SSIM Loss**: Structural similarity computed with correct image dynamic range.
- **FFT Loss**: 2D Orthonormal real FFT loss for high-frequency edge fidelity without energy blowup.

---

## 🚀 Execution Commands

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Local Preflight Integrity Check (Zero Local Training)
```bash
python scripts/preflight_check.py
```
*Runs an 11-step verification suite (imports, topology, shapes, forward, compound loss, backward, checkpoint serialization, and evaluation arguments).*

### 3. Remote GPU Training (NVIDIA H100 / A100 / RTX 4090)
```bash
python train.py --config configs/h100.yaml --device cuda
```
*Trains V2 with EMA (0.999), AMP, Cosine Annealing, 30 epochs, saving checkpoints to `outputs/v2/checkpoints/` (`latest.pth`, `best_psnr.pth`, `best_ssim.pth`).*

### 4. Official 400-Image Test Set Evaluation
```bash
python evaluate.py \
  --input_dir ./Test_NoisyLR/NoisyLR \
  --output_dir ./outputs/v2/test_outputs \
  --weights ./outputs/v2/checkpoints/best_psnr.pth \
  --device cuda \
  --amp
```
*Processes all 400 real test samples, generates $256 \times 256$ float32 `.npy` arrays in `outputs/v2/test_outputs/`, verifies shapes, and outputs latency/throughput benchmarks.*

### 5. Interactive Streamlit Dashboard (Optional Demonstration UI)
```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```
*Provides an 8-tab interactive visualization, single-image restoration inspector, error heatmaps, and batch test runner for judges and reviewers. Note: The Streamlit dashboard is an optional presentation layer and is NOT part of the official benchmark path.*

---

## 📁 Repository Structure

```
KLA FINAL/
├── datasets/
│   ├── kla_dataset.py        # Real KLA dataset loader, paired augmentations, split
│   └── __init__.py
├── models/
│   ├── naf_block.py          # NAFBlock (LayerNorm2D, DWConv, SimpleGate, SCA)
│   ├── swin_block.py         # SwinTransformerBlock (W-MSA, SW-MSA, RelPosBias)
│   ├── frequency_guidance.py # Structure-Frequency Guidance (SFG)
│   ├── hybrid_restoration.py # 6-Group Hybrid Architecture + 2x PixelShuffle + Global Residual
│   └── __init__.py
├── losses/
│   ├── charbonnier.py        # Charbonnier Loss
│   ├── ssim.py               # Differentiable float32 SSIM Loss
│   ├── fft_loss.py           # 2D Orthonormal FFT Frequency Loss
│   ├── compound_loss.py      # Authoritative Compound Loss (0.60 Charb + 0.25 SSIM + 0.15 FFT)
│   └── __init__.py
├── metrics/
│   ├── psnr_ssim.py          # Real float32 PSNR & SSIM metrics (original domain)
│   └── __init__.py
├── trainer/
│   ├── trainer.py            # Trainer with AdamW, Cosine, EMA, AMP, Checkpointing
│   └── __init__.py
├── utils/
│   ├── normalization.py      # Unified normalize() / denormalize() module
│   ├── ema.py                # Model Exponential Moving Average
│   ├── logger.py             # Console & CSV Metric Logger
│   └── __init__.py
├── configs/
│   ├── v2.yaml               # Canonical V2 configuration
│   ├── h100.yaml             # H100 Remote GPU training profile
│   └── local_check.yaml      # Fast local smoke check profile
├── dashboard/
│   ├── app.py                # 8-tab Streamlit dashboard interface
│   ├── components.py         # Architecture cards & stats formatters
│   ├── inference.py          # Model loading & inference engine
│   ├── visualization.py      # Multi-panel plots, error heatmaps, FFT spectra
│   └── __init__.py
├── scripts/
│   └── preflight_check.py    # 11-step automated local preflight check
├── app.py                    # Root proxy for Streamlit
├── train.py                  # Training entry point
├── evaluate.py               # Official test evaluation script (400 files)
├── infer.py                  # Single / batch inference tool
├── benchmark.py              # Latency & throughput benchmarking tool
├── requirements.txt          # Core dependencies
├── requirements-dashboard.txt# Optional dashboard dependencies
├── .gitignore                # Strict exclusion of dataset, checkpoints, secrets
└── README.md
```
