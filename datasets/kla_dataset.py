"""
Official KLA Semiconductor Image Restoration Dataset Module (V2).
Loads real float32 .npy arrays for Low-Quality (128x128) and Ground Truth (256x256).
Strictly ignores __MACOSX and hidden files.
Preserves float32 precision, performs paired geometric augmentations,
and uses consistent normalization from utils.normalization.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.normalization import (
    normalize as normalize_fn,
    denormalize as denormalize_fn,
    KLA_MEAN,
    KLA_STD,
)


def resolve_dataset_dir(
    dir_path: Optional[Union[str, Path]],
    target_type: str = "lq_train",  # "lq_train", "gt_train", "test"
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Resolves directory path across Windows and Linux environments,
    handling case variations ('Train' vs 'train'), nesting ('Train/Train' vs 'Train'),
    and naming variations ('NoisyLR' vs 'Noisy_LR').
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    # 1. Try provided path directly if specified
    if dir_path is not None:
        p = Path(dir_path)
        if p.is_dir() and any(f.suffix.lower() == ".npy" for f in p.iterdir() if not f.name.startswith(".")):
            return p.resolve()
        p_repo = repo_root / dir_path
        if p_repo.is_dir() and any(f.suffix.lower() == ".npy" for f in p_repo.iterdir() if not f.name.startswith(".")):
            return p_repo.resolve()

    # 2. Comprehensive candidate list based on target_type
    candidates = []
    if target_type in ["lq_train", "lq"]:
        candidates = [
            repo_root / "Train" / "train" / "NoisyLR",
            repo_root / "Train" / "train" / "Noisy_LR",
            repo_root / "Train" / "Train" / "NoisyLR",
            repo_root / "Train" / "Train" / "Noisy_LR",
            repo_root / "train" / "train" / "NoisyLR",
            repo_root / "train" / "train" / "Noisy_LR",
            repo_root / "Train" / "NoisyLR",
            repo_root / "Train" / "Noisy_LR",
            repo_root / "train" / "NoisyLR",
            repo_root / "train" / "Noisy_LR",
            repo_root / "NoisyLR",
            repo_root / "Noisy_LR",
        ]
    elif target_type in ["gt_train", "gt"]:
        candidates = [
            repo_root / "Train" / "train" / "GT",
            repo_root / "Train" / "train" / "gt",
            repo_root / "Train" / "Train" / "GT",
            repo_root / "Train" / "Train" / "gt",
            repo_root / "train" / "train" / "GT",
            repo_root / "train" / "train" / "gt",
            repo_root / "Train" / "GT",
            repo_root / "Train" / "gt",
            repo_root / "train" / "GT",
            repo_root / "train" / "gt",
            repo_root / "GT",
            repo_root / "gt",
        ]
    elif target_type in ["test", "lq_test"]:
        candidates = [
            repo_root / "Test_NoisyLR" / "NoisyLR",
            repo_root / "Test_NoisyLR" / "Noisy_LR",
            repo_root / "test_noisylr" / "noisylr",
            repo_root / "Test_Noisy_LR" / "Noisy_LR",
            repo_root / "Test_Noisy_LR" / "NoisyLR",
            repo_root / "test" / "NoisyLR",
            repo_root / "Test" / "NoisyLR",
        ]

    for cand in candidates:
        if cand.is_dir():
            valid_files = [
                f
                for f in cand.iterdir()
                if f.suffix.lower() == ".npy"
                and not f.name.startswith(".")
                and not f.name.startswith("._")
                and "__MACOSX" not in str(f)
            ]
            if len(valid_files) > 0:
                return cand.resolve()

    # 3. Dynamic recursive search across repo_root as fallback
    for root_dir, dirs, files in os.walk(str(repo_root)):
        if "__MACOSX" in root_dir:
            continue
        p = Path(root_dir)
        folder_lower = p.name.lower()
        if target_type in ["lq_train", "lq"] and folder_lower in ["noisylr", "noisy_lr"] and "test" not in str(p).lower():
            npy_count = len([f for f in files if f.lower().endswith(".npy") and not f.startswith(".")])
            if npy_count > 0:
                return p.resolve()
        elif target_type in ["gt_train", "gt"] and folder_lower == "gt":
            npy_count = len([f for f in files if f.lower().endswith(".npy") and not f.startswith(".")])
            if npy_count > 0:
                return p.resolve()
        elif target_type in ["test", "lq_test"] and folder_lower in ["noisylr", "noisy_lr"] and "test" in str(p).lower():
            npy_count = len([f for f in files if f.lower().endswith(".npy") and not f.startswith(".")])
            if npy_count > 0:
                return p.resolve()

    raise FileNotFoundError(
        f"Could not automatically resolve directory for target '{target_type}'. "
        f"Searched in: {dir_path} and root {repo_root}"
    )


def get_valid_npy_files(directory: Union[str, Path]) -> List[Path]:
    """Finds all non-hidden, non-__MACOSX .npy files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    valid_files = []
    for file_path in directory.iterdir():
        # Strictly ignore __MACOSX, hidden files, and non-.npy files
        if file_path.is_file() and file_path.suffix.lower() == ".npy":
            if not file_path.name.startswith(".") and not file_path.name.startswith("._") and "__MACOSX" not in str(file_path):
                valid_files.append(file_path)

    valid_files.sort(key=lambda p: p.name)
    return valid_files


class PairedAugmentation:
    """Synchronized geometric augmentations for 128x128 LQ and 256x256 GT images."""

    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = True,
        rot90: bool = True,
        transpose: bool = True,
        patch_size_lq: Optional[int] = None,
        scale: int = 2,
    ):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90
        self.transpose = transpose
        self.patch_size_lq = patch_size_lq
        self.scale = scale

    def __call__(
        self, lq: np.ndarray, gt: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applies identical geometric transforms to lq and gt.
        lq: [H, W] float32
        gt: [scale*H, scale*W] float32 (or None)
        """
        # 1. Optional synchronized random crop
        if self.patch_size_lq is not None and gt is not None:
            h_lq, w_lq = lq.shape
            p_lq = self.patch_size_lq
            p_gt = p_lq * self.scale

            if h_lq >= p_lq and w_lq >= p_lq:
                top_lq = np.random.randint(0, h_lq - p_lq + 1)
                left_lq = np.random.randint(0, w_lq - p_lq + 1)

                top_gt = top_lq * self.scale
                left_gt = left_lq * self.scale

                lq = lq[top_lq : top_lq + p_lq, left_lq : left_lq + p_lq]
                gt = gt[top_gt : top_gt + p_gt, left_gt : left_gt + p_gt]

        # 2. Horizontal flip
        if self.hflip and np.random.rand() > 0.5:
            lq = np.fliplr(lq)
            if gt is not None:
                gt = np.fliplr(gt)

        # 3. Vertical flip
        if self.vflip and np.random.rand() > 0.5:
            lq = np.flipud(lq)
            if gt is not None:
                gt = np.flipud(gt)

        # 4. 90-degree rotations (k in [0, 1, 2, 3])
        if self.rot90:
            k = np.random.randint(0, 4)
            if k > 0:
                lq = np.rot90(lq, k)
                if gt is not None:
                    gt = np.rot90(gt, k)

        # 5. Transpose (diagonal flip)
        if self.transpose and np.random.rand() > 0.5:
            lq = lq.T
            if gt is not None:
                gt = gt.T

        return np.ascontiguousarray(lq), (
            np.ascontiguousarray(gt) if gt is not None else None
        )


class KLADataset(Dataset):
    """
    KLA Semiconductor Image Restoration Dataset (V2).
    Loads real float32 .npy arrays for Low-Quality (128x128) and Ground Truth (256x256).
    """

    def __init__(
        self,
        lq_dir: Optional[Union[str, Path]] = None,
        gt_dir: Optional[Union[str, Path]] = None,
        split: str = "train",  # 'train', 'val', 'test', 'all'
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        patch_size_lq: Optional[int] = None,
        normalize: bool = True,
        mean: float = KLA_MEAN,
        std: float = KLA_STD,
        file_list: Optional[List[str]] = None,
        cache_in_memory: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        
        # Resolve LQ directory
        target_lq = "test" if split == "test" else "lq_train"
        self.lq_dir = resolve_dataset_dir(lq_dir, target_type=target_lq)

        # Resolve GT directory if required
        if split == "test" or (gt_dir is None and split not in ["train", "val"]):
            self.gt_dir = None
        else:
            self.gt_dir = resolve_dataset_dir(gt_dir, target_type="gt_train")

        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.cache_in_memory = cache_in_memory

        self.augment = augment
        self.transform = (
            PairedAugmentation(patch_size_lq=patch_size_lq) if augment else None
        )

        # Discover all valid LQ files
        all_lq_files = get_valid_npy_files(self.lq_dir)
        if not all_lq_files:
            raise ValueError(f"No valid .npy files found in {self.lq_dir}")

        # Check pairing if GT directory is provided
        if self.gt_dir is not None:
            gt_file_map = {f.name: f for f in get_valid_npy_files(self.gt_dir)}
            paired_filenames = []
            for lq_path in all_lq_files:
                if lq_path.name in gt_file_map:
                    paired_filenames.append(lq_path.name)
                else:
                    raise FileNotFoundError(
                        f"Missing matching GT file for {lq_path.name} in {self.gt_dir}"
                    )
        else:
            paired_filenames = [f.name for f in all_lq_files]

        # Handle train/val split (90% train = 2880, 10% val = 320 with seed=42)
        if file_list is not None:
            self.filenames = [f for f in file_list if f in paired_filenames]
        elif self.gt_dir is not None and split in ["train", "val"]:
            rng = np.random.RandomState(seed)
            shuffled = paired_filenames.copy()
            rng.shuffle(shuffled)
            val_count = int(len(shuffled) * val_ratio)
            if split == "val":
                self.filenames = sorted(shuffled[:val_count])
            else:  # train
                self.filenames = sorted(shuffled[val_count:])
        else:
            self.filenames = paired_filenames

        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]

        # Pre-cache arrays in memory for near-instant GPU batch feeding
        self.cached_lq = []
        self.cached_gt = []
        if self.cache_in_memory:
            for fname in self.filenames:
                lq_arr = np.load(str(self.lq_dir / fname)).astype(np.float32)
                self.cached_lq.append(lq_arr)
                if self.gt_dir is not None:
                    gt_arr = np.load(str(self.gt_dir / fname)).astype(np.float32)
                    self.cached_gt.append(gt_arr)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        fname = self.filenames[idx]

        if self.cache_in_memory:
            lq_arr = self.cached_lq[idx].copy()
            gt_arr = self.cached_gt[idx].copy() if self.gt_dir is not None else None
        else:
            lq_path = self.lq_dir / fname
            lq_arr = np.load(str(lq_path)).astype(np.float32)
            gt_arr = np.load(str(self.gt_dir / fname)).astype(np.float32) if self.gt_dir is not None else None

        if not np.all(np.isfinite(lq_arr)):
            raise ValueError(f"Corrupted array with NaN/Inf detected in {fname}")
        if lq_arr.shape != (128, 128):
            raise ValueError(f"Unexpected shape {lq_arr.shape} in {fname}, expected (128, 128)")

        if gt_arr is not None:
            if not np.all(np.isfinite(gt_arr)):
                raise ValueError(f"Corrupted array with NaN/Inf detected in {fname}")
            if gt_arr.shape != (256, 256):
                raise ValueError(f"Unexpected shape {gt_arr.shape} in {fname}, expected (256, 256)")

        # Keep original arrays for exact reference
        lq_raw = lq_arr.copy()
        gt_raw = gt_arr.copy() if gt_arr is not None else None

        # Apply paired augmentation if enabled
        if self.transform is not None:
            lq_arr, gt_arr = self.transform(lq_arr, gt_arr)

        # Normalize without clamping
        if self.normalize:
            lq_arr = normalize_fn(lq_arr, self.mean, self.std)
            if gt_arr is not None:
                gt_arr = normalize_fn(gt_arr, self.mean, self.std)

        # Convert to torch.Tensor [1, H, W]
        lq_tensor = torch.from_numpy(lq_arr).unsqueeze(0).float()
        item = {
            "lq": lq_tensor,
            "filename": fname,
        }

        if gt_arr is not None:
            gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0).float()
            item["gt"] = gt_tensor

        return item
