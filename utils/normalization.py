"""
Official KLA Semiconductor Dataset Normalization Module.
Provides centralized, consistent normalization and denormalization utilities
for PyTorch tensors and NumPy arrays without artificial clamping.
"""

from typing import Union
import numpy as np
import torch

# Official verified KLA dataset statistics
KLA_MEAN: float = 0.43353602
KLA_STD: float = 0.28478748


def normalize(
    x: Union[torch.Tensor, np.ndarray],
    mean: float = KLA_MEAN,
    std: float = KLA_STD,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Applies standard z-score normalization using dataset mean and standard deviation.
    Does NOT clamp values to [0, 1] or uint8.
    
    Args:
        x: Input image tensor or array (float32)
        mean: Dataset mean (default: 0.43353602)
        std: Dataset standard deviation (default: 0.28478748)
        
    Returns:
        Normalized tensor or array
    """
    return (x - mean) / std


def denormalize(
    x: Union[torch.Tensor, np.ndarray],
    mean: float = KLA_MEAN,
    std: float = KLA_STD,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Restores normalized tensor/array back to original float32 intensity range.
    
    Args:
        x: Normalized image tensor or array
        mean: Dataset mean (default: 0.43353602)
        std: Dataset standard deviation (default: 0.28478748)
        
    Returns:
        Denormalized tensor or array in original intensity domain
    """
    return (x * std) + mean
