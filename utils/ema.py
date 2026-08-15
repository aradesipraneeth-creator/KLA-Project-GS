"""
Exponential Moving Average (EMA) for PyTorch model weights.
"""

from copy import deepcopy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional = None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model: nn.Module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model: nn.Module):
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.module.load_state_dict(state_dict)
