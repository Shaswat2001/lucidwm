"""Misc functions to be used across the package."""

from __future__ import annotations

import torch
import numpy as np

def set_seed(seed):
    """Set seed across the training run"""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device