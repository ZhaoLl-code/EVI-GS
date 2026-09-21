"""Diagnostics JSON export."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import torch

from utils.system_utils import mkdir_p


def _to_serializable(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_diagnostics(path: str, payload: Dict) -> None:
    mkdir_p(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def label_range_summary(gaussians) -> Dict:
    z = gaussians.get_label.detach().squeeze(-1)
    return {
        "mean_z": float(z.mean().item()),
        "frac_fg_0.5": float((z >= 0.5).float().mean().item()),
        "frac_uncertain": float(((z > 0.2) & (z < 0.8)).float().mean().item()),
        "num_gaussians": int(z.numel()),
    }
