"""Mask supervision intervals [ell, u]."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def boundary_band_mask(binary: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Return 1 on morphological boundary band of a {0,1} mask."""
    if width <= 0:
        return torch.zeros_like(binary)
    x = binary.detach().float()
    if x.ndim == 3:
        x2 = x
    else:
        x2 = x[None]
    # dilate - erode via max/min pools
    k = 2 * width + 1
    pad = width
    dil = F.max_pool2d(x2, kernel_size=k, stride=1, padding=pad)
    er = 1.0 - F.max_pool2d(1.0 - x2, kernel_size=k, stride=1, padding=pad)
    band = (dil - er).clamp(0, 1)
    return band if binary.ndim == 3 else band[0]


def build_intervals(
    object_mask: torch.Tensor,
    mask_valid: torch.Tensor,
    boundary_width: int = 3,
    eps_fg: float = 0.05,
    eps_bg: float = 0.05,
    eps_bound: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns ell, u, weight in same shape as object_mask.
    Missing annotation (mask_valid=0): weight=0 (no semantic constraint).
    """
    band = boundary_band_mask(object_mask, boundary_width)
    ell = torch.zeros_like(object_mask)
    u = torch.ones_like(object_mask)
    weight = mask_valid.clone()

    fg = (object_mask > 0.5) & (mask_valid > 0.5) & (band < 0.5)
    bg = (object_mask <= 0.5) & (mask_valid > 0.5) & (band < 0.5)
    bd = (band > 0.5) & (mask_valid > 0.5)

    ell = torch.where(fg, torch.full_like(ell, 1.0 - eps_fg), ell)
    u = torch.where(fg, torch.ones_like(u), u)

    ell = torch.where(bg, torch.zeros_like(ell), ell)
    u = torch.where(bg, torch.full_like(u, eps_bg), u)

    ell = torch.where(bd, torch.full_like(ell, eps_bound), ell)
    u = torch.where(bd, torch.full_like(u, 1.0 - eps_bound), u)

    return ell, u, weight


def intervals_for_camera(
    cam,
    boundary_width: int = 3,
    eps_fg: float = 0.05,
    eps_bg: float = 0.05,
    eps_bound: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache [ell,u,weight] on the camera; masks are static during training."""
    key = (int(boundary_width), float(eps_fg), float(eps_bg), float(eps_bound))
    cached = getattr(cam, "_evi_interval_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    result = build_intervals(
        cam.object_mask,
        cam.mask_valid,
        boundary_width=boundary_width,
        eps_fg=eps_fg,
        eps_bg=eps_bg,
        eps_bound=eps_bound,
    )
    cam._evi_interval_cache = (key, result)
    return result


def interval_distance(pred: torch.Tensor, ell: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """dist(p, [ell,u]) = max(ell-p, p-u, 0)."""
    return torch.relu(ell - pred) + torch.relu(pred - u)
