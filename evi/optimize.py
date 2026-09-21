"""Trainable-subset helpers: freeze groups, clear Adam state, mask inactive grads."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


GEO_NAMES = ("xyz", "opacity", "scaling", "rotation")
APP_NAMES = ("f_dc", "f_rest")
LABEL_NAMES = ("label_logit",)


def set_trainable(gaussians, geometry: bool, appearance: bool, label: bool) -> None:
    gaussians._xyz.requires_grad_(geometry)
    gaussians._scaling.requires_grad_(geometry)
    gaussians._rotation.requires_grad_(geometry)
    gaussians._opacity.requires_grad_(geometry)
    gaussians._features_dc.requires_grad_(appearance)
    gaussians._features_rest.requires_grad_(appearance)
    gaussians._label_logit.requires_grad_(label)


def clear_adam_state(gaussians, names: Sequence[str]) -> None:
    """Drop Adam moments for named groups.

    Must delete the state entry entirely. Leaving ``{}`` breaks densify/prune,
    which treat any non-None state as a full Adam buffer with ``exp_avg``.
    """
    opt = getattr(gaussians, "optimizer", None)
    if opt is None:
        return
    name_set = set(names)
    for group in opt.param_groups:
        if group.get("name") not in name_set:
            continue
        p = group["params"][0]
        if p in opt.state:
            del opt.state[p]


def scale_group_lrs(gaussians, names: Sequence[str], mult: float, store_key: str = "_evi_base_lr") -> None:
    opt = getattr(gaussians, "optimizer", None)
    if opt is None:
        return
    name_set = set(names)
    for group in opt.param_groups:
        if group.get("name") not in name_set:
            continue
        if store_key not in group:
            group[store_key] = float(group["lr"])
        group["lr"] = float(group[store_key]) * float(mult)


def restore_group_lrs(gaussians, store_key: str = "_evi_base_lr") -> None:
    opt = getattr(gaussians, "optimizer", None)
    if opt is None:
        return
    for group in opt.param_groups:
        if store_key in group:
            group["lr"] = float(group[store_key])


def zero_inactive_grads(gaussians, active: Optional[torch.Tensor]) -> None:
    """active: (N,) bool. None means all live parameters may update."""
    if active is None:
        return
    n = int(gaussians.get_xyz.shape[0])
    if active.numel() != n:
        return
    inactive = ~active.reshape(n)
    for p in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ):
        if p.grad is None:
            continue
        g = p.grad
        if g.ndim == 1:
            g[inactive] = 0
        else:
            g[inactive] = 0


@torch.no_grad()
def neighbor_active_mask(
    gaussians,
    seed_ids: torch.Tensor,
    radius_mult: float = 4.0,
    extra_ids: Optional[torch.Tensor] = None,
    cap: int = 32768,
) -> torch.Tensor:
    n = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    active = torch.zeros(n, dtype=torch.bool, device=device)
    if extra_ids is not None and extra_ids.numel() > 0:
        active[extra_ids.long().clamp(0, n - 1)] = True
    if seed_ids is None or seed_ids.numel() == 0:
        if not active.any():
            active[:] = True
        return active
    seeds = seed_ids.long().unique().clamp(0, n - 1)
    if seeds.numel() > 2048:
        seeds = seeds[torch.randperm(seeds.numel(), device=device)[:2048]]
    active[seeds] = True
    xyz = gaussians.get_xyz
    scale = gaussians.get_scaling.max(dim=-1).values
    radius = float(radius_mult) * float(scale[seeds].median().clamp_min(1e-4).item())
    seed_xyz = xyz[seeds]
    chunk = 8192
    for i in range(0, n, chunk):
        dist = torch.cdist(xyz[i : i + chunk], seed_xyz).min(dim=1).values
        active[i : i + chunk] |= dist <= radius
    if int(active.sum().item()) > cap:
        # Keep seeds plus highest-opacity neighbors.
        o = gaussians.get_opacity.reshape(-1)
        score = o.clone()
        score[~active] = -1
        score[seeds] = 1e6
        _, top = torch.topk(score, cap)
        keep = torch.zeros(n, dtype=torch.bool, device=device)
        keep[top] = True
        keep[seeds] = True
        active = keep
    return active


class CamQueue:
    def __init__(self, cameras: Iterable):
        self.cams = list(cameras)
        self._order = []
        self._i = 0
        self._reshuffle()

    def _reshuffle(self):
        n = len(self.cams)
        if n == 0:
            self._order = []
        else:
            self._order = torch.randperm(n).tolist()
        self._i = 0

    def next(self):
        if not self.cams:
            raise RuntimeError("empty camera queue")
        if self._i >= len(self._order):
            self._reshuffle()
        cam = self.cams[self._order[self._i]]
        self._i += 1
        return cam
