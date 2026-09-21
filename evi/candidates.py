"""Object-fidelity candidate eligibility m_i (not the final z threshold)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from evi.evidence.visibility_evidence import project_points_to_cam


@torch.no_grad()
def build_object_candidate_mask(
    gaussians,
    cameras: List,
    ref,
    s_pos: Optional[torch.Tensor] = None,
    s_neg: Optional[torch.Tensor] = None,
    spatial_ev=None,
    a_min: float = 0.05,
    depth_tol: float = 0.05,
    front_support_tau: float = 1e-3,
    radius_mult: float = 4.0,
    max_views: int = 6,
) -> torch.Tensor:
    """
    Eligible for the object-fidelity branch:
      - has FG support, or is in a seed neighborhood
      - not a confident background-surface / free-space occupant without FG support
    Missed object primitives with G0 FG evidence can be added back.
    Background cannot be added just to fill coverage.
    """
    device = gaussians.get_xyz.device
    n = int(gaussians.get_xyz.shape[0])
    xyz = gaussians.get_xyz
    o = gaussians.get_opacity.reshape(-1)
    m = torch.zeros(n, dtype=torch.bool, device=device)

    if s_pos is not None:
        m = m | (s_pos.reshape(-1) >= float(front_support_tau))

    cams = [c for c in cameras if getattr(c, "has_object_mask", False)][:max_views]
    fg_hit = torch.zeros(n, device=device)
    bg_hit = torch.zeros(n, device=device)
    free_hit = torch.zeros(n, device=device)
    for cam in cams:
        name = cam.image_name
        proj = project_points_to_cam(xyz, cam)
        u, v, d = proj[..., 0], proj[..., 1], proj[..., 2]
        in_img = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (d > 1e-4)
        if not in_img.any():
            continue
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).view(1, -1, 1, 2)
        gt = F.grid_sample(cam.object_mask[None], grid, align_corners=True, mode="nearest").view(-1)
        vld = F.grid_sample(cam.mask_valid[None], grid, align_corners=True, mode="nearest").view(-1)
        if ref is not None and name in ref.depth_by_name:
            d0 = F.grid_sample(ref.depth_by_name[name][None], grid, align_corners=True, mode="bilinear").view(-1)
            a0 = F.grid_sample(ref.alpha_by_name[name][None], grid, align_corners=True, mode="bilinear").view(-1)
            near = in_img & ((d - d0).abs() <= depth_tol * d0.clamp_min(1e-3)) & (a0 >= a_min)
            free = in_img & (d < d0 * (1.0 - depth_tol)) & (a0 >= a_min)
        else:
            near = in_img
            free = torch.zeros_like(in_img)
        fg = near & (vld > 0.5) & (gt > 0.5)
        bg = near & (vld > 0.5) & (gt <= 0.5)
        fg_hit = fg_hit + fg.float() * o
        bg_hit = bg_hit + bg.float() * o
        free_hit = free_hit + free.float() * o

    m = m | (fg_hit >= float(front_support_tau))

    # Grow around current seeds (G0 surface neighborhood). Cap seed count for memory.
    n_seed = int(m.sum().item())
    if 0 < n_seed <= 4096:
        seeds = m.nonzero(as_tuple=False).view(-1)
        scale = gaussians.get_scaling.max(dim=-1).values
        radius = float(radius_mult) * float(scale[seeds].median().clamp_min(1e-4).item())
        chunk = 8192
        near_seed = torch.zeros(n, dtype=torch.bool, device=device)
        seed_xyz = xyz[seeds]
        for i in range(0, n, chunk):
            sl = xyz[i : i + chunk]
            dist = torch.cdist(sl, seed_xyz).min(dim=1).values
            near_seed[i : i + chunk] = dist <= radius
        m = m | near_seed

    # Hard negatives: background surface or free space without FG support.
    no_fg = fg_hit < float(front_support_tau)
    if s_pos is not None:
        no_fg = no_fg & (s_pos.reshape(-1) < float(front_support_tau))
    m = m & ~(no_fg & ((bg_hit > fg_hit + 1e-4) | (free_hit > 0.05)))
    if not m.any():
        if fg_hit.any():
            m = fg_hit >= float(front_support_tau)
        elif s_pos is not None and s_pos.numel() == n:
            k = min(1024, n)
            _, top = torch.topk(s_pos.reshape(-1), k)
            m = torch.zeros(n, dtype=torch.bool, device=device)
            m[top] = True
    return m
