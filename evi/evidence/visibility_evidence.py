"""Visibility-layered spatial evidence from frozen G0 depth and masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from evi.evidence.reference_geometry import ReferenceGeometry


@dataclass
class SpatialEvidence:
    """Semi-explicit per-view evidence maps (unknown defaults to 0 contribution)."""

    E_free: Dict[str, torch.Tensor]
    E_F: Dict[str, torch.Tensor]
    E_B: Dict[str, torch.Tensor]


def project_points_to_cam(points: torch.Tensor, cam) -> torch.Tensor:
    """
    points: (...,3) world -> (...,3) where [...,0]=u_norm, [...,1]=v_norm, [...,2]=depth
    """
    ones = torch.ones(*points.shape[:-1], 1, device=points.device)
    homo = torch.cat([points, ones], dim=-1)
    view = homo @ cam.world_view_transform
    x, y, z = view[..., 0], view[..., 1], view[..., 2]
    tanx = torch.tan(torch.tensor(cam.FoVx * 0.5, device=points.device))
    tany = torch.tan(torch.tensor(cam.FoVy * 0.5, device=points.device))
    u = (x / (z.clamp_min(1e-6) * tanx) + 1.0) * 0.5
    v = (y / (z.clamp_min(1e-6) * tany) + 1.0) * 0.5
    return torch.stack([u, v, z], dim=-1)


@torch.no_grad()
def build_spatial_evidence(
    cameras: List,
    ref: ReferenceGeometry,
    depth_tol: float = 0.05,
    a_min: float = 0.05,
) -> SpatialEvidence:
    """
    Semi-explicit E_free / E_F / E_B from G0 depth + mask.
    Unknown regions stay 0 (no free-space wipe).
    """
    E_free, E_F, E_B = {}, {}, {}
    for cam in cameras:
        name = cam.image_name
        if name not in ref.depth_by_name:
            continue
        A = ref.alpha_by_name[name]
        R = ref.reliable_by_name.get(name, (A >= a_min).float())
        covered = (A >= a_min).float() * R
        e_free = covered.clone()
        e_f = torch.zeros_like(A)
        e_b = torch.zeros_like(A)
        if getattr(cam, "has_object_mask", False):
            fg = (cam.object_mask > 0.5).float() * cam.mask_valid * covered
            bg = (cam.object_mask <= 0.5).float() * cam.mask_valid * covered
            e_f = fg
            e_b = bg
            e_free = covered * cam.mask_valid
        E_free[name] = e_free
        E_F[name] = e_f
        E_B[name] = e_b
    return SpatialEvidence(E_free=E_free, E_F=E_F, E_B=E_B)
