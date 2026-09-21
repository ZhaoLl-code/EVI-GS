"""Read-only reference geometry from a frozen RGB warmup model (G^0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from evi.render.label_renderer import render_alpha_depth

# Dict used by render_with_g0 return type


@dataclass
class FrozenG0:
    """Detached snapshot of geometry/opacity used as read-only evidence source.

    ``shadow`` is an independent GaussianModel clone so G0 can be rendered after
    the live model densifies / splits (N no longer matches G0).
    """

    xyz: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    shadow: Optional[object] = None

    @staticmethod
    def from_gaussians(gaussians) -> "FrozenG0":
        from torch import nn

        from scene.gaussian_model import GaussianModel

        xyz = gaussians._xyz.detach().clone()
        opacity = gaussians._opacity.detach().clone()
        scaling = gaussians._scaling.detach().clone()
        rotation = gaussians._rotation.detach().clone()
        n = int(xyz.shape[0])
        device = xyz.device

        shadow = GaussianModel(gaussians.max_sh_degree)
        shadow._xyz = nn.Parameter(xyz, requires_grad=False)
        shadow._opacity = nn.Parameter(opacity, requires_grad=False)
        shadow._scaling = nn.Parameter(scaling, requires_grad=False)
        shadow._rotation = nn.Parameter(rotation, requires_grad=False)
        shadow._features_dc = nn.Parameter(gaussians._features_dc.detach().clone(), requires_grad=False)
        shadow._features_rest = nn.Parameter(gaussians._features_rest.detach().clone(), requires_grad=False)
        if getattr(gaussians, "_label_logit", None) is not None and gaussians._label_logit.numel() == n:
            shadow._label_logit = nn.Parameter(gaussians._label_logit.detach().clone(), requires_grad=False)
        else:
            shadow._label_logit = nn.Parameter(torch.zeros(n, 1, device=device), requires_grad=False)
        shadow.active_sh_degree = int(gaussians.active_sh_degree)
        shadow.max_radii2D = torch.zeros(n, device=device)

        return FrozenG0(
            xyz=shadow._xyz.data,
            opacity=shadow._opacity.data,
            scaling=shadow._scaling.data,
            rotation=shadow._rotation.data,
            shadow=shadow,
        )


@dataclass
class ReferenceGeometry:
    depth_by_name: Dict[str, torch.Tensor]
    alpha_by_name: Dict[str, torch.Tensor]
    reliable_by_name: Dict[str, torch.Tensor] = field(default_factory=dict)
    unknown_by_name: Dict[str, torch.Tensor] = field(default_factory=dict)
    g0: Optional[FrozenG0] = None
    a_min: float = 0.05


def _apply_frozen_geometry(gaussians, g0: FrozenG0):
    """Temporarily copy G0 into live geometry buffers (caller must restore)."""
    backup = {
        "xyz": gaussians._xyz.data.clone(),
        "opacity": gaussians._opacity.data.clone(),
        "scaling": gaussians._scaling.data.clone(),
        "rotation": gaussians._rotation.data.clone(),
    }
    gaussians._xyz.data.copy_(g0.xyz)
    gaussians._opacity.data.copy_(g0.opacity)
    gaussians._scaling.data.copy_(g0.scaling)
    gaussians._rotation.data.copy_(g0.rotation)
    return backup


def _restore_geometry(gaussians, backup: Dict[str, torch.Tensor]):
    gaussians._xyz.data.copy_(backup["xyz"])
    gaussians._opacity.data.copy_(backup["opacity"])
    gaussians._scaling.data.copy_(backup["scaling"])
    gaussians._rotation.data.copy_(backup["rotation"])


@torch.no_grad()
def build_reference_from_gaussians(
    cameras: List,
    gaussians,
    pipe,
    background: torch.Tensor,
    max_views: Optional[int] = None,
    a_min: float = 0.05,
    g0: Optional[FrozenG0] = None,
    cross_view_depth_tol: float = 0.05,
) -> ReferenceGeometry:
    """
    Render depth/alpha from frozen G0 for each training camera.
    Alpha is true accumulated alpha (override_color=1), not RGB brightness.
    Weak cross-view reliable/unknown: pixel reliable if alpha>=a_min and
    (when a second nearby view exists) relative depth agrees within tol.
    """
    if g0 is None:
        g0 = FrozenG0.from_gaussians(gaussians)

    cams = cameras if max_views is None else cameras[:max_views]
    # Prefer independent shadow model so we never mutate live densified buffers.
    pc = g0.shadow if g0.shadow is not None else gaussians
    backup = None
    if g0.shadow is None:
        if gaussians._xyz.shape[0] != g0.xyz.shape[0]:
            raise RuntimeError(
                f"G0 size {g0.xyz.shape[0]} != live {gaussians._xyz.shape[0]}; "
                "rebuild FrozenG0 with shadow=True"
            )
        backup = _apply_frozen_geometry(gaussians, g0)
        pc = gaussians
    try:
        depth_by_name: Dict[str, torch.Tensor] = {}
        alpha_by_name: Dict[str, torch.Tensor] = {}
        for cam in cams:
            alpha, depth, _ = render_alpha_depth(cam, pc, pipe, background)
            if depth is None:
                d = torch.linalg.norm(g0.xyz - cam.camera_center[None], dim=-1)
                med = d.median()
                depth = torch.full(
                    (1, cam.image_height, cam.image_width),
                    float(med.item()),
                    device=cam.data_device,
                )
            depth_by_name[cam.image_name] = depth.detach()
            alpha_by_name[cam.image_name] = alpha.detach()
    finally:
        if backup is not None:
            _restore_geometry(gaussians, backup)

    reliable_by_name, unknown_by_name = _mark_reliable_unknown(
        cams, depth_by_name, alpha_by_name, a_min=a_min, depth_tol=cross_view_depth_tol
    )
    return ReferenceGeometry(
        depth_by_name=depth_by_name,
        alpha_by_name=alpha_by_name,
        reliable_by_name=reliable_by_name,
        unknown_by_name=unknown_by_name,
        g0=g0,
        a_min=a_min,
    )


def _mark_reliable_unknown(
    cameras: List,
    depth_by_name: Dict[str, torch.Tensor],
    alpha_by_name: Dict[str, torch.Tensor],
    a_min: float,
    depth_tol: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Weak marks:
      covered = alpha >= a_min
      reliable = covered (and if >=2 views, keep all covered; inconsistent → unknown)
    Without dense multi-view reprojection, treat covered as reliable and
    low-alpha as unknown. Optional pairwise check on shared downsample.
    """
    reliable_by_name: Dict[str, torch.Tensor] = {}
    unknown_by_name: Dict[str, torch.Tensor] = {}
    names = [c.image_name for c in cameras if c.image_name in alpha_by_name]
    for name in names:
        alpha = alpha_by_name[name]
        covered = (alpha >= a_min).float()
        reliable_by_name[name] = covered
        unknown_by_name[name] = 1.0 - covered

    # Pairwise consistency on overlapping covered pixels via coarse center crop stats:
    # if two views both have high coverage fraction, flag nothing extra (weak version).
    # Stronger: for each cam, compare median depth of covered region to neighbors' medians.
    if len(names) >= 2:
        med = {}
        for name in names:
            D = depth_by_name[name]
            R = reliable_by_name[name] > 0.5
            if R.any():
                med[name] = float(D[R].median().item())
            else:
                med[name] = None
        global_vals = [v for v in med.values() if v is not None]
        if global_vals:
            gmed = float(sum(global_vals) / len(global_vals))
            for name in names:
                if med[name] is None:
                    continue
                if abs(med[name] - gmed) > depth_tol * max(gmed, 1e-3) * 10.0:
                    # Whole-view scale outlier → demote reliability slightly
                    reliable_by_name[name] = reliable_by_name[name] * 0.0
                    unknown_by_name[name] = torch.ones_like(unknown_by_name[name])
    return reliable_by_name, unknown_by_name


@torch.no_grad()
def render_with_g0(
    cam, gaussians, pipe, background: torch.Tensor, g0: FrozenG0
) -> Optional[Dict[str, torch.Tensor]]:
    """Render depth/alpha using frozen G0 geometry (one raster pass).

    Prefer ``g0.shadow`` (independent of live densify). If only tensor snapshots
    exist and live N mismatches G0, returns None (callers must fall back).
    """
    if g0.shadow is not None:
        pc = g0.shadow
        alpha, depth, pkg = render_alpha_depth(cam, pc, pipe, background)
        return {"render": pkg["render"], "depth": depth, "alpha": alpha}

    if gaussians._xyz.shape[0] != g0.xyz.shape[0]:
        return None

    backup = _apply_frozen_geometry(gaussians, g0)
    try:
        alpha, depth, pkg = render_alpha_depth(cam, gaussians, pipe, background)
        return {"render": pkg["render"], "depth": depth, "alpha": alpha}
    finally:
        _restore_geometry(gaussians, backup)


@torch.no_grad()
def unproject_reliable_surface(
    cam,
    ref: ReferenceGeometry,
    max_points: int = 20000,
    stride: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unproject reliable reference depth pixels to world points.
    Returns xyz (P,3), uv source (P,2) in pixels, confidence (P,).
    """
    name = cam.image_name
    if name not in ref.depth_by_name:
        device = cam.data_device
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 2, device=device),
            torch.zeros(0, device=device),
        )
    D = ref.depth_by_name[name][0]
    A = ref.alpha_by_name[name][0]
    R = ref.reliable_by_name.get(name, (A >= ref.a_min).float())[0]
    H, W = D.shape
    ys = torch.arange(0, H, stride, device=D.device)
    xs = torch.arange(0, W, stride, device=D.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = R[yy, xx] > 0.5
    if not mask.any():
        return (
            torch.zeros(0, 3, device=D.device),
            torch.zeros(0, 2, device=D.device),
            torch.zeros(0, device=D.device),
        )
    yy, xx = yy[mask], xx[mask]
    d = D[yy, xx]
    conf = A[yy, xx]
    # NDC-ish from FoV
    tanx = torch.tan(torch.tensor(cam.FoVx * 0.5, device=D.device))
    tany = torch.tan(torch.tensor(cam.FoVy * 0.5, device=D.device))
    u_n = (xx.float() / max(W - 1, 1)) * 2 - 1
    v_n = (yy.float() / max(H - 1, 1)) * 2 - 1
    x_cam = u_n * d * tanx
    y_cam = v_n * d * tany
    z_cam = d
    ones = torch.ones_like(z_cam)
    cam_pts = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # P,4
    # world_view_transform is W2V^T in this codebase → V2W = inverse
    w2c = cam.world_view_transform
    c2w = torch.linalg.inv(w2c)
    world = cam_pts @ c2w  # P,4
    xyz = world[:, :3]
    uv = torch.stack([xx.float(), yy.float()], dim=-1)
    if xyz.shape[0] > max_points:
        sel = torch.randperm(xyz.shape[0], device=xyz.device)[:max_points]
        xyz, uv, conf = xyz[sel], uv[sel], conf[sel]
    return xyz, uv, conf
