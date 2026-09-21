"""FlashSplat-style batch FG/BG support via auxiliary-color autograd."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from gaussian_renderer import render


def collect_binary_support(
    cameras: List,
    gaussians,
    pipe,
    background: torch.Tensor,
    q_by_name: Optional[dict] = None,
    max_views: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One forward+backward per view. Geometry/opacity are treated as constants.

    S_i^+ = dPhi / d u_i^+   (FG mass)
    S_i^- = dPhi / d u_i^-   (BG mass)
    """
    device = gaussians.get_xyz.device
    n = int(gaussians.get_xyz.shape[0])
    s_pos = torch.zeros(n, device=device)
    s_neg = torch.zeros(n, device=device)
    cams = [c for c in cameras if getattr(c, "has_object_mask", False)]
    # max_views<=0 means all masked cameras (needed for support_all ablation).
    if int(max_views) > 0:
        cams = cams[: int(max_views)]
    if not cams:
        return s_pos, s_neg

    opac = gaussians.get_opacity.detach()
    bg0 = torch.zeros_like(background)

    for cam in cams:
        u = torch.zeros(n, 3, device=device, dtype=torch.float32, requires_grad=True)
        pkg = render(
            cam,
            gaussians,
            pipe,
            bg0,
            override_color=u,
            override_opacity=opac,
        )
        c_img = pkg["render"]
        m = cam.object_mask.to(device)
        valid = cam.mask_valid.to(device)
        q = valid
        if q_by_name is not None and cam.image_name in q_by_name:
            q = q * q_by_name[cam.image_name].to(device)
        phi = (q * (m * c_img[:1] + (1.0 - m) * c_img[1:2])).sum()
        grad = torch.autograd.grad(phi, u, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            continue
        s_pos = s_pos + grad[:, 0].detach().clamp_min(0)
        s_neg = s_neg + grad[:, 1].detach().clamp_min(0)
        del pkg, c_img, grad, u, phi
    return s_pos, s_neg


def init_label_from_support(
    gaussians,
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    mass_tau: float = 1e-4,
) -> None:
    from utils.general_utils import inverse_sigmoid

    mass = s_pos + s_neg
    z = torch.where(mass > mass_tau, s_pos / mass.clamp_min(1e-8), torch.full_like(s_pos, 0.5))
    z = z.clamp(1e-4, 1.0 - 1e-4)
    logit = inverse_sigmoid(z).unsqueeze(-1)
    with torch.no_grad():
        gaussians._label_logit.data.copy_(logit.to(gaussians._label_logit.data.dtype))
