"""Independent object rendering: tilde o = m * z * o (soft) or h * o (hard)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from evi.evidence.mask_intervals import boundary_band_mask
from gaussian_renderer import render
from utils.loss_utils import create_window


def _as_col(opacity: torch.Tensor) -> torch.Tensor:
    return opacity.unsqueeze(-1) if opacity.ndim == 1 else opacity


def render_object(
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    opacity: torch.Tensor,
    need_alpha: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Rasterize with a caller-supplied opacity (already in (0,1)).
    RGB uses SH appearance; alpha is a second pass with color=1, bg=0.
    """
    opac = _as_col(opacity)
    pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        override_opacity=opac,
    )
    out = {"render": pkg["render"], "radii": pkg["radii"], "depth": pkg.get("depth")}
    if need_alpha:
        ones = torch.ones(opac.shape[0], 3, device=opac.device, dtype=opac.dtype)
        bg0 = torch.zeros_like(background)
        apkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg0,
            override_color=ones,
            override_opacity=opac,
        )
        out["alpha"] = apkg["render"][:1].clamp(0, 1)
    return out


def render_object_soft(
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    candidate_mask: torch.Tensor,
    need_alpha: bool = True,
    use_label: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Soft object render. When use_label=True: tilde = m * z * o.
    When use_label=False: tilde = m * o (m detached) so L_app does not
    supervise unreliable labels — only geometry/opacity/appearance.
    """
    o = gaussians.get_opacity.reshape(-1)
    m = candidate_mask.to(device=o.device, dtype=o.dtype).reshape(-1)
    if use_label:
        z = gaussians.get_label.reshape(-1)
        tilde = (m * z * o).clamp(0, 1)
    else:
        tilde = (m.detach() * o).clamp(0, 1)
    return render_object(viewpoint_cam, gaussians, pipe, background, tilde, need_alpha=need_alpha)


def render_object_hard(
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    hard_mask: torch.Tensor,
    need_alpha: bool = True,
) -> Dict[str, torch.Tensor]:
    o = gaussians.get_opacity.reshape(-1)
    h = hard_mask.to(device=o.device, dtype=o.dtype).reshape(-1)
    # Detach h so restore does not leak gradients into labels.
    tilde = (h.detach() * o).clamp(0, 1)
    return render_object(viewpoint_cam, gaussians, pipe, background, tilde, need_alpha=need_alpha)


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Per-pixel SSIM map averaged over channels, shape (1,H,W)."""
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    channel = img1.size(1)
    window = create_window(window_size, channel)
    window = window.to(img1.device).type_as(img1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    out = ssim.mean(dim=1, keepdim=True)
    return out[0] if squeeze else out


def erode_binary(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    x = mask.float()
    if x.ndim == 3:
        x2 = x
    else:
        x2 = x[None]
    k = 2 * width + 1
    er = 1.0 - F.max_pool2d(1.0 - x2, kernel_size=k, stride=1, padding=width)
    return er if mask.ndim == 3 else er[0]


def object_supervision_weights(
    cam,
    ref,
    a_min: float = 0.05,
    boundary_width: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fixed (stop-grad) weights from GT mask + G0 reference.
    V_rgb / V_cov do not depend on the current object alpha.
    """
    device = cam.object_mask.device
    gt = cam.object_mask
    valid = cam.mask_valid
    name = cam.image_name
    if ref is None or name not in getattr(ref, "alpha_by_name", {}):
        zeros = torch.zeros_like(gt)
        return zeros, zeros, zeros
    a0 = ref.alpha_by_name[name].to(device)
    rel = ref.reliable_by_name.get(name)
    if rel is None:
        rel = (a0 >= a_min).float()
    else:
        rel = rel.to(device)
    fg = (gt > 0.5) & (valid > 0.5)
    band = boundary_band_mask(gt, boundary_width)
    interior = fg & (band < 0.5) & (rel > 0.5) & (a0 >= a_min)
    v_rgb = interior.float()
    v_cov = (interior & (a0 >= 0.8)).float()
    v_neg = ((gt <= 0.5) & (valid > 0.5) & (rel > 0.5) & (band < 0.5)).float()
    return v_rgb.detach(), v_cov.detach(), v_neg.detach()


def object_fidelity_loss(
    image: torch.Tensor,
    alpha: Optional[torch.Tensor],
    cam,
    v_rgb: torch.Tensor,
    v_cov: torch.Tensor,
    lambda_dssim: float = 0.2,
    alpha_star: float = 0.95,
    window_size: int = 11,
) -> Dict[str, torch.Tensor]:
    """
    L_app^obj = 0.8 L1 + 0.2 L_SSIM on reliable interior, plus coverage hole penalty.
    SSIM is a local map restricted to windows fully inside V_rgb — not a black-bg image SSIM.
    """
    device = image.device
    zero = torch.zeros((), device=device)
    gt = cam.original_image
    if gt.device != device:
        gt = gt.to(device)
    if cam.alpha_mask is not None:
        am = cam.alpha_mask
        if am.device != device:
            am = am.to(device)
        image = image * am
        gt = gt * am

    eps = 1e-6
    v = v_rgb.to(device)
    denom = v.sum().clamp_min(1.0)
    per_pix = (image - gt).abs().sum(dim=0, keepdim=True)
    l1 = (per_pix * v).sum() / (3.0 * denom + eps)

    l_ssim = zero
    if v.sum() > window_size * window_size:
        smap = ssim_map(image, gt, window_size=window_size)
        # Window fully inside V_rgb: min-pool of V equals 1
        inside = erode_binary(v, window_size // 2)
        w = inside
        if w.sum() > 0:
            l_ssim = ((1.0 - smap) * w).sum() / (w.sum() + eps)
        # Thin structures: keep L1/coverage even if no valid SSIM window.

    l_app = (1.0 - lambda_dssim) * l1 + lambda_dssim * l_ssim

    l_cov = zero
    if alpha is not None:
        vc = v_cov.to(device)
        if vc.sum() > 0:
            l_cov = (torch.relu(float(alpha_star) - alpha) * vc).sum() / (vc.sum() + eps)

    return {
        "l1": l1,
        "ssim": l_ssim,
        "app": l_app,
        "cov": l_cov,
    }
