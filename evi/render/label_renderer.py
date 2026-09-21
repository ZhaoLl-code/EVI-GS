"""Label rendering via override_color, with conditional (alpha-normalized) labels."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from gaussian_renderer import render


def label_as_rgb(labels: torch.Tensor) -> torch.Tensor:
    """(N,1) or (N,) -> (N,3) grayscale RGB for rasterizer."""
    if labels.ndim == 1:
        labels = labels[:, None]
    return labels.repeat(1, 3)


def render_alpha_depth(viewpoint_cam, gaussians, pipe, background, scaling_modifier: float = 1.0):
    """
    One raster pass: accumulated alpha a_r (override_color=1) plus rasterizer depth.
    Background is 0 so unused mass does not inflate alpha.
    Returns (alpha 1xHxW, depth 1xHxW or None, full render pkg).
    """
    ones = torch.ones(gaussians.get_xyz.shape[0], 3, device=gaussians.get_xyz.device, dtype=torch.float32)
    bg0 = torch.zeros_like(background)
    pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg0,
        scaling_modifier=scaling_modifier,
        override_color=ones,
    )
    alpha = pkg["render"][:1].clamp(0, 1)
    depth = pkg.get("depth", None)
    if depth is not None and depth.ndim == 2:
        depth = depth[None]
    return alpha, depth, pkg


def render_alpha(viewpoint_cam, gaussians, pipe, background, scaling_modifier: float = 1.0) -> torch.Tensor:
    """
    Accumulated alpha a_r via override_color=1 (all Gaussians).
    out = sum_i w_ri when c_i = 1.
    """
    alpha, _, _ = render_alpha_depth(viewpoint_cam, gaussians, pipe, background, scaling_modifier)
    return alpha


def render_label(viewpoint_cam, gaussians, pipe, background, scaling_modifier: float = 1.0):
    """
    Render soft object label map using affiliation z as per-Gaussian color.
    Returns package with:
      label: unnormalized contribution sum_i w_ri z_i  (1,H,W)
      alpha: accumulated a_r (lazy; call ensure_conditional to fill)
    """
    colors = label_as_rgb(gaussians.get_label)
    bg0 = torch.zeros_like(background)
    pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg0,
        scaling_modifier=scaling_modifier,
        override_color=colors,
    )
    label_img = pkg["render"][:1]
    pkg["label"] = label_img
    pkg["label_unnormalized"] = label_img
    return pkg


def _label_alpha_override_colors(gaussians) -> torch.Tensor:
    """Pack unnormalized label into R and accumulated alpha into G (B unused)."""
    z = gaussians.get_label.reshape(-1)
    colors = torch.zeros(z.shape[0], 3, device=z.device, dtype=torch.float32)
    colors[:, 0] = z
    colors[:, 1] = 1.0
    return colors


def render_label_alpha(
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    scaling_modifier: float = 1.0,
):
    """
    Single rasterization: channel 0 = sum w_ri z_i, channel 1 = a_r = sum w_ri.
    Mathematically identical to render_label + render_alpha.
    """
    bg0 = torch.zeros_like(background)
    pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg0,
        scaling_modifier=scaling_modifier,
        override_color=_label_alpha_override_colors(gaussians),
    )
    img = pkg["render"]
    pkg["label_unnormalized"] = img[:1]
    pkg["label"] = pkg["label_unnormalized"]
    pkg["alpha"] = img[1:2].clamp(0, 1)
    return pkg


def ensure_conditional(
    pkg: Dict,
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    a_min: float = 0.05,
) -> Dict:
    """
    Attach conditional label p = (sum w z) / a for a >= a_min.
    Pixels with a < a_min get coverage_mask=0 (exclude from strict label intervals).
    """
    if "alpha" not in pkg or pkg["alpha"] is None:
        pkg["alpha"] = render_alpha(viewpoint_cam, gaussians, pipe, background)
    alpha = pkg["alpha"]
    unnorm = pkg.get("label_unnormalized", pkg["label"])
    cond = torch.where(alpha >= a_min, unnorm / alpha.clamp_min(1e-8), torch.zeros_like(unnorm))
    pkg["label_conditional"] = cond.clamp(0, 1)
    pkg["coverage_mask"] = (alpha >= a_min).float()
    pkg["label"] = pkg["label_conditional"]
    return pkg


def render_label_conditional(
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    a_min: float = 0.05,
    scaling_modifier: float = 1.0,
) -> Dict:
    """Render + convert to conditional labels with coverage mask (one raster pass)."""
    pkg = render_label_alpha(
        viewpoint_cam, gaussians, pipe, background, scaling_modifier=scaling_modifier
    )
    return ensure_conditional(pkg, viewpoint_cam, gaussians, pipe, background, a_min=a_min)


def mask_bce_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on valid pixels only."""
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    w = valid
    if w.sum() < 1:
        return pred.new_zeros(())
    loss = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def soft_iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    w = valid > 0.5
    if w.sum() < 1:
        return 0.0
    p = pred[w]
    t = target[w]
    inter = (p * t).sum()
    union = (p + t - p * t).sum().clamp_min(1e-6)
    return float((inter / union).item())


def hard_iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, thr: float = 0.5) -> float:
    w = valid > 0.5
    if w.sum() < 1:
        return 0.0
    p = (pred[w] >= thr)
    t = (target[w] >= 0.5)
    inter = (p & t).sum().float()
    union = (p | t).sum().float().clamp_min(1.0)
    return float((inter / union).item())


@torch.no_grad()
def evaluate_mask_metrics(
    cameras,
    gaussians,
    pipe,
    background,
    max_views: int = 8,
    a_min: float = 0.05,
):
    """Train-view soft/hard IoU on masked cameras (P1 diagnostic)."""
    cams = [c for c in cameras if getattr(c, "has_object_mask", False)][:max_views]
    soft, hard = [], []
    for cam in cams:
        pkg = render_label_conditional(cam, gaussians, pipe, background, a_min=a_min)
        valid = cam.mask_valid * pkg["coverage_mask"]
        soft.append(soft_iou(pkg["label"], cam.object_mask, valid))
        hard.append(hard_iou(pkg["label"], cam.object_mask, valid))
    return {
        "soft_iou": float(sum(soft) / max(1, len(soft))),
        "hard_iou": float(sum(hard) / max(1, len(hard))),
        "num_views": len(cams),
    }
