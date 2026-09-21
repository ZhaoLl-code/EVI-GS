"""Post-segmentation stage: learn a scalar export label_z threshold."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from evi.object_render import render_object
from evi.optimize import CamQueue, set_trainable


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(p)).item())


def cam_has_foreground(cam, min_pixels: int = 1) -> bool:
    m = getattr(cam, "object_mask", None)
    if m is None:
        return False
    return int((m > 0.5).sum().item()) >= int(min_pixels)


def filter_foreground_cameras(cameras: List, min_pixels: int = 1) -> List:
    return [c for c in cameras if cam_has_foreground(c, min_pixels=min_pixels)]


def soft_miou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Differentiable 1 - soft-IoU on valid pixels (pred/target in [0,1])."""
    w = valid.to(dtype=pred.dtype)
    if w.sum() < 1:
        return pred.new_zeros(())
    p = pred.clamp(0, 1)
    t = target.to(dtype=pred.dtype).clamp(0, 1)
    inter = (p * t * w).sum()
    union = ((p + t - p * t) * w).sum().clamp_min(1e-6)
    return 1.0 - inter / union


def soft_gate(z: torch.Tensor, tau: torch.Tensor, temp: float) -> torch.Tensor:
    """Smooth surrogate of 1[z >= tau]; temp>0 smaller => sharper."""
    t = max(float(temp), 1e-4)
    return torch.sigmoid((z - tau) / t)


def render_gated_object_alpha(
    cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    tau: torch.Tensor,
    eta: Optional[torch.Tensor],
    temp: float,
) -> torch.Tensor:
    """
    Soft-threshold object alpha: tilde_o = gate(z,tau) * eta * o.
    z / o / eta are detached so only tau receives gradients.
    """
    z = gaussians.get_label.reshape(-1).detach()
    o = gaussians.get_opacity.reshape(-1).detach()
    gate = soft_gate(z, tau, temp)
    if eta is not None:
        gate = gate * eta.to(device=gate.device, dtype=gate.dtype).reshape(-1).detach()
    tilde = (gate * o).clamp(0, 1)
    pkg = render_object(cam, gaussians, pipe, background, tilde, need_alpha=True)
    return pkg["alpha"]


def _view_soft_miou_loss(cam, gaussians, pipe, background, tau, eta, temp) -> torch.Tensor:
    alpha = render_gated_object_alpha(cam, gaussians, pipe, background, tau, eta, temp)
    gt = cam.object_mask
    if gt.device != alpha.device:
        gt = gt.to(alpha.device)
    valid = cam.mask_valid
    if valid.device != alpha.device:
        valid = valid.to(alpha.device)
    return soft_miou_loss(alpha, gt, valid)


def run_learn_export_tau(
    gaussians,
    cameras: List,
    pipe,
    background: torch.Tensor,
    eta: Optional[torch.Tensor],
    *,
    init_tau: float = 0.9,
    n_steps: int = 64,
    lr: float = 0.01,
    temp: float = 0.02,
    min_views: int = 8,
    views_per_step: int = 8,
    min_fg_pixels: int = 1,
    progress=None,
    tick=None,
) -> Dict:
    """
    Freeze all Gaussian parameters; optimize a single scalar export threshold.

    Runs a fixed number of steps (typically 1P). Each step averages soft
    (1 - mIoU) over K non-empty-FG train views. Gradients are accumulated
    per-view so a large K cannot explode VRAM by retaining all render graphs.

    views_per_step / min_views:
      views_per_step<=0 -> use ALL non-empty-FG cameras in ``cameras``
      views_per_step>0  -> K = min(num_fg, max(min_views, views_per_step))
        (min_views<=0 is treated as 1 when capping)

    Export uses the tau with the best soft_miou seen.
    """
    fg_cams = filter_foreground_cameras(cameras, min_pixels=min_fg_pixels)
    use_all = int(views_per_step) <= 0
    if fg_cams:
        if use_all:
            k = int(len(fg_cams))
        else:
            floor_k = max(1, int(min_views)) if int(min_views) > 0 else 1
            k = min(len(fg_cams), max(floor_k, int(views_per_step)))
        k = max(1, int(k))
    else:
        k = 0
    n_steps = max(0, int(n_steps))

    if not fg_cams or n_steps <= 0:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "no_fg_cameras_or_steps",
            "export_tau": float(init_tau),
            "best_soft_miou": None,
            "best_tau": float(init_tau),
            "num_fg_cameras": int(len(fg_cams)),
            "views_per_step": int(k),
            "min_views": int(min_views),
            "use_all_fg_views": bool(use_all),
            "n_steps": 0,
            "history": [],
        }

    set_trainable(gaussians, False, False, False)
    device = gaussians.get_xyz.device
    tau_logit = torch.nn.Parameter(
        torch.tensor(_logit(init_tau), device=device, dtype=torch.float32)
    )
    opt = torch.optim.Adam([tau_logit], lr=float(lr))
    queue = CamQueue(fg_cams)
    history: List[Dict] = []
    best_soft_miou = -1.0
    best_tau = float(init_tau)
    best_step = 0
    best_logit = float(tau_logit.detach().item())

    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_vals: List[float] = []
        # Per-view backward: keeps peak VRAM ~1 render graph regardless of K.
        # Recompute tau each view so the graph is not freed mid-step.
        for _ in range(k):
            tau = torch.sigmoid(tau_logit)
            cam = queue.next()
            loss_i = _view_soft_miou_loss(
                cam, gaussians, pipe, background, tau, eta, temp
            )
            loss_vals.append(float(loss_i.detach().item()))
            if loss_i.requires_grad:
                (loss_i / float(k)).backward()
        opt.step()

        loss_now = float(sum(loss_vals) / max(1, len(loss_vals)))
        miou_now = float(1.0 - loss_now)
        tau_now = float(torch.sigmoid(tau_logit).detach().clamp(1e-4, 1.0 - 1e-4).item())
        history.append(
            {
                "step": step,
                "tau": tau_now,
                "soft_miou": miou_now,
                "loss": loss_now,
                "n_views": int(len(loss_vals)),
            }
        )

        if miou_now >= best_soft_miou:
            best_soft_miou = miou_now
            best_tau = tau_now
            best_step = step
            best_logit = float(tau_logit.detach().item())

        if progress is not None:
            progress.set_postfix(
                stage="learn_tau",
                tau=f"{tau_now:.4f}",
                miou=f"{miou_now:.4f}",
                best=f"{best_soft_miou:.4f}",
                k=str(len(loss_vals)),
                loss=f"{loss_now:.4f}",
            )
        if tick is not None:
            tick(1)

    with torch.no_grad():
        tau_logit.copy_(torch.tensor(best_logit, device=device, dtype=tau_logit.dtype))
    final_tau = float(torch.sigmoid(tau_logit).detach().clamp(1e-4, 1.0 - 1e-4).item())
    return {
        "enabled": True,
        "skipped": False,
        "export_tau": float(best_tau),
        "best_tau": float(best_tau),
        "best_soft_miou": float(best_soft_miou) if best_soft_miou >= 0 else None,
        "best_step": int(best_step),
        "final_tau": final_tau,
        "init_tau": float(init_tau),
        "n_steps": int(n_steps),
        "stopped_early": False,
        "lr": float(lr),
        "temp": float(temp),
        "min_views": int(min_views),
        "views_per_step": int(k),
        "use_all_fg_views": bool(k >= len(fg_cams)),
        "num_fit_cameras": int(len(cameras)),
        "num_fg_cameras": int(len(fg_cams)),
        "final_soft_miou": history[-1]["soft_miou"] if history else None,
        "history": history[:: max(1, len(history) // 32)] if history else [],
    }
