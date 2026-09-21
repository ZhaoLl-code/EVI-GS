"""
Unified 3DGS object-segmentation metrics (scheme §5–§8).

Prediction masks come from independent G_obj alpha rendering, not scene label maps.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evi.render.label_renderer import render_alpha
from gaussian_renderer import render
from utils.loss_utils import create_window

PROTOCOL_NAME = "3dgs_seg_metrics_v20260912"
TAU_ALPHA = 0.5
LPIPS_NET = "vgg"
BACKGROUND = "black"
MIN_FG_FIDELITY = 100
METHOD_NAME = "EVI-GS"
METHOD_TYPE = "C"
STRIDE = 8

_LPIPS = None
_LPIPS_DEVICE = None


def protocol_config() -> Dict:
    cfg = {
        "protocol": PROTOCOL_NAME,
        "method": METHOD_NAME,
        "type": METHOD_TYPE,
        "mask_source": "dataset_gt",
        "split": "uniform_stride",
        "stride": STRIDE,
        "tau_alpha": TAU_ALPHA,
        "lpips_net": LPIPS_NET,
        "background": BACKGROUND,
        "min_fg_fidelity_pixels": MIN_FG_FIDELITY,
        "ssim_window": 11,
        "ssim_sigma": 1.5,
        "boundary_d_rule": "max(1, floor(0.01 * min(H,W)))",
        "fidelity_domain": "gt_foreground_mask_crop_black_bg",
    }
    blob = json.dumps(cfg, sort_keys=True)
    cfg["config_hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return cfg


def boundary_width(h: int, w: int) -> int:
    return max(1, int(np.floor(0.01 * min(int(h), int(w)))))


def _to_numpy_mask(t: torch.Tensor) -> np.ndarray:
    x = t.detach()
    if x.ndim == 3:
        x = x[0]
    return (x > 0.5).to(torch.uint8).cpu().numpy()


def iou_np(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return None
    return float(inter / union)


def pixel_acc_np(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred.astype(bool) == gt.astype(bool)).mean())


def boundary_band(mask: np.ndarray, d: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = 2 * int(d) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(m, kernel)
    ero = cv2.erode(m, kernel)
    return (dil > 0) & (ero == 0)


def boundary_iou_np(pred: np.ndarray, gt: np.ndarray, d: int) -> Optional[float]:
    bp = boundary_band(pred, d)
    bg = boundary_band(gt, d)
    union = np.logical_or(bp, bg).sum()
    if union == 0:
        return None
    inter = np.logical_and(bp, bg).sum()
    return float(inter / union)


def ssim_map_hw(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Per-pixel SSIM map averaged over channels. img: (3,H,W) in [0,1]."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    channel = img1.size(1)
    window = create_window(window_size, channel)
    window = window.to(device=img1.device, dtype=img1.dtype)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean(dim=1).squeeze(0).clamp(-1.0, 1.0)


def get_lpips(device: torch.device, net: str = LPIPS_NET):
    global _LPIPS, _LPIPS_DEVICE
    import os

    hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    if net == "vgg":
        ckpt = os.path.join(hub, "vgg16-397923af.pth")
        min_bytes = 500_000_000
    else:
        ckpt = os.path.join(hub, "alexnet-owt-7be5be79.pth")
        min_bytes = 200_000_000
    if (not os.path.isfile(ckpt)) or os.path.getsize(ckpt) < min_bytes:
        raise RuntimeError(f"LPIPS backbone weights missing or incomplete: {ckpt}")
    if _LPIPS is None or _LPIPS_DEVICE != device:
        from lpipsPyTorch.modules.lpips import LPIPS

        _LPIPS = LPIPS(net_type=net, version="0.1").to(device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
        _LPIPS_DEVICE = device
    return _LPIPS


@torch.no_grad()
def render_gobj_rgb_alpha(cam, gaussians, pipe) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent G_obj render: RGB over black, accumulated alpha."""
    device = gaussians.get_xyz.device
    bg0 = torch.zeros(3, device=device, dtype=torch.float32)
    rgb = render(cam, gaussians, pipe, bg0)["render"].clamp(0, 1)
    alpha = render_alpha(cam, gaussians, pipe, bg0)
    return rgb, alpha


def _cam_gt_cuda(cam) -> torch.Tensor:
    img = cam.original_image
    if img.device.type != "cuda":
        img = img.cuda()
    return img.clamp(0, 1)


def _cam_mask_cuda(cam) -> torch.Tensor:
    m = cam.object_mask
    if m.device.type != "cuda":
        m = m.cuda()
    return m


@torch.no_grad()
def evaluate_gobj_protocol(
    test_cameras: List,
    gaussians,
    pipe,
    tau_alpha: float = TAU_ALPHA,
    object_id: int = -1,
    n_obj: Optional[int] = None,
) -> Dict:
    """
    Metrics on V_test^{+,(o)} using independent G_obj rendering.
    Empty-FG / missing-mask test views are skipped (optional hallucination logged).
    """
    cfg = protocol_config()
    if n_obj is None:
        n_obj = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    lpips_fn = None
    try:
        lpips_fn = get_lpips(device)
    except Exception as exc:
        print(f"[protocol] LPIPS unavailable ({exc}); SSIM still reported")

    per_view = []
    skipped = []
    hallucination = []
    ious, accs, bious, ssims, lpips_vals = [], [], [], [], []

    for cam in test_cameras:
        name = cam.image_name
        present = bool(getattr(cam, "has_object_mask", False))
        if not present:
            skipped.append({"name": name, "reason": "no_mask"})
            continue
        gt_m = _cam_mask_cuda(cam)
        fg = int((gt_m > 0.5).sum().item())
        h, w = int(cam.image_height), int(cam.image_width)
        d = boundary_width(h, w)

        rgb, alpha = render_gobj_rgb_alpha(cam, gaussians, pipe)
        pred_m = (alpha >= float(tau_alpha)).float()
        pred_np = _to_numpy_mask(pred_m)
        gt_np = _to_numpy_mask(gt_m)

        if fg == 0:
            hall = float(pred_np.mean())
            hallucination.append({"name": name, "hallucination": hall})
            skipped.append({"name": name, "reason": "empty_gt", "hallucination": hall})
            continue

        iou = iou_np(pred_np, gt_np)
        if iou is None:
            skipped.append({"name": name, "reason": "empty_union"})
            continue
        acc = pixel_acc_np(pred_np, gt_np)
        biou = boundary_iou_np(pred_np, gt_np, d)

        rec = {
            "name": name,
            "fg_pixels": fg,
            "iou": iou,
            "acc": acc,
            "boundary_d": d,
        }
        if biou is not None:
            rec["biou"] = biou
            bious.append(biou)
        ious.append(iou)
        accs.append(acc)

        if fg >= MIN_FG_FIDELITY:
            gt_rgb = _cam_gt_cuda(cam)
            m3 = (gt_m > 0.5).float()
            if m3.shape[0] != 3:
                m3c = m3.expand_as(gt_rgb)
            else:
                m3c = m3
            hat = rgb * m3c
            ref = gt_rgb * m3c
            smap = ssim_map_hw(hat, ref)
            ssim_v = float(smap[gt_np.astype(bool)].mean().item()) if gt_np.any() else None
            if ssim_v is not None:
                rec["ssim"] = ssim_v
                ssims.append(ssim_v)
            if lpips_fn is not None:
                lp = lpips_fn(hat.unsqueeze(0), ref.unsqueeze(0))
                lp_v = float(lp.reshape(-1)[0].item())
                rec["lpips"] = lp_v
                lpips_vals.append(lp_v)
        else:
            rec["fidelity_skipped"] = True

        per_view.append(rec)

    n_eval = len(ious)
    out = {
        "protocol": cfg,
        "object_id": int(object_id),
        "num_eval_views": n_eval,
        "num_test_cameras": len(test_cameras),
        "N_obj": int(n_obj),
        "mIoU": float(sum(ious) / n_eval) if n_eval else None,
        "mAcc": float(sum(accs) / n_eval) if n_eval else None,
        "mBIoU": float(sum(bious) / len(bious)) if bious else None,
        "SSIM": float(sum(ssims) / len(ssims)) if ssims else None,
        "LPIPS": float(sum(lpips_vals) / len(lpips_vals)) if lpips_vals else None,
        "num_fidelity_views": len(ssims),
        "skipped": skipped,
        "hallucination_mean": (
            float(sum(x["hallucination"] for x in hallucination) / len(hallucination))
            if hallucination
            else None
        ),
        "per_view": per_view,
    }
    if n_eval == 0:
        out["skip_reason"] = "no_evaluable_test_views"
    return out


def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def summarize_scene_objects(object_reports: List[Dict], scene_id: str) -> Dict:
    valid = [r for r in object_reports if int(r.get("num_eval_views") or 0) >= 1]
    skipped = [
        {"object_id": r.get("object_id"), "reason": r.get("skip_reason", "num_eval_views=0")}
        for r in object_reports
        if int(r.get("num_eval_views") or 0) < 1
    ]
    n = len(valid)
    return {
        "protocol": protocol_config(),
        "scene_id": scene_id,
        "method": METHOD_NAME,
        "type": METHOD_TYPE,
        "num_objects_total": len(object_reports),
        "num_objects_eval": n,
        "skipped_objects": skipped,
        "mIoU": mean_or_none([r.get("mIoU") for r in valid]),
        "mAcc": mean_or_none([r.get("mAcc") for r in valid]),
        "mBIoU": mean_or_none([r.get("mBIoU") for r in valid]),
        "SSIM": mean_or_none([r.get("SSIM") for r in valid]),
        "LPIPS": mean_or_none([r.get("LPIPS") for r in valid]),
        "N_obj": mean_or_none([r.get("N_obj") for r in valid]),
        "objects": [
            {
                "object_id": r.get("object_id"),
                "num_eval_views": r.get("num_eval_views"),
                "mIoU": r.get("mIoU"),
                "mAcc": r.get("mAcc"),
                "mBIoU": r.get("mBIoU"),
                "SSIM": r.get("SSIM"),
                "LPIPS": r.get("LPIPS"),
                "N_obj": r.get("N_obj"),
            }
            for r in object_reports
        ],
    }
