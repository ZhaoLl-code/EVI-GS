"""Hard extract object Gaussians to standard 3DGS PLY."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p


def hard_extract_mask(
    gaussians,
    tau: float = 0.5,
    eta: Optional[torch.Tensor] = None,
    retain: Optional[torch.Tensor] = None,
    tau_r: float = 0.5,
) -> torch.Tensor:
    """h = (z >= tau) [& eta] [& r >= tau_r]."""
    m = gaussians.get_label.squeeze(-1) >= float(tau)
    if eta is not None:
        m = m & eta.bool().to(device=m.device)
    if retain is not None:
        m = m & (retain.reshape(-1) >= float(tau_r))
    return m


def save_object_ply(
    gaussians,
    path: str,
    tau: float = 0.5,
    mask: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    retain: Optional[torch.Tensor] = None,
    tau_r: float = 0.5,
) -> int:
    """Write extracted object as a standard Gaussian PLY (with label_z)."""
    if mask is None:
        mask = hard_extract_mask(gaussians, tau=tau, eta=eta, retain=retain, tau_r=tau_r)
    mask = mask.bool()
    n = int(mask.sum().item())
    n_all = int(mask.numel())
    if n == 0:
        raise RuntimeError(f"Object extract produced 0 Gaussians at tau={tau}")
    if n == n_all:
        z = gaussians.get_label.detach().reshape(-1)
        raise RuntimeError(
            f"Object extract kept ALL {n_all} Gaussians at tau={tau} "
            f"(label mean={float(z.mean()):.4f}, frac>=tau={float((z >= tau).float().mean()):.4f}). "
            "Likely label_z was not loaded (e.g. wrong checkpoint without labels)."
        )
    mkdir_p(os.path.dirname(path) or ".")

    m = mask.detach().cpu().numpy()
    xyz = gaussians._xyz.detach().cpu().numpy()[m]
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[m]
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[m]
    opacities = gaussians._opacity.detach().cpu().numpy()[m]
    scale = gaussians._scaling.detach().cpu().numpy()[m]
    rotation = gaussians._rotation.detach().cpu().numpy()[m]
    label_z = gaussians.get_label.detach().cpu().numpy()[m]

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    attrs += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    attrs += ["opacity"]
    attrs += [f"scale_{i}" for i in range(scale.shape[1])]
    attrs += [f"rot_{i}" for i in range(rotation.shape[1])]
    attrs += ["label_z"]

    dtype_full = [(a, "f4") for a in attrs]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    data = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, label_z), axis=1)
    elements[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    print(f"[EVI] saved object.ply n={n}/{n_all} ({100.0 * n / n_all:.2f}%) tau={tau}")
    return n
