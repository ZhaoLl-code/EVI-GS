"""
Brute-force 3DGS object segmentation by multi-view mask voting.

Standalone diagnostic tool (NOT part of the EVI training pipeline).

For each Gaussian center:
  1) project into every COLMAP camera;
  2) if it lands in-image with positive depth, check whether that pixel
     belongs to the selected object_id in the multi-label mask;
  3) keep the Gaussian when it has enough positive votes / positive ratio.

Example (data/bear, object_id=34):
  set PYTHONPATH=%CD%
  python tools/bruteforce_mask_segment.py ^
    --ply output/bear/point_cloud/iteration_30000/point_cloud.ply ^
    --source data/bear --object_id 34 ^
    --out output/bruteforce/bear30k_id34.ply
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Allow running from repo root without installing the package.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evi.data.mask_loader import load_binary_mask, resolve_mask_path, auto_select_object_id
from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import getWorld2View2


def load_xyz_from_ply(path: str) -> Tuple[PlyData, np.ndarray]:
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    return ply, xyz


def save_filtered_ply(ply: PlyData, keep: np.ndarray, out_path: str) -> int:
    """Keep a subset of vertices; preserve all original properties."""
    keep = np.asarray(keep, dtype=bool)
    n = int(keep.sum())
    if n == 0:
        raise RuntimeError("Brute-force segment kept 0 Gaussians.")
    if n == keep.size:
        print("[bruteforce] WARNING: kept ALL Gaussians (mask may be wrong / criteria too loose).")

    src = ply["vertex"].data
    filtered = src[keep]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    PlyData([PlyElement.describe(filtered, "vertex")], text=False).write(out_path)
    return n


def build_view_matrices(cam_infos) -> List[dict]:
    views = []
    for c in cam_infos:
        w2c = getWorld2View2(c.R, c.T)  # 4x4
        # Match Camera: store transposed W2V so homo @ M works.
        wvt = torch.tensor(w2c.T, dtype=torch.float32)
        views.append(
            {
                "name": c.image_name,
                "world_view": wvt,
                "FoVx": float(c.FovX),
                "FoVy": float(c.FovY),
                "width": int(c.width),
                "height": int(c.height),
            }
        )
    return views


@torch.no_grad()
def vote_centers(
    xyz: np.ndarray,
    views: List[dict],
    masks_folder: str,
    object_id: int,
    device: str = "cuda",
    batch_size: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pos_votes: (N,) int — views where center hits FG mask
      vis_votes: (N,) int — views where center is in front & in image
      hit_ratio: (N,) float — pos/vis (0 if never visible)
    """
    N = xyz.shape[0]
    pos = torch.zeros(N, dtype=torch.int32, device=device)
    vis = torch.zeros(N, dtype=torch.int32, device=device)
    xyz_t = torch.from_numpy(xyz).to(device)

    for vi, view in enumerate(views):
        path = resolve_mask_path(masks_folder, view["name"])
        binary, valid, present = load_binary_mask(
            path, object_id, resolution=(view["width"], view["height"])
        )
        if not present or not valid.any():
            continue
        # FG where object_id matches and annotation valid
        fg = (binary > 0.5) & (valid > 0.5)
        if not fg.any():
            continue
        fg_t = torch.from_numpy(fg.astype(np.uint8)).to(device)  # H,W
        H, W = fg_t.shape
        wvt = view["world_view"].to(device)
        tanx = float(np.tan(0.5 * view["FoVx"]))
        tany = float(np.tan(0.5 * view["FoVy"]))

        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            pts = xyz_t[s:e]
            ones = torch.ones((e - s, 1), device=device, dtype=pts.dtype)
            view_xyz = torch.cat([pts, ones], dim=1) @ wvt  # (B,4)
            x, y, z = view_xyz[:, 0], view_xyz[:, 1], view_xyz[:, 2]
            in_front = z > 1e-4
            u = (x / (z.clamp_min(1e-6) * tanx) + 1.0) * 0.5
            v = (y / (z.clamp_min(1e-6) * tany) + 1.0) * 0.5
            in_img = in_front & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            if not in_img.any():
                continue
            vis[s:e] += in_img.to(torch.int32)

            px = (u * W).long().clamp(0, W - 1)
            py = (v * H).long().clamp(0, H - 1)
            hit = in_img & (fg_t[py, px] > 0)
            pos[s:e] += hit.to(torch.int32)

        if (vi + 1) % 20 == 0 or vi == 0:
            print(f"[bruteforce] projected view {vi + 1}/{len(views)} ({view['name']})")

    pos_np = pos.cpu().numpy()
    vis_np = vis.cpu().numpy()
    ratio = np.zeros(N, dtype=np.float32)
    m = vis_np > 0
    ratio[m] = pos_np[m].astype(np.float32) / vis_np[m].astype(np.float32)
    return pos_np, vis_np, ratio


def decide_keep(
    pos: np.ndarray,
    vis: np.ndarray,
    ratio: np.ndarray,
    min_pos_views: int,
    min_pos_ratio: float,
    require_visible: bool = True,
) -> np.ndarray:
    keep = (pos >= min_pos_views) & (ratio >= min_pos_ratio)
    if require_visible:
        keep &= vis > 0
    return keep


def main():
    ap = argparse.ArgumentParser(description="Brute-force mask-voting 3DGS segmentation (standalone).")
    ap.add_argument("--ply", required=True, help="Input 3DGS point_cloud.ply")
    ap.add_argument("--source", required=True, help="COLMAP scene folder (e.g. data/scene)")
    ap.add_argument("--images", default="images", help="Images subfolder name")
    ap.add_argument("--masks", default="object_masks", help="Multi-label mask folder under source")
    ap.add_argument("--object_id", type=int, default=34, help="Target id; <0 = auto largest area")
    ap.add_argument("--out", required=True, help="Output filtered PLY path")
    ap.add_argument("--min_pos_views", type=int, default=3, help="Min views where center hits FG mask")
    ap.add_argument("--min_pos_ratio", type=float, default=0.5, help="Min FG fraction among visible views")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=500_000)
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    masks_folder = args.masks if os.path.isabs(args.masks) else os.path.join(source, args.masks)
    if not os.path.isdir(masks_folder):
        raise FileNotFoundError(f"Masks folder not found: {masks_folder}")

    print(f"[bruteforce] loading COLMAP from {source}")
    scene_info = readColmapSceneInfo(source, args.images, "", eval=False, train_test_exp=False)
    cam_infos = list(scene_info.train_cameras) + list(scene_info.test_cameras)
    print(f"[bruteforce] cameras={len(cam_infos)}")

    object_id = int(args.object_id)
    if object_id < 0:
        object_id = auto_select_object_id(masks_folder, [c.image_name for c in cam_infos])
    print(f"[bruteforce] object_id={object_id}")

    print(f"[bruteforce] loading PLY {args.ply}")
    ply, xyz = load_xyz_from_ply(args.ply)
    print(f"[bruteforce] N={xyz.shape[0]}")

    views = build_view_matrices(cam_infos)
    pos, vis, ratio = vote_centers(
        xyz,
        views,
        masks_folder,
        object_id,
        device=args.device,
        batch_size=args.batch_size,
    )
    keep = decide_keep(pos, vis, ratio, args.min_pos_views, args.min_pos_ratio)
    n_keep = int(keep.sum())
    print(
        f"[bruteforce] votes: pos mean={pos.mean():.2f} vis mean={vis.mean():.2f} "
        f"ratio mean={ratio.mean():.3f} | keep={n_keep}/{len(keep)} "
        f"({100.0 * n_keep / max(1, len(keep)):.2f}%) "
        f"[min_pos_views={args.min_pos_views}, min_pos_ratio={args.min_pos_ratio}]"
    )

    n = save_filtered_ply(ply, keep, args.out)
    meta = args.out + ".vote.npz"
    np.savez_compressed(
        meta,
        keep=keep,
        pos_votes=pos,
        vis_votes=vis,
        hit_ratio=ratio,
        object_id=object_id,
        min_pos_views=args.min_pos_views,
        min_pos_ratio=args.min_pos_ratio,
    )
    print(f"[bruteforce] wrote {args.out} (n={n}) and {meta}")


if __name__ == "__main__":
    main()
