"""
Re-extract object PLYs from saved labeled checkpoints at a given tau_z.
Does not overwrite the original object.ply (tau=0.5) from training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from arguments import ModelParams, PipelineParams
from evi.data.split_protocol import load_split
from evi.data.training_loader import attach_object_masks
from evi.evaluation.protocol import TAU_ALPHA, evaluate_gobj_protocol, summarize_scene_objects
from evi.export.diagnostics import save_diagnostics
from evi.export.object_ply import save_object_ply
from extract_evi import find_latest_labeled_iteration
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def load_cfg_args(model_path: str) -> Namespace:
    path = os.path.join(model_path, "cfg_args")
    with open(path, "r", encoding="utf-8") as f:
        return eval(f.read(), {"Namespace": Namespace})


def extract_one(obj_dir: Path, tau: float, data_device: str) -> dict:
    cfg = load_cfg_args(str(obj_dir))
    cfg.model_path = str(obj_dir)
    cfg.data_device = data_device
    cfg.eval = True
    object_id = int(cfg.object_id)

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    dummy = parser.parse_args([])
    for k, v in vars(cfg).items():
        if hasattr(dummy, k) or k in vars(lp) or k in vars(pp) or k.lstrip("_") in (
            "source_path", "model_path", "images", "depths", "resolution", "white_background",
        ):
            setattr(dummy, k, v)
    dummy.source_path = cfg.source_path
    dummy.model_path = str(obj_dir)
    dummy.data_device = data_device
    dummy.eval = True
    dummy.white_background = bool(getattr(cfg, "white_background", False))
    dummy.images = getattr(cfg, "images", "images")
    dummy.depths = getattr(cfg, "depths", "")
    dummy.resolution = getattr(cfg, "resolution", -1)
    dummy.sh_degree = getattr(cfg, "sh_degree", 3)
    dummy.masks = getattr(cfg, "masks", "object_masks")
    dummy.object_id = object_id
    dummy.train_test_exp = False

    dataset = lp.extract(dummy)
    dataset.source_path = os.path.abspath(cfg.source_path)
    dataset.model_path = str(obj_dir)
    dataset.eval = True
    dataset.data_device = data_device
    pipe = pp.extract(dummy)

    it = find_latest_labeled_iteration(str(obj_dir))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False)

    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    split_path = os.path.join(str(obj_dir), "split.json")
    scene_split = os.path.join(str(obj_dir.parent), "split.json")
    if os.path.isfile(split_path):
        split = load_split(split_path)
    elif os.path.isfile(scene_split):
        split = load_split(scene_split)
    else:
        split = None
    if split is not None:
        test_names = set(split.get("test_ids") or split.get("test") or [])
        train_cams = [c for c in all_cams if c.image_name not in test_names]
        test_cams = [c for c in all_cams if c.image_name in test_names]
    else:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()

    masks_folder = os.path.join(dataset.source_path, dataset.masks) if dataset.masks else ""
    attach_object_masks(train_cams, masks_folder, object_id)
    attach_object_masks(test_cams, masks_folder, object_id)

    tag = f"tau{tau:g}"
    meta = {
        "object_id": object_id,
        "iteration": int(it),
        "tau_z": float(tau),
    }
    out_ply = os.path.join(str(obj_dir), f"object_{tag}.ply")
    try:
        n_obj = save_object_ply(gaussians, out_ply, tau=float(tau), eta=None, retain=None)
    except Exception as exc:
        meta.update({"num_eval_views": 0, "error": str(exc), "N_obj": 0})
        save_diagnostics(os.path.join(str(obj_dir), f"metrics_{tag}.json"), meta)
        print(f"[extract] object {object_id}: {exc}")
        return meta

    gobj_ply = os.path.join(str(obj_dir), f"point_cloud_obj_{tag}.ply")
    shutil.copy2(out_ply, gobj_ply)
    meta["N_obj"] = int(n_obj)
    save_diagnostics(os.path.join(str(obj_dir), f"extract_{tag}.json"), meta)

    proto = {"object_id": object_id, "num_eval_views": 0, "N_obj": int(n_obj)}
    try:
        g_obj = GaussianModel(dataset.sh_degree)
        g_obj.load_ply(gobj_ply)
        proto = evaluate_gobj_protocol(
            test_cams, g_obj, pipe, tau_alpha=TAU_ALPHA, object_id=object_id, n_obj=int(n_obj)
        )
        proto["tau_z"] = float(tau)
        save_diagnostics(os.path.join(str(obj_dir), f"metrics_{tag}.json"), proto)
        save_diagnostics(os.path.join(str(obj_dir), f"per_view_{tag}.json"), proto.get("per_view", []))
        del g_obj
        torch.cuda.empty_cache()
    except Exception as exc:
        proto = {"object_id": object_id, "num_eval_views": 0, "error": str(exc), "N_obj": int(n_obj), "tau_z": float(tau)}
        save_diagnostics(os.path.join(str(obj_dir), f"metrics_{tag}.json"), proto)
        print(f"[extract] protocol failed object {object_id}: {exc}")
    print(
        f"[extract] object {object_id} tau={tau} N={n_obj} "
        f"mIoU={proto.get('mIoU')}"
    )
    return proto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(_REPO, "output", "evi", "bear"))
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--data_device", default="cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    safe_state(args.quiet)

    root = Path(args.root)
    obj_json = root / "objects.json"
    if obj_json.is_file():
        with open(obj_json, "r", encoding="utf-8") as f:
            ids = [int(x) for x in json.load(f).get("object_ids") or []]
    else:
        ids = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    reports = []
    for oid in ids:
        obj_dir = root / str(oid)
        ply_root = obj_dir / "point_cloud"
        if not ply_root.is_dir():
            print(f"[skip] {oid} no point_cloud")
            continue
        print(f"==== extract object {oid} tau={args.tau} ====")
        reports.append(extract_one(obj_dir, args.tau, args.data_device))
    summary = summarize_scene_objects(reports, root.name)
    summary["tau_z"] = float(args.tau)
    tag = f"tau{args.tau:g}"
    save_diagnostics(str(root / f"scene_metrics_{tag}.json"), summary)
    print(json.dumps({k: summary.get(k) for k in ("scene_id", "tau_z", "num_objects_eval", "mIoU", "mAcc", "mBIoU", "SSIM", "LPIPS", "N_obj")}, indent=2))


if __name__ == "__main__":
    main()
