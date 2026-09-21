"""
Re-evaluate exported G_obj under the unified protocol (no retraining).

python tools/eval_evi_protocol.py --scene_out output/evi/bear --source data/bear
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from arguments import ModelParams, PipelineParams
from evi.data.split_protocol import load_split
from evi.data.training_loader import attach_object_masks
from evi.evaluation.protocol import TAU_ALPHA, evaluate_gobj_protocol, summarize_scene_objects
from evi.export.diagnostics import save_diagnostics
from scene import GaussianModel, Scene
from argparse import ArgumentParser, Namespace


def eval_object(source, model_path, object_id, split_json, data_device="cpu"):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    args.source_path = os.path.abspath(source)
    args.model_path = os.path.abspath(model_path)
    args.data_device = data_device
    args.eval = True
    dataset = lp.extract(args)
    dataset.source_path = args.source_path
    dataset.model_path = args.model_path
    dataset.data_device = data_device
    dataset.eval = True
    pipe = pp.extract(args)

    ply = os.path.join(model_path, "point_cloud_obj.ply")
    if not os.path.isfile(ply):
        ply = os.path.join(model_path, "object.ply")
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, GaussianModel(dataset.sh_degree), shuffle=False, load_ply_path=None)
    # Scene above would create_from_pcd; we only need cameras. Reload via eval split.
    # Use already-constructed cameras from a dummy load:
    # Re-create Scene without ply by eval=True then ignore gaussians.
    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    split = load_split(split_json)
    test_names = set(split["test_ids"])
    test_cams = [c for c in all_cams if c.image_name in test_names]
    masks_folder = os.path.join(source, "object_masks")
    attach_object_masks(test_cams, masks_folder, int(object_id))
    g.load_ply(ply)
    proto = evaluate_gobj_protocol(test_cams, g, pipe, tau_alpha=TAU_ALPHA, object_id=int(object_id))
    save_diagnostics(os.path.join(model_path, "metrics.json"), proto)
    return proto


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_out", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--data_device", default="cpu")
    args = ap.parse_args()
    split_json = os.path.join(args.scene_out, "split.json")
    reports = []
    for name in sorted(os.listdir(args.scene_out)):
        obj_dir = os.path.join(args.scene_out, name)
        if not os.path.isdir(obj_dir):
            continue
        if not (os.path.isfile(os.path.join(obj_dir, "object.ply")) or os.path.isfile(os.path.join(obj_dir, "point_cloud_obj.ply"))):
            continue
        try:
            oid = int(name)
        except ValueError:
            continue
        print(f"eval object {oid}")
        reports.append(eval_object(args.source, obj_dir, oid, split_json, args.data_device))
    scene_id = os.path.basename(os.path.abspath(args.scene_out))
    summary = summarize_scene_objects(reports, scene_id)
    save_diagnostics(os.path.join(args.scene_out, "scene_metrics.json"), summary)
    print(summary)
