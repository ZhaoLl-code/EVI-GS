#
# Extract object PLY and protocol metrics from a trained EVI checkpoint.
#

import json
import os
import shutil
from argparse import ArgumentParser

import torch
from plyfile import PlyData

from arguments import ModelParams, PipelineParams, get_combined_args
from evi.data.mask_loader import auto_select_object_id
from evi.data.split_protocol import load_split
from evi.data.training_loader import attach_object_masks
from evi.evaluation.protocol import TAU_ALPHA, evaluate_gobj_protocol
from evi.export.diagnostics import label_range_summary, save_diagnostics
from evi.export.object_ply import hard_extract_mask, save_object_ply
from evi.render.label_renderer import evaluate_mask_metrics
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration


def find_latest_labeled_iteration(model_path: str) -> int:
    """Prefer the latest point_cloud iteration that contains label_z."""
    root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    candidates = []
    for name in os.listdir(root):
        if not name.startswith("iteration_"):
            continue
        try:
            it = int(name.split("_")[-1])
        except ValueError:
            continue
        ply = os.path.join(root, name, "point_cloud.ply")
        if not os.path.isfile(ply):
            continue
        props = [p.name for p in PlyData.read(ply)["vertex"].properties]
        if "label_z" in props:
            candidates.append(it)
    if candidates:
        return max(candidates)
    return searchForMaxIteration(root)


def main():
    parser = ArgumentParser(description="EVI-GS extract")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    load_iter = args.iteration
    if load_iter == -1:
        load_iter = find_latest_labeled_iteration(dataset.model_path)
        print(f"[EVI] auto-selected labeled iteration {load_iter}")
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=False)

    split_path = os.path.join(dataset.model_path, "split.json")
    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    if os.path.isfile(split_path):
        split = load_split(split_path)
        test_names = set(split.get("test_ids") or split.get("test") or [])
        train_cams = [c for c in all_cams if c.image_name not in test_names]
        test_cams = [c for c in all_cams if c.image_name in test_names]
    else:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()

    masks_name = getattr(dataset, "masks", "object_masks")
    masks_folder = os.path.join(dataset.source_path, masks_name) if masks_name else ""
    object_id = getattr(dataset, "object_id", -1)
    if object_id < 0 and masks_folder:
        object_id = auto_select_object_id(masks_folder, [c.image_name for c in train_cams])
    attach_object_masks(train_cams + test_cams, masks_folder, object_id)

    bg = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    out_ply = os.path.join(dataset.model_path, "object.ply")
    n = save_object_ply(gaussians, out_ply, tau=args.tau, eta=None)
    gobj_ply = os.path.join(dataset.model_path, "point_cloud_obj.ply")
    shutil.copy2(out_ply, gobj_ply)
    mask = hard_extract_mask(gaussians, args.tau, eta=None)

    proto = None
    try:
        g_obj = GaussianModel(dataset.sh_degree)
        g_obj.load_ply(gobj_ply)
        proto = evaluate_gobj_protocol(
            test_cams, g_obj, pipe, tau_alpha=TAU_ALPHA, object_id=int(object_id), n_obj=int(n)
        )
        save_diagnostics(os.path.join(dataset.model_path, "metrics.json"), proto)
        del g_obj
        torch.cuda.empty_cache()
    except Exception as exc:
        proto = {"object_id": int(object_id), "num_eval_views": 0, "error": str(exc), "N_obj": int(n)}
        save_diagnostics(os.path.join(dataset.model_path, "metrics.json"), proto)
        print(f"[EVI] protocol eval failed: {exc}")

    report = {
        "object_id": object_id,
        "iteration": scene.loaded_iter,
        "tau": args.tau,
        "object_ply": out_ply,
        "num_object_gaussians": n,
        "N_obj": int(mask.sum().item()) if mask is not None else int(n),
        "P1": evaluate_mask_metrics(train_cams, gaussians, pipe, bg),
        "protocol": proto,
        "labels": label_range_summary(gaussians),
    }
    save_diagnostics(os.path.join(dataset.model_path, "extract_report.json"), report)
    print(json.dumps({k: report[k] for k in report if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
