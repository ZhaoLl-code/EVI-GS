"""
Run EVI-GS on the bundled sample scenes (data/bear, data/truck).

Pipeline: support_init, optional learn_tau, optional post_tau_admission
(default 1+2+3). Disable with --no_learn_export_tau / --no_post_tau_admission.

Default:
  source  = <repo>/data/<scene>
  warmup  = <repo>/output/<scene>   (pre-trained 3DGS)
  output  = <repo>/output/evi/<scene>/<object_id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evi.data.mask_loader import discover_object_ids, read_id_mask, resolve_mask_path
from evi.data.split_protocol import build_stride_split, list_colmap_image_names, load_split, save_split
from evi.evaluation.protocol import STRIDE, protocol_config, summarize_scene_objects
from evi.export.diagnostics import save_diagnostics

UNTRAINED_DEFAULT = os.path.join(_REPO, "data")
TRAINED_DEFAULT = os.path.join(_REPO, "output")
OUT_DEFAULT = os.path.join(_REPO, "output", "evi")
LOCKED_SPLIT_DEFAULT = os.path.join(_REPO, "output")
METHOD_NAME = "EVI-GS"

SCENES = [
    "bear",
    "truck",
]


def log(fp, msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def cpu_device_for(scene: str, n_images: int) -> bool:
    return n_images > 80


def object_has_split_fg(masks_folder: str, names: list, object_id: int) -> bool:
    for name in names:
        path = resolve_mask_path(masks_folder, name)
        if not os.path.isfile(path):
            continue
        try:
            m = read_id_mask(path)
        except Exception:
            continue
        if (m == int(object_id)).any():
            return True
    return False


def object_done(obj_dir: Path) -> bool:
    mp = obj_dir / "metrics.json"
    meta = obj_dir / "object_meta.json"
    ply_ok = (obj_dir / "point_cloud_obj.ply").is_file() or (obj_dir / "object.ply").is_file()
    if not mp.is_file() or not meta.is_file() or not ply_ok:
        return False
    try:
        with open(mp, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    if payload.get("skip_reason") or payload.get("error"):
        return True
    return True


def write_or_load_split(scene_out: Path, source: str, scene: str, locked_root: str) -> dict:
    split_path = scene_out / "split.json"
    if split_path.is_file():
        return load_split(str(split_path))
    locked = Path(locked_root) / scene / "split.json"
    if locked.is_file() and locked.resolve() != split_path.resolve():
        shutil.copy2(locked, split_path)
        return load_split(str(split_path))
    names = list_colmap_image_names(source)
    split = build_stride_split(names, stride=STRIDE)
    save_split(split, str(split_path))
    return split


def refresh_scene_summary(scene_out: Path, scene: str) -> dict:
    reports = []
    for child in sorted(scene_out.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        mp = child / "metrics.json"
        if not mp.is_file():
            continue
        with open(mp, "r", encoding="utf-8") as f:
            reports.append(json.load(f))
    summary = summarize_scene_objects(reports, scene)
    summary["method"] = METHOD_NAME
    save_diagnostics(str(scene_out / "scene_metrics.json"), summary)
    return summary


def mean_or_none(vals):
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def aggregate_global(out_root: Path, scenes: list) -> dict:
    scene_rows = []
    object_rows = []
    for scene in scenes:
        sp = out_root / scene / "scene_metrics.json"
        if not sp.is_file():
            continue
        with open(sp, "r", encoding="utf-8") as f:
            scene_json = json.load(f)
        scene_rows.append(scene_json)
        for obj in scene_json.get("objects") or []:
            if int(obj.get("num_eval_views") or 0) >= 1:
                object_rows.append(obj)

    valid_scenes = [s for s in scene_rows if int(s.get("num_objects_eval") or 0) >= 1]

    def scene_mean(key):
        return mean_or_none([s.get(key) for s in valid_scenes])

    def object_mean(key):
        return mean_or_none([o.get(key) for o in object_rows])

    cfg = protocol_config()
    cfg["method"] = METHOD_NAME
    summary = {
        "protocol": cfg,
        "method": METHOD_NAME,
        "type": "C",
        "mask_source": "dataset_gt",
        "num_scenes": len(scene_rows),
        "num_scenes_eval": len(valid_scenes),
        "num_objects_eval": len(object_rows),
        "scene_macro": {
            "mIoU": scene_mean("mIoU"),
            "mAcc": scene_mean("mAcc"),
            "mBIoU": scene_mean("mBIoU"),
            "SSIM": scene_mean("SSIM"),
            "LPIPS": scene_mean("LPIPS"),
            "N_obj": scene_mean("N_obj"),
        },
        "object_macro": {
            "mIoU": object_mean("mIoU"),
            "mAcc": object_mean("mAcc"),
            "mBIoU": object_mean("mBIoU"),
            "SSIM": object_mean("SSIM"),
            "LPIPS": object_mean("LPIPS"),
            "N_obj": object_mean("N_obj"),
        },
        "mIoU": object_mean("mIoU"),
        "mAcc": object_mean("mAcc"),
        "mBIoU": object_mean("mBIoU"),
        "SSIM": object_mean("SSIM"),
        "LPIPS": object_mean("LPIPS"),
        "N_obj": object_mean("N_obj"),
        "scenes": [
            {
                "scene_id": s.get("scene_id"),
                "num_objects_eval": s.get("num_objects_eval"),
                "mIoU": s.get("mIoU"),
                "mAcc": s.get("mAcc"),
                "mBIoU": s.get("mBIoU"),
                "SSIM": s.get("SSIM"),
                "LPIPS": s.get("LPIPS"),
                "N_obj": s.get("N_obj"),
            }
            for s in scene_rows
        ],
    }
    save_diagnostics(str(out_root / "protocol_summary.json"), summary)
    return summary


def run_object(
    python: str,
    source: str,
    trained: str,
    obj_dir: Path,
    split_path: Path,
    object_id: int,
    data_device: str,
    extra_args: list,
) -> int:
    cmd = [
        python,
        os.path.join(_REPO, "train_evi.py"),
        "-s",
        source,
        "-m",
        str(obj_dir),
        "--warmup_from",
        trained,
        "--object_id",
        str(object_id),
        "--split_json",
        str(split_path),
        "--data_device",
        data_device,
    ] + extra_args
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    obj_dir.mkdir(parents=True, exist_ok=True)
    log_path = obj_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=_REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--untrained", default=UNTRAINED_DEFAULT)
    ap.add_argument("--trained", default=TRAINED_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--locked_split_root", default=LOCKED_SPLIT_DEFAULT)
    ap.add_argument("--scenes", default="", help="comma-separated subset; empty = bear,truck")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max_objects", type=int, default=0)
    ap.add_argument("--data_device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--dry_run", action="store_true")
    args, extra = ap.parse_known_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes else list(SCENES)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "batch_evi.log"
    with open(log_path, "a", encoding="utf-8") as lf:
        log(lf, f"==== EVI BATCH START scenes={len(scenes)} extra={extra} out={out_root}")
        cfg = protocol_config()
        cfg["method"] = METHOD_NAME
        save_diagnostics(str(out_root / "protocol_config.json"), cfg)

        for scene in scenes:
            source = os.path.join(args.untrained, scene)
            trained = os.path.join(args.trained, scene)
            scene_out = out_root / scene
            scene_out.mkdir(parents=True, exist_ok=True)
            if not os.path.isdir(os.path.join(source, "images")):
                log(lf, f"[MISS] {scene} no images")
                continue
            ply = os.path.join(trained, "point_cloud", "iteration_30000", "point_cloud.ply")
            if not os.path.isfile(ply) and not os.path.isdir(os.path.join(trained, "point_cloud")):
                log(lf, f"[MISS] {scene} no trained point_cloud")
                continue

            split = write_or_load_split(scene_out, source, scene, args.locked_split_root)
            save_diagnostics(str(scene_out / "protocol_config.json"), cfg)
            masks_folder = os.path.join(source, "object_masks")
            disc = discover_object_ids(masks_folder, split["train_ids"] + split["test_ids"])
            ids = []
            skipped = list(disc.get("skipped") or [])
            for oid in disc.get("object_ids") or []:
                if not object_has_split_fg(masks_folder, split["train_ids"], oid):
                    skipped.append({"object_id": oid, "reason": "no_train_fg"})
                    continue
                ids.append(int(oid))
            if args.max_objects and len(ids) > args.max_objects:
                ids = ids[: args.max_objects]
            save_diagnostics(
                str(scene_out / "objects.json"),
                {
                    "scene_id": scene,
                    "object_ids": ids,
                    "skipped": skipped,
                    "jpeg_collapsed": disc.get("jpeg_collapsed"),
                    "num_mask_files": disc.get("num_mask_files"),
                    "split_hash": split.get("hash"),
                    "method": METHOD_NAME,
                },
            )
            n_img = int(split.get("num_total") or len(split.get("train_ids") or []) + len(split.get("test_ids") or []))
            if args.data_device == "auto":
                data_device = "cpu" if cpu_device_for(scene, n_img) else "cuda"
            else:
                data_device = args.data_device
            log(
                lf,
                f"[SCENE] {scene} train={split.get('num_train')} test={split.get('num_test')} "
                f"objects={ids} device={data_device} hash={split.get('hash')} jpeg_collapsed={disc.get('jpeg_collapsed')}",
            )
            if args.dry_run:
                continue

            fail = 0
            for oid in ids:
                obj_dir = scene_out / str(oid)
                if object_done(obj_dir):
                    log(lf, f"[SKIP] {scene}/{oid} already done")
                    continue
                t0 = time.time()
                log(lf, f"[TRAIN] {scene}/{oid} start")
                rc = run_object(
                    args.python,
                    source,
                    trained,
                    obj_dir,
                    scene_out / "split.json",
                    oid,
                    data_device,
                    extra,
                )
                dt = time.time() - t0
                if rc != 0:
                    fail += 1
                    log(lf, f"[FAIL] {scene}/{oid} rc={rc} {dt:.1f}s")
                else:
                    log(lf, f"[OK] {scene}/{oid} {dt:.1f}s")
            summary = refresh_scene_summary(scene_out, scene)
            log(
                lf,
                f"[SCENE-DONE] {scene} eval={summary.get('num_objects_eval')} "
                f"mIoU={summary.get('mIoU')} fails={fail}",
            )
            glob = aggregate_global(out_root, scenes)
            log(
                lf,
                f"[GLOBAL] scenes={glob.get('num_scenes_eval')} objects={glob.get('num_objects_eval')} "
                f"mIoU={glob.get('mIoU')}",
            )

        glob = aggregate_global(out_root, scenes)
        log(
            lf,
            f"==== EVI BATCH DONE scenes={glob.get('num_scenes_eval')} "
            f"objects={glob.get('num_objects_eval')} mIoU={glob.get('mIoU')}",
        )


if __name__ == "__main__":
    main()
