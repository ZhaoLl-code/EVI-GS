#
# EVI-GS: support_init, optional learn_tau, optional post_tau_admission.
# Valid combos: 1 | 1+2 | 1+2+3. No learn_tau → tau=0.5; no post_tau → z-cut only.
#

import json
import os
import shutil
import sys
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from typing import Optional

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from evi.data.mask_loader import auto_select_object_id
from evi.data.split_protocol import SplitGuard, build_stride_split, load_split, save_split
from evi.data.training_loader import attach_object_masks
from evi.evaluation.protocol import (
    METHOD_TYPE,
    TAU_ALPHA,
    evaluate_gobj_protocol,
    protocol_config,
)
from evi.evidence.reference_geometry import FrozenG0, build_reference_from_gaussians
from evi.evidence.visibility_evidence import build_spatial_evidence
from evi.export.diagnostics import label_range_summary, save_diagnostics
from evi.export.object_ply import hard_extract_mask, save_object_ply
from evi.render.label_renderer import evaluate_mask_metrics
from evi.candidates import build_object_candidate_mask
from evi.config import EviConfig
from evi.learn_export_tau import run_learn_export_tau
from evi.optimize import set_trainable
from evi.post_tau_admission import (
    evaluate_hard_export,
    run_post_tau_admission,
    save_candidate_diagnostics,
    select_diverse_mask_cameras,
    split_interleaved_mask_cameras,
)
from evi.support_stats import collect_binary_support, init_label_from_support
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration

METHOD_NAME = "EVI-GS"
DEFAULT_SUPPORT_TAU = 0.5


def describe_stages(run_learn: bool, run_post: bool) -> str:
    if run_learn and run_post:
        return "support_init → learn_tau → post_tau_admission"
    if run_learn:
        return "support_init → learn_tau"
    return "support_init"


def resolve_enabled_stages(cfg: EviConfig):
    """Only 1, 1+2, and 1+2+3 are valid. post_tau requires learn_tau."""
    run_learn = bool(cfg.enable_learn_export_tau)
    run_post = bool(cfg.enable_post_tau_admission)
    if run_post and not run_learn:
        print("[EVI] post_tau_admission requires learn_tau; disabling post_tau")
        cfg.enable_post_tau_admission = False
        run_post = False
    return run_learn, run_post


def prepare_output(args):
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:10])
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def resolve_warmup_ply(warmup_from: str):
    src_root = os.path.join(warmup_from, "point_cloud")
    it = searchForMaxIteration(src_root)
    ply = os.path.join(src_root, f"iteration_{it}", "point_cloud.ply")
    if not os.path.isfile(ply):
        raise FileNotFoundError(ply)
    return ply, it


def copy_warmup_sidecars(warmup_from: str, model_path: str) -> None:
    os.makedirs(model_path, exist_ok=True)
    for name in ("cameras.json", "input.ply", "exposure.json"):
        p = os.path.join(warmup_from, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(model_path, name))


def split_fit_calib(masked, calib_frac: float):
    n = len(masked)
    if n <= 1:
        return list(masked), list(masked)
    n_calib = max(1, int(round(n * calib_frac)))
    n_calib = min(n_calib, n - 1)
    ordered = sorted(masked, key=lambda c: c.image_name)
    calib = ordered[-n_calib:]
    calib_names = {c.image_name for c in calib}
    fit = [c for c in ordered if c.image_name not in calib_names]
    return fit, calib


def save_stage_debug(
    stage: str,
    model_path: str,
    gaussians,
    cfg: EviConfig,
    extra: Optional[dict] = None,
    tau: Optional[float] = None,
) -> dict:
    """Checkpoint one training stage for inspection."""
    dbg_dir = os.path.join(model_path, "debug", stage)
    os.makedirs(dbg_dir, exist_ok=True)
    tau = float(DEFAULT_SUPPORT_TAU if tau is None else tau)
    info = {
        "stage": stage,
        "tau": tau,
        "tau_z": float(cfg.tau_z),
        "num_gaussians_all": int(gaussians.get_xyz.shape[0]),
        "admission": None,
    }
    try:
        gaussians.save_ply(os.path.join(dbg_dir, "full_gaussians.ply"))
    except Exception as exc:
        info["full_ply_error"] = str(exc)
    try:
        n_obj = save_object_ply(
            gaussians,
            os.path.join(dbg_dir, "object_init_tau.ply"),
            tau=tau,
            eta=None,
            retain=None,
        )
        info["N_obj"] = int(n_obj)
    except Exception as exc:
        info["N_obj"] = 0
        info["object_ply_error"] = str(exc)
        try:
            n_obj = save_object_ply(
                gaussians,
                os.path.join(dbg_dir, "object_init_tau_zonly.ply"),
                tau=tau,
                eta=None,
                retain=None,
            )
            info["N_obj_zonly"] = int(n_obj)
        except Exception as exc2:
            info["object_zonly_error"] = str(exc2)
    if extra:
        info.update(extra)
    save_diagnostics(os.path.join(dbg_dir, "stage_meta.json"), info)
    print(f"[EVI][debug] stage={stage} N_obj={info.get('N_obj', info.get('N_obj_zonly', 0))} tau={tau}")
    return info


def run_support_init(gaussians, fit_cams, train_cams, pipe, background, cfg, args, g0):
    ref = build_reference_from_gaussians(
        train_cams,
        gaussians,
        pipe,
        background,
        max_views=min(32, len(train_cams)),
        a_min=cfg.a_min,
        g0=g0,
    )
    spatial_ev = build_spatial_evidence(train_cams, ref, depth_tol=cfg.depth_tol, a_min=cfg.a_min)
    q_by_name = {n: ref.reliable_by_name[n] for n in ref.reliable_by_name}

    n_fit_mask = sum(1 for c in fit_cams if getattr(c, "has_object_mask", False))
    n_support = n_fit_mask if int(cfg.support_max_views) <= 0 else min(n_fit_mask, int(cfg.support_max_views))
    print(
        f"[EVI] support_init: collecting binary support "
        f"views={n_support}/{n_fit_mask} max={int(cfg.support_max_views)}"
    )
    set_trainable(gaussians, False, False, False)
    s_pos, s_neg = collect_binary_support(
        fit_cams,
        gaussians,
        pipe,
        background,
        q_by_name=q_by_name,
        max_views=int(cfg.support_max_views),
    )
    init_label_from_support(gaussians, s_pos, s_neg)
    cand_m = build_object_candidate_mask(
        gaussians,
        fit_cams,
        ref,
        s_pos=s_pos,
        s_neg=s_neg,
        spatial_ev=spatial_ev,
        a_min=cfg.a_min,
        depth_tol=cfg.depth_tol,
        front_support_tau=cfg.front_support_tau,
        radius_mult=cfg.admit_radius_mult,
    )
    n_cand = int(cand_m.sum())
    print(f"[EVI] support_init: candidate eligible {n_cand}/{int(cand_m.numel())}")
    save_stage_debug(
        "00_support_init",
        args.model_path,
        gaussians,
        cfg,
        extra={
            "candidate_eligible": n_cand,
            "note": "after support-init labels; export tau=0.5 if learn_tau is off",
        },
        tau=float(cfg.export_tau_z),
    )


def run_learn_tau_stage(gaussians, fit_cams, pipe, background, cfg, args, progress, tick):
    learn_tau_steps = int(cfg.learn_export_tau_mult) * max(len(fit_cams), cfg.min_P)
    print(
        f"[EVI] learn_tau: steps={learn_tau_steps}(=1P) init={cfg.learn_export_tau_init} "
        f"lr={cfg.learn_export_tau_lr} temp={cfg.learn_export_tau_temp} "
        f"views/step={cfg.learn_export_tau_views} min_views={cfg.learn_export_tau_min_views} "
        f"fit_cams={len(fit_cams)}"
    )
    learn_tau_info = run_learn_export_tau(
        gaussians,
        list(fit_cams),
        pipe,
        background,
        None,
        init_tau=float(cfg.learn_export_tau_init),
        n_steps=int(learn_tau_steps),
        lr=float(cfg.learn_export_tau_lr),
        temp=float(cfg.learn_export_tau_temp),
        min_views=int(cfg.learn_export_tau_min_views),
        views_per_step=int(cfg.learn_export_tau_views),
        progress=progress,
        tick=tick,
    )
    export_tau = float(learn_tau_info.get("export_tau", DEFAULT_SUPPORT_TAU))
    save_diagnostics(os.path.join(args.model_path, "learn_export_tau.json"), learn_tau_info)
    print(
        f"[EVI] learned export_tau={export_tau:.6f} "
        f"steps={learn_tau_info.get('n_steps')} "
        f"views/step={learn_tau_info.get('views_per_step')} "
        f"best_soft={learn_tau_info.get('best_soft_miou')}"
    )
    save_stage_debug(
        "04_after_learn_tau",
        args.model_path,
        gaussians,
        cfg,
        extra={"learned_export_tau": export_tau, "learn_export_tau": learn_tau_info},
        tau=export_tau,
    )
    return export_tau, learn_tau_info, learn_tau_steps


def run_post_tau_stage(
    gaussians,
    fit_cams,
    calib_cams,
    masked_train,
    pipe,
    background,
    g0,
    cfg,
    args,
    export_tau,
):
    """Read-only cleanup. Raw z-cut is always written; proposal accepted only if calib holds."""
    raw_hard_mask = hard_extract_mask(gaussians, export_tau, eta=None, retain=None)
    n_obj_raw = int(raw_hard_mask.sum().item())
    export_eta = None
    post_tau_result = None
    post_tau_accepted = False

    try:
        save_object_ply(
            gaussians,
            os.path.join(args.model_path, "object_raw.ply"),
            tau=export_tau,
            eta=None,
            retain=None,
        )

        post_evidence_pool = fit_cams
        post_calib_cams = calib_cams
        if cfg.post_tau_interleaved_split:
            post_evidence_pool, post_calib_cams = split_interleaved_mask_cameras(
                masked_train,
                holdout_frac=float(cfg.post_tau_calib_frac),
            )
        post_cams = select_diverse_mask_cameras(
            post_evidence_pool,
            max_views=int(cfg.post_tau_admission_views),
        )
        print(
            f"[EVI] post_tau_admission: candidates={n_obj_raw} "
            f"views={len(post_cams)} calib={len(post_calib_cams)} "
            f"split={'interleaved' if cfg.post_tau_interleaved_split else 'training'} "
            f"tau={export_tau:.6f}"
        )
        post_ref = build_reference_from_gaussians(
            post_cams,
            gaussians,
            pipe,
            background,
            max_views=None,
            a_min=cfg.a_min,
            g0=g0,
        )
        post_tau_result = run_post_tau_admission(
            gaussians,
            post_cams,
            post_ref,
            pipe,
            background,
            export_tau=export_tau,
            base_eta=None,
            a_min=cfg.a_min,
            depth_tol=cfg.depth_tol,
            min_view_support=cfg.post_tau_min_view_support,
            support_abs_min=cfg.post_tau_support_abs_min,
            support_rel_median=cfg.post_tau_support_rel_median,
            min_seed_views=cfg.post_tau_min_seed_views,
            min_surface_views=cfg.post_tau_min_surface_views,
            min_purity=cfg.post_tau_min_purity,
            radius_mult=cfg.post_tau_radius_mult,
            max_hops=cfg.post_tau_max_hops,
            weak_rescue=cfg.post_tau_weak_rescue,
            weak_support_rel=cfg.post_tau_weak_support_rel,
            weak_rgb_views=cfg.post_tau_weak_rgb_views,
            weak_radius_mult=cfg.post_tau_weak_radius_mult,
            low_z_bridge=cfg.post_tau_low_z_bridge,
            bridge_delta=cfg.post_tau_bridge_delta,
            ellipsoid_hard_negative=cfg.post_tau_ellipsoid_hard_negative,
            probe_sigma=cfg.post_tau_probe_sigma,
            probe_negative_fraction=cfg.post_tau_probe_negative_fraction,
        )
        proposal_hard_mask = hard_extract_mask(
            gaussians,
            export_tau,
            eta=post_tau_result.eta,
            retain=None,
        )
        save_candidate_diagnostics(
            os.path.join(args.model_path, "post_tau_candidates.npz"),
            post_tau_result,
            gaussians,
        )
        if proposal_hard_mask.any():
            save_object_ply(
                gaussians,
                os.path.join(args.model_path, "object_post_tau_proposal.ply"),
                tau=export_tau,
                mask=proposal_hard_mask,
            )

        post_tau_raw_calib = evaluate_hard_export(
            post_calib_cams,
            gaussians,
            pipe,
            raw_hard_mask,
            tau_alpha=TAU_ALPHA,
        )
        post_tau_clean_calib = evaluate_hard_export(
            post_calib_cams,
            gaussians,
            pipe,
            proposal_hard_mask,
            tau_alpha=TAU_ALPHA,
        )

        reject_reasons = []
        if not post_tau_result.stats.get("proposal_applied", False):
            reject_reasons.append(
                str(post_tau_result.stats.get("fallback_reason") or "proposal_not_applied")
            )
        keep_fraction = float(post_tau_result.stats.get("keep_fraction", 0.0))
        if keep_fraction < float(cfg.post_tau_min_keep_fraction):
            reject_reasons.append(
                f"keep_fraction={keep_fraction:.4f}"
                f"<{float(cfg.post_tau_min_keep_fraction):.4f}"
            )
        if int(proposal_hard_mask.sum().item()) == 0:
            reject_reasons.append("empty_proposal")
        if post_tau_raw_calib["num_fg_views"] > 0:
            iou_drop = float(post_tau_raw_calib["mIoU"]) - float(post_tau_clean_calib["mIoU"])
            recall_drop = (
                float(post_tau_raw_calib["mean_recall"])
                - float(post_tau_clean_calib["mean_recall"])
            )
            if iou_drop > float(cfg.post_tau_max_calib_iou_drop):
                reject_reasons.append(
                    f"calib_iou_drop={iou_drop:.6f}"
                    f">{float(cfg.post_tau_max_calib_iou_drop):.6f}"
                )
            if recall_drop > float(cfg.post_tau_max_calib_recall_drop):
                reject_reasons.append(
                    f"calib_recall_drop={recall_drop:.6f}"
                    f">{float(cfg.post_tau_max_calib_recall_drop):.6f}"
                )

        post_tau_accepted = len(reject_reasons) == 0
        post_tau_result.stats.update(
            {
                "accepted": bool(post_tau_accepted),
                "reject_reasons": reject_reasons,
                "raw_calib": post_tau_raw_calib,
                "proposal_calib": post_tau_clean_calib,
                "view_split": (
                    "interleaved" if cfg.post_tau_interleaved_split else "training_fit_and_calib"
                ),
                "num_evidence_pool_views": len(post_evidence_pool),
                "num_evidence_views": len(post_cams),
                "num_calib_views": len(post_calib_cams),
                "evidence_view_names": [c.image_name for c in post_cams],
                "calib_view_names": [c.image_name for c in post_calib_cams],
                "min_keep_fraction": float(cfg.post_tau_min_keep_fraction),
                "max_calib_iou_drop": float(cfg.post_tau_max_calib_iou_drop),
                "max_calib_recall_drop": float(cfg.post_tau_max_calib_recall_drop),
            }
        )
        if post_tau_accepted:
            export_eta = post_tau_result.eta
        print(
            f"[EVI] post_tau_admission "
            f"{'ACCEPTED' if post_tau_accepted else 'REJECTED'}: "
            f"raw={n_obj_raw} "
            f"proposal={int(proposal_hard_mask.sum())} "
            f"seeds={post_tau_result.stats.get('num_seeds', 0)} "
            f"reasons={reject_reasons}"
        )
        save_diagnostics(
            os.path.join(args.model_path, "post_tau_admission.json"),
            post_tau_result.stats,
        )
        del post_ref
        torch.cuda.empty_cache()
    except Exception as exc:
        export_eta = None
        post_tau_accepted = False
        save_diagnostics(
            os.path.join(args.model_path, "post_tau_admission.json"),
            {
                "enabled": True,
                "accepted": False,
                "fallback_reason": "exception",
                "error": str(exc),
                "num_candidates": n_obj_raw,
            },
        )
        print(f"[EVI] post_tau_admission failed open: {exc}")

    return export_eta, post_tau_result, post_tau_accepted, n_obj_raw


def export_and_evaluate(
    dataset,
    gaussians,
    fit_cams,
    test_cams,
    pipe,
    background,
    cfg,
    args,
    split,
    object_id,
    export_tau,
    export_eta,
    learn_tau_info,
    post_tau_result,
    post_tau_accepted,
    n_obj_raw,
    P,
    learn_tau_steps,
    global_it,
    progress,
    run_learn: bool,
    run_post: bool,
):
    try:
        n_obj = save_object_ply(
            gaussians,
            os.path.join(args.model_path, "object.ply"),
            tau=export_tau,
            eta=export_eta,
            retain=None,
        )
    except Exception as exc:
        save_diagnostics(
            os.path.join(args.model_path, "metrics.json"),
            {"object_id": int(object_id), "num_eval_views": 0, "error": str(exc), "N_obj": 0},
        )
        progress.close()
        return

    obj_ply = os.path.join(args.model_path, "object.ply")
    gobj_ply = os.path.join(args.model_path, "point_cloud_obj.ply")
    shutil.copy2(obj_ply, gobj_ply)

    p1 = evaluate_mask_metrics(fit_cams, gaussians, pipe, background)
    stages = describe_stages(run_learn, run_post)
    if run_learn and run_post and post_tau_accepted:
        export_rule = "support_init + z >= learned_export_tau AND post_tau_eta"
    elif run_learn and run_post:
        export_rule = "support_init + z >= learned_export_tau (post_tau rejected; no eta)"
    elif run_learn:
        export_rule = "support_init + z >= learned_export_tau"
    else:
        export_rule = f"support_init + z >= {float(export_tau):g}"

    object_meta = {
        "scene_id": os.path.basename(os.path.abspath(args.source_path)),
        "object_id": int(object_id),
        "method": METHOD_NAME,
        "type": METHOD_TYPE,
        "mask_source": "dataset_gt",
        "N_obj": int(n_obj),
        "N_obj_raw": int(n_obj_raw),
        "tau_z": float(cfg.tau_z),
        "export_tau_z": float(export_tau),
        "tau_alpha": float(TAU_ALPHA),
        "export_rule": export_rule,
        "stages": stages,
        "warmup_from": str(args.warmup_from or ""),
        "split_hash": split.get("hash"),
        "protocol": protocol_config(),
        "admission": None,
        "post_tau_admission": post_tau_result.stats if post_tau_result else {
            "enabled": bool(run_post),
            "accepted": False,
        },
        "learn_export_tau": learn_tau_info,
        "P1": p1,
        "hard_audit": None,
        "labels": label_range_summary(gaussians),
        "schedule": {
            "P": P,
            "global_it": global_it,
            "learn_tau_steps": learn_tau_steps,
            "stages": stages,
            "enable_learn_export_tau": bool(run_learn),
            "enable_post_tau_admission": bool(run_post),
            "post_tau_admission_accepted": bool(post_tau_accepted),
        },
        "evi_config": asdict(cfg),
    }
    save_diagnostics(os.path.join(args.model_path, "object_meta.json"), object_meta)
    save_diagnostics(os.path.join(args.model_path, "final_eval.json"), object_meta)

    proto = None
    try:
        g_obj = GaussianModel(dataset.sh_degree)
        g_obj.load_ply(gobj_ply)
        proto = evaluate_gobj_protocol(
            test_cams, g_obj, pipe, tau_alpha=TAU_ALPHA, object_id=int(object_id), n_obj=int(n_obj)
        )
        proto["method"] = METHOD_NAME
        proto["tau_z"] = float(cfg.tau_z)
        proto["export_tau_z"] = float(export_tau)
        proto["export_rule"] = export_rule
        proto["N_obj_raw"] = int(n_obj_raw)
        proto["post_tau_admission_enabled"] = bool(run_post)
        proto["post_tau_admission_accepted"] = bool(post_tau_accepted)
        if learn_tau_info is not None:
            proto["learned_export_tau"] = float(learn_tau_info.get("export_tau", export_tau))
        save_diagnostics(os.path.join(args.model_path, "metrics.json"), proto)
        save_diagnostics(os.path.join(args.model_path, "per_view.json"), proto.get("per_view", []))
        del g_obj
        torch.cuda.empty_cache()
    except Exception as exc:
        proto = {"object_id": int(object_id), "num_eval_views": 0, "error": str(exc), "N_obj": int(n_obj)}
        save_diagnostics(os.path.join(args.model_path, "metrics.json"), proto)
        print(f"[EVI] protocol eval failed: {exc}")

    if os.path.isfile(os.path.join(args.model_path, "object_raw.ply")):
        try:
            g_raw = GaussianModel(dataset.sh_degree)
            g_raw.load_ply(os.path.join(args.model_path, "object_raw.ply"))
            proto_raw = evaluate_gobj_protocol(
                test_cams,
                g_raw,
                pipe,
                tau_alpha=TAU_ALPHA,
                object_id=int(object_id),
                n_obj=int(n_obj_raw),
            )
            proto_raw["method"] = f"{METHOD_NAME}-raw"
            proto_raw["tau_z"] = float(cfg.tau_z)
            proto_raw["export_tau_z"] = float(export_tau)
            proto_raw["export_rule"] = "z-cut before post-tau admission"
            save_diagnostics(os.path.join(args.model_path, "metrics_raw.json"), proto_raw)
            del g_raw
            torch.cuda.empty_cache()
        except Exception as exc:
            save_diagnostics(
                os.path.join(args.model_path, "metrics_raw.json"),
                {
                    "object_id": int(object_id),
                    "num_eval_views": 0,
                    "error": str(exc),
                    "N_obj": int(n_obj_raw),
                },
            )

    progress.close()
    print(
        "[EVI] done:",
        json.dumps(
            {
                "object_id": object_id,
                "N_obj": n_obj,
                "export_tau": export_tau,
                "stages": stages,
                "post_tau_accepted": bool(post_tau_accepted),
                "protocol": {k: (proto or {}).get(k) for k in ("mIoU", "mAcc", "mBIoU", "SSIM", "LPIPS")},
            },
            indent=2,
        ),
    )


def training_evi(dataset, opt, pipe, args, cfg: EviConfig):
    run_learn, run_post = resolve_enabled_stages(cfg)
    stages = describe_stages(run_learn, run_post)
    prepare_output(args)
    save_diagnostics(os.path.join(args.model_path, "evi_config.json"), asdict(cfg))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    warmup_ply = None
    if args.warmup_from:
        warmup_ply, warmup_it = resolve_warmup_ply(args.warmup_from)
        copy_warmup_sidecars(args.warmup_from, args.model_path)
        print(f"[EVI] warmup ply={warmup_ply} iter={warmup_it}")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    dataset.eval = True
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False, load_ply_path=warmup_ply)
    n0 = int(gaussians.get_xyz.shape[0])
    print(f"[EVI] loaded gaussians n={n0}")
    if args.warmup_from and n0 < 50_000:
        raise RuntimeError(f"Warmup load looks wrong: only {n0} gaussians")

    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    split_path = str(getattr(args, "split_json", "") or "").strip()
    if split_path and os.path.isfile(split_path):
        split = load_split(split_path)
    else:
        split = build_stride_split([c.image_name for c in all_cams], stride=8)
    save_split(split, os.path.join(args.model_path, "split.json"))
    guard = SplitGuard(split)
    test_names = set(split["test_ids"])
    train_cams = [c for c in all_cams if c.image_name not in test_names]
    test_cams = [c for c in all_cams if c.image_name in test_names]
    for cam in train_cams:
        guard.assert_train_name(cam.image_name)

    masks_folder = os.path.join(args.source_path, args.masks) if args.masks else ""
    object_id = args.object_id
    if object_id < 0:
        object_id = auto_select_object_id(masks_folder, [c.image_name for c in train_cams])
    attach_object_masks(train_cams, masks_folder, object_id)
    attach_object_masks(test_cams, masks_folder, object_id)

    masked_train = [c for c in train_cams if getattr(c, "has_object_mask", False)]
    fit_cams, calib_cams = split_fit_calib(masked_train, cfg.calib_frac)
    if not calib_cams:
        calib_cams = list(fit_cams)
    save_diagnostics(
        os.path.join(args.model_path, "split_fit_calib.json"),
        {
            "fit": [c.image_name for c in fit_cams],
            "calib": [c.image_name for c in calib_cams],
            "test": [c.image_name for c in test_cams],
        },
    )
    n_fg = sum(int((c.object_mask > 0.5).any().item()) for c in fit_cams)
    if n_fg == 0:
        save_diagnostics(
            os.path.join(args.model_path, "metrics.json"),
            {"object_id": int(object_id), "num_eval_views": 0, "skip_reason": "no_train_foreground", "N_obj": 0},
        )
        return

    gaussians.training_setup(opt)
    g0 = FrozenG0.from_gaussians(gaussians)
    P = max(len(fit_cams), cfg.min_P)
    learn_tau_steps = int(cfg.learn_export_tau_mult) * P if run_learn else 0
    print(
        f"[EVI] stages: {stages} "
        f"learn_tau={'on' if run_learn else 'off'} "
        f"post_tau={'on' if run_post else 'off'} "
        f"P={P} learn_tau_steps={learn_tau_steps}"
    )
    progress = tqdm(total=max(1, learn_tau_steps), desc="EVI")
    global_it = 0

    def tick(n=1):
        nonlocal global_it
        global_it += n
        progress.update(n)

    run_support_init(gaussians, fit_cams, train_cams, pipe, background, cfg, args, g0)
    scene.save(global_it)

    export_tau = float(cfg.export_tau_z)
    learn_tau_info = None
    if run_learn:
        export_tau, learn_tau_info, learn_tau_steps = run_learn_tau_stage(
            gaussians, fit_cams, pipe, background, cfg, args, progress, tick
        )
    else:
        print(f"[EVI] skipped learn_tau; export_tau={export_tau:g}")

    export_eta = None
    post_tau_result = None
    post_tau_accepted = False
    n_obj_raw = int(hard_extract_mask(gaussians, export_tau, eta=None, retain=None).sum().item())
    if run_post:
        export_eta, post_tau_result, post_tau_accepted, n_obj_raw = run_post_tau_stage(
            gaussians,
            fit_cams,
            calib_cams,
            masked_train,
            pipe,
            background,
            g0,
            cfg,
            args,
            export_tau,
        )
    else:
        print(f"[EVI] skipped post_tau_admission; z-cut only at tau={export_tau:.6f}")

    export_and_evaluate(
        dataset,
        gaussians,
        fit_cams,
        test_cams,
        pipe,
        background,
        cfg,
        args,
        split,
        object_id,
        export_tau,
        export_eta,
        learn_tau_info,
        post_tau_result,
        post_tau_accepted,
        n_obj_raw,
        P,
        learn_tau_steps,
        global_it,
        progress,
        run_learn,
        run_post,
    )


def _apply_optional(cfg, args, name, caster):
    val = getattr(args, name, None)
    if val is not None:
        setattr(cfg, name, caster(val))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="EVI-GS: support_init, optional learn_tau, optional post_tau_admission"
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--learn_export_tau",
        action="store_true",
        default=False,
        help="Enable learn_tau (default on). Pair with --no_learn_export_tau to disable.",
    )
    parser.add_argument(
        "--no_learn_export_tau",
        action="store_true",
        default=False,
        help="Disable learn_tau; export uses z >= 0.5 (support_init only).",
    )
    parser.add_argument(
        "--export_tau_z",
        type=float,
        default=None,
        help="Export tau when learn_tau is off (default 0.5). Ignored if learn_tau runs.",
    )
    parser.add_argument("--learn_export_tau_init", type=float, default=None)
    parser.add_argument(
        "--calib_frac",
        type=float,
        default=None,
        help="Tail fraction of masked train views held out as calib (default 0.2).",
    )
    parser.add_argument(
        "--support_max_views",
        type=int,
        default=None,
        help="Fit views for support_init S+/S- (default 8). <=0 uses all fit masked views.",
    )
    parser.add_argument("--learn_export_tau_mult", type=int, default=None)
    parser.add_argument("--learn_export_tau_lr", type=float, default=None)
    parser.add_argument("--learn_export_tau_temp", type=float, default=None)
    parser.add_argument("--learn_export_tau_views", type=int, default=None)
    parser.add_argument("--learn_export_tau_views_div", type=int, default=None)
    parser.add_argument("--learn_export_tau_min_views", type=int, default=None)
    parser.add_argument("--learn_export_tau_patience", type=int, default=None)
    parser.add_argument("--learn_export_tau_max_steps", type=int, default=None)
    parser.add_argument(
        "--post_tau_admission",
        action="store_true",
        default=False,
        help="Enable post_tau_admission (default on; requires learn_tau).",
    )
    parser.add_argument(
        "--no_post_tau_admission",
        action="store_true",
        default=False,
        help="Disable post_tau_admission; export uses learned_tau only (or 0.5 if learn_tau is off).",
    )
    parser.add_argument("--post_tau_admission_views", type=int, default=None)
    parser.add_argument(
        "--post_tau_interleaved_split",
        action="store_true",
        default=False,
        help="Use a sequence-wide interleaved evidence/calibration split for post-tau cleanup",
    )
    parser.add_argument("--post_tau_calib_frac", type=float, default=None)
    parser.add_argument("--post_tau_min_view_support", type=float, default=None)
    parser.add_argument("--post_tau_support_abs_min", type=float, default=None)
    parser.add_argument("--post_tau_support_rel_median", type=float, default=None)
    parser.add_argument("--post_tau_min_seed_views", type=int, default=None)
    parser.add_argument("--post_tau_min_surface_views", type=int, default=None)
    parser.add_argument("--post_tau_min_purity", type=float, default=None)
    parser.add_argument("--post_tau_radius_mult", type=float, default=None)
    parser.add_argument("--post_tau_max_hops", type=int, default=None)
    parser.add_argument(
        "--post_tau_weak_rescue",
        action="store_true",
        default=False,
        help="Attach weak one-view/RGB candidates to the admitted core for one local hop",
    )
    parser.add_argument("--post_tau_weak_support_rel", type=float, default=None)
    parser.add_argument("--post_tau_weak_rgb_views", type=int, default=None)
    parser.add_argument("--post_tau_weak_radius_mult", type=float, default=None)
    parser.add_argument(
        "--post_tau_low_z_bridge",
        action="store_true",
        default=False,
        help="Allow near-surface z >= tau-delta nodes as non-exported connectivity bridges",
    )
    parser.add_argument("--post_tau_bridge_delta", type=float, default=None)
    parser.add_argument(
        "--post_tau_ellipsoid_hard_negative",
        action="store_true",
        default=False,
        help="Use centre and principal-axis probes for conservative hard-negative geometry",
    )
    parser.add_argument("--post_tau_probe_sigma", type=float, default=None)
    parser.add_argument("--post_tau_probe_negative_fraction", type=float, default=None)
    parser.add_argument("--post_tau_min_keep_fraction", type=float, default=None)
    parser.add_argument("--post_tau_max_calib_iou_drop", type=float, default=None)
    parser.add_argument("--post_tau_max_calib_recall_drop", type=float, default=None)
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    cfg = EviConfig()
    if args.no_learn_export_tau:
        cfg.enable_learn_export_tau = False
    elif args.learn_export_tau:
        cfg.enable_learn_export_tau = True
    if args.no_post_tau_admission:
        cfg.enable_post_tau_admission = False
    elif args.post_tau_admission:
        cfg.enable_post_tau_admission = True
    if args.export_tau_z is not None:
        cfg.export_tau_z = float(args.export_tau_z)
    for name in (
        "learn_export_tau_init",
        "learn_export_tau_lr",
        "learn_export_tau_temp",
        "calib_frac",
        "post_tau_calib_frac",
        "post_tau_min_view_support",
        "post_tau_support_abs_min",
        "post_tau_support_rel_median",
        "post_tau_min_purity",
        "post_tau_radius_mult",
        "post_tau_weak_support_rel",
        "post_tau_weak_radius_mult",
        "post_tau_bridge_delta",
        "post_tau_probe_sigma",
        "post_tau_probe_negative_fraction",
        "post_tau_min_keep_fraction",
        "post_tau_max_calib_iou_drop",
        "post_tau_max_calib_recall_drop",
    ):
        _apply_optional(cfg, args, name, float)
    for name in (
        "learn_export_tau_mult",
        "learn_export_tau_views",
        "support_max_views",
        "learn_export_tau_views_div",
        "learn_export_tau_min_views",
        "learn_export_tau_patience",
        "learn_export_tau_max_steps",
        "post_tau_admission_views",
        "post_tau_min_seed_views",
        "post_tau_min_surface_views",
        "post_tau_max_hops",
        "post_tau_weak_rgb_views",
    ):
        _apply_optional(cfg, args, name, int)
    if args.post_tau_interleaved_split:
        cfg.post_tau_interleaved_split = True
    if args.post_tau_weak_rescue:
        cfg.post_tau_weak_rescue = True
    if args.post_tau_low_z_bridge:
        cfg.post_tau_low_z_bridge = True
    if args.post_tau_ellipsoid_hard_negative:
        cfg.post_tau_ellipsoid_hard_negative = True
    args.source_path = dataset.source_path
    args.masks = dataset.masks
    args.object_id = dataset.object_id
    args.warmup_from = dataset.warmup_from
    args.model_path = dataset.model_path
    args.split_json = dataset.split_json
    training_evi(dataset, opt, pipe, args, cfg)
