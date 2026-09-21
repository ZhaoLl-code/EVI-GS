"""Read-only post-tau admission for unsupported object floaters.

This module deliberately does not change Gaussian labels, geometry, opacity, or
appearance.  It builds an export-only gate

    h_i = 1[z_i >= tau] * eta_i

from absolute rendered support, frozen-G0 surface evidence, and bounded
seed-to-neighbour connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evi.evidence.visibility_evidence import project_points_to_cam
from gaussian_renderer import render
from utils.general_utils import build_rotation


@dataclass
class PostTauAdmissionResult:
    """Post-tau export gate plus candidate-level diagnostics."""

    eta: torch.Tensor
    candidate_ids: torch.Tensor
    seed_local: torch.Tensor
    hard_negative_local: torch.Tensor
    admitted_local: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]
    stats: Dict


def _pick_evenly(cameras: Sequence, count: int) -> List:
    cams = sorted(cameras, key=lambda c: c.image_name)
    if count <= 0 or len(cams) <= count:
        return list(cams)
    if count == 1:
        return [cams[len(cams) // 2]]
    idx = np.linspace(0, len(cams) - 1, num=count)
    chosen = sorted({int(round(float(i))) for i in idx})
    if len(chosen) < count:
        chosen_set = set(chosen)
        for i in range(len(cams)):
            if i not in chosen_set:
                chosen.append(i)
                chosen_set.add(i)
            if len(chosen) >= count:
                break
        chosen.sort()
    return [cams[i] for i in chosen[:count]]


def select_diverse_mask_cameras(cameras: Sequence, max_views: int = 32) -> List:
    """Deterministically cover the sequence and retain a few empty-GT views."""

    masked = [c for c in cameras if getattr(c, "has_object_mask", False)]
    if int(max_views) <= 0 or len(masked) <= int(max_views):
        return sorted(masked, key=lambda c: c.image_name)

    foreground = [c for c in masked if bool((c.object_mask > 0.5).any().item())]
    empty = [c for c in masked if not bool((c.object_mask > 0.5).any().item())]
    empty_budget = min(len(empty), max(1, int(max_views) // 8))
    foreground_budget = max(1, int(max_views) - empty_budget)
    selected = _pick_evenly(foreground, foreground_budget)
    selected.extend(_pick_evenly(empty, empty_budget))

    # Fill any unused budget (for example, when there are few empty views).
    selected_names = {c.image_name for c in selected}
    remaining = [c for c in masked if c.image_name not in selected_names]
    selected.extend(_pick_evenly(remaining, int(max_views) - len(selected)))
    return sorted(selected[: int(max_views)], key=lambda c: c.image_name)


def split_interleaved_mask_cameras(
    cameras: Sequence,
    holdout_frac: float = 0.2,
) -> Tuple[List, List]:
    """Create sequence-wide evidence/holdout sets instead of a tail holdout."""

    masked = sorted(
        [c for c in cameras if getattr(c, "has_object_mask", False)],
        key=lambda c: c.image_name,
    )
    if len(masked) <= 1:
        return list(masked), list(masked)
    holdout_count = max(1, int(round(len(masked) * float(holdout_frac))))
    holdout_count = min(holdout_count, len(masked) - 1)
    holdout = _pick_evenly(masked, holdout_count)
    holdout_names = {c.image_name for c in holdout}
    evidence = [c for c in masked if c.image_name not in holdout_names]
    return evidence, holdout


def _sample_map(
    map_1hw: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    x = map_1hw.to(device=u.device, dtype=torch.float32)
    grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1).view(1, -1, 1, 2)
    out = F.grid_sample(
        x[None],
        grid,
        align_corners=True,
        mode=mode,
        padding_mode="zeros",
    )
    return out.view(u.shape)


def _gaussian_ellipsoid_probes(
    gaussians,
    ids: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Return centre and +/- principal-axis sigma probes as [N, 7, 3]."""

    centres = gaussians.get_xyz[ids].detach()
    rotations = build_rotation(gaussians.get_rotation[ids].detach())
    scales = gaussians.get_scaling[ids].detach()
    axis_vectors = (
        rotations * scales[:, None, :] * float(sigma)
    ).transpose(1, 2)
    return torch.cat(
        [
            centres[:, None, :],
            centres[:, None, :] + axis_vectors,
            centres[:, None, :] - axis_vectors,
        ],
        dim=1,
    )


def _collect_candidate_evidence(
    gaussians,
    cameras: Sequence,
    ref,
    pipe,
    background: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    a_min: float,
    depth_tol: float,
    min_view_support: float,
    ellipsoid_probes: bool = False,
    probe_sigma: float = 1.0,
    probe_negative_fraction: float = 0.7,
) -> Dict[str, torch.Tensor]:
    """Collect absolute alpha contribution and frozen-surface evidence."""

    device = gaussians.get_xyz.device
    ids = candidate_ids.long()
    count = int(ids.numel())
    out = {
        "s_pos": torch.zeros(count, device=device),
        "s_neg": torch.zeros(count, device=device),
        "pos_views": torch.zeros(count, device=device),
        "rgb_views": torch.zeros(count, device=device),
        "fg_surface_views": torch.zeros(count, device=device),
        "bg_surface_views": torch.zeros(count, device=device),
        "free_views": torch.zeros(count, device=device),
        "joint_fg_views": torch.zeros(count, device=device),
        "foreground_probe_hits": torch.zeros(count, device=device),
        "negative_probe_fraction_sum": torch.zeros(count, device=device),
    }
    if count == 0:
        return out

    xyz = gaussians.get_xyz[ids].detach()
    if bool(ellipsoid_probes):
        probe_points = _gaussian_ellipsoid_probes(
            gaussians,
            ids,
            sigma=float(probe_sigma),
        )
    else:
        probe_points = xyz[:, None, :]
    opacity = gaussians.get_opacity.detach()
    bg0 = torch.zeros_like(background)
    n_all = int(gaussians.get_xyz.shape[0])

    for cam in cameras:
        name = cam.image_name
        if name not in ref.depth_by_name or name not in ref.alpha_by_name:
            continue

        # One auxiliary-colour render/backward gives absolute contribution for
        # every candidate while preserving full-scene transmittance.
        with torch.enable_grad():
            aux = torch.zeros(
                n_all,
                3,
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            pkg = render(
                cam,
                gaussians,
                pipe,
                bg0,
                override_color=aux,
                override_opacity=opacity,
            )
            image = pkg["render"]
            gt_mask = cam.object_mask.to(device=device, dtype=image.dtype)
            valid = cam.mask_valid.to(device=device, dtype=image.dtype)
            covered = (
                ref.alpha_by_name[name].to(device=device, dtype=image.dtype) >= float(a_min)
            ).to(image.dtype)
            q = valid * covered
            phi = (q * (gt_mask * image[:1] + (1.0 - gt_mask) * image[1:2])).sum()
            grad = torch.autograd.grad(
                phi,
                aux,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

        if grad is None:
            del aux, pkg, image, phi
            continue

        g_pos = grad[ids, 0].detach().clamp_min(0)
        g_neg = grad[ids, 1].detach().clamp_min(0)
        out["s_pos"].add_(g_pos)
        out["s_neg"].add_(g_neg)
        has_pos = g_pos >= float(min_view_support)
        out["pos_views"].add_(has_pos.float())
        out["rgb_views"].add_(
            ((g_pos + g_neg) >= float(min_view_support)).float()
        )

        # A trusted seed must also agree with a reliable foreground surface.
        proj = project_points_to_cam(probe_points.reshape(-1, 3), cam).view(
            count,
            probe_points.shape[1],
            3,
        )
        u, v, depth = proj[..., 0], proj[..., 1], proj[..., 2]
        in_image = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (depth > 1e-4)
        alpha0 = _sample_map(ref.alpha_by_name[name], u, v, mode="bilinear")
        inv_depth0 = _sample_map(ref.depth_by_name[name], u, v, mode="bilinear")
        # The bundled diff-gaussian rasterizer returns accumulated expected
        # inverse depth sum_i(w_i / d_i), despite the public package key being
        # named "depth". Convert it to an alpha-conditioned harmonic depth so
        # it is comparable to the Gaussian centre's camera-space z.
        depth0 = torch.where(
            (alpha0 >= float(a_min)) & (inv_depth0 > 1e-8),
            alpha0 / inv_depth0.clamp_min(1e-8),
            torch.zeros_like(inv_depth0),
        )
        mask_at = _sample_map(cam.object_mask, u, v, mode="nearest")
        valid_at = _sample_map(cam.mask_valid, u, v, mode="nearest") > 0.5
        reliable = in_image & valid_at & (alpha0 >= float(a_min)) & (depth0 > 1e-4)
        tolerance = float(depth_tol) * depth0.clamp_min(1e-3)
        near_surface = reliable & ((depth - depth0).abs() <= tolerance)
        foreground_probe = near_surface & (mask_at > 0.5)
        background_probe = near_surface & (mask_at <= 0.5)
        free_probe = reliable & (depth < depth0 - tolerance)
        negative_probe = background_probe | free_probe
        foreground_surface = foreground_probe.any(dim=1)
        negative_fraction = negative_probe.float().mean(dim=1)
        negative_view = (
            (negative_fraction >= float(probe_negative_fraction))
            & ~foreground_surface
        )
        background_count = background_probe.sum(dim=1)
        free_count = free_probe.sum(dim=1)
        background_surface = negative_view & (background_count >= free_count)
        known_free = negative_view & (free_count > background_count)

        out["fg_surface_views"].add_(foreground_surface.float())
        out["bg_surface_views"].add_(background_surface.float())
        out["free_views"].add_(known_free.float())
        out["joint_fg_views"].add_((foreground_surface & has_pos).float())
        out["foreground_probe_hits"].add_(foreground_probe.sum(dim=1).float())
        out["negative_probe_fraction_sum"].add_(negative_fraction)

        del aux, pkg, image, phi, grad, g_pos, g_neg

    return out


def _bounded_seed_connectivity(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    seed_mask: torch.Tensor,
    connectable_mask: torch.Tensor,
    *,
    radius_mult: float,
    max_hops: int,
) -> Tuple[torch.Tensor, float]:
    """Grow from trusted seeds with both an edge-radius and a hop limit."""

    device = xyz.device
    keep = seed_mask.clone()
    if not seed_mask.any() or not connectable_mask.any() or int(max_hops) <= 0:
        return keep, 0.0

    seed_scale = scales[seed_mask].clamp_min(1e-4)
    radius = max(float(radius_mult) * float(seed_scale.median().item()), 1e-4)
    radius2 = radius * radius
    voxel = radius

    active_ids = torch.where(connectable_mask)[0]
    xyz_active = xyz[active_ids]
    q = torch.floor(xyz_active / voxel).to(torch.int64)
    qmin = q.min(dim=0).values
    qmax = q.max(dim=0).values
    qn = q - qmin
    span = (qmax - qmin + 1).to(torch.int64).clamp(min=1)
    volume = int((span[0] * span[1] * span[2]).item())
    use_linear = 0 < volume <= (1 << 40)
    nx, ny = span[0], span[1]
    if use_linear:
        key = qn[:, 0] + qn[:, 1] * nx + qn[:, 2] * (nx * ny)
    else:
        key = qn[:, 0] * 73856093 ^ qn[:, 1] * 19349663 ^ qn[:, 2] * 83492791

    unique_key, inverse = torch.unique(key, return_inverse=True)
    voxel_count = int(unique_key.numel())
    voxel_qn = torch.zeros(voxel_count, 3, dtype=qn.dtype, device=device)
    voxel_qn[inverse] = qn
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        device=device,
        dtype=qn.dtype,
    )
    neighbour_q = voxel_qn[:, None, :] + offsets[None, :, :]
    in_bounds = ((neighbour_q >= 0) & (neighbour_q < span[None, None, :])).all(dim=-1)
    if use_linear:
        neighbour_key = (
            neighbour_q[..., 0]
            + neighbour_q[..., 1] * nx
            + neighbour_q[..., 2] * (nx * ny)
        )
    else:
        neighbour_key = (
            neighbour_q[..., 0] * 73856093
            ^ neighbour_q[..., 1] * 19349663
            ^ neighbour_q[..., 2] * 83492791
        )

    sorted_key, sort_idx = torch.sort(unique_key)
    flat = neighbour_key.reshape(-1)
    pos = torch.searchsorted(sorted_key, flat).clamp(max=max(sorted_key.numel() - 1, 0))
    match = (sorted_key[pos] == flat) & in_bounds.reshape(-1)
    neighbour_voxel = torch.full(
        (flat.numel(),),
        -1,
        dtype=torch.long,
        device=device,
    )
    neighbour_voxel[match] = sort_idx[pos[match]]
    neighbour_voxel = neighbour_voxel.view(voxel_count, 27)

    active_local_keep = keep[active_ids].clone()
    active_voxel = torch.zeros(voxel_count, dtype=torch.bool, device=device)
    active_voxel[inverse[active_local_keep]] = True
    point_chunk, kept_chunk = 4096, 1024

    for _ in range(int(max_hops)):
        valid_neighbour = neighbour_voxel >= 0
        touches_kept = torch.where(
            valid_neighbour,
            active_voxel[neighbour_voxel.clamp_min(0)],
            torch.zeros_like(valid_neighbour),
        ).any(dim=1)
        frontier = touches_kept[inverse] & ~active_local_keep
        if not frontier.any():
            break

        kept_points = xyz_active[active_local_keep]
        frontier_ids = torch.where(frontier)[0]
        frontier_points = xyz_active[frontier_ids]
        newly_connected = torch.zeros(frontier_ids.numel(), dtype=torch.bool, device=device)
        for start in range(0, frontier_points.shape[0], point_chunk):
            points = frontier_points[start : start + point_chunk]
            best_distance2 = torch.full(
                (points.shape[0],),
                float("inf"),
                dtype=points.dtype,
                device=device,
            )
            for kept_start in range(0, kept_points.shape[0], kept_chunk):
                neighbours = kept_points[kept_start : kept_start + kept_chunk]
                distance2 = (points[:, None, :] - neighbours[None, :, :]).pow(2).sum(-1)
                best_distance2 = torch.minimum(best_distance2, distance2.min(dim=1).values)
            newly_connected[start : start + points.shape[0]] = best_distance2 <= radius2
        if not newly_connected.any():
            break
        active_local_keep[frontier_ids[newly_connected]] = True
        active_voxel[inverse[active_local_keep]] = True

    keep[active_ids] = active_local_keep
    return keep, radius


def _distribution(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    x = values.detach().float()
    q = torch.quantile(
        x,
        torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=x.device),
    )
    return {
        "min": float(q[0].item()),
        "p10": float(q[1].item()),
        "median": float(q[2].item()),
        "p90": float(q[3].item()),
        "max": float(q[4].item()),
    }


def run_post_tau_admission(
    gaussians,
    cameras: Sequence,
    ref,
    pipe,
    background: torch.Tensor,
    *,
    export_tau: float,
    base_eta: Optional[torch.Tensor] = None,
    a_min: float = 0.05,
    depth_tol: float = 0.05,
    min_view_support: float = 1e-4,
    support_abs_min: float = 1e-3,
    support_rel_median: float = 0.05,
    min_seed_views: int = 2,
    min_surface_views: int = 2,
    min_purity: float = 0.6,
    radius_mult: float = 2.0,
    max_hops: int = 4,
    weak_rescue: bool = False,
    weak_support_rel: float = 0.1,
    weak_rgb_views: int = 2,
    weak_radius_mult: float = 12.0,
    low_z_bridge: bool = False,
    bridge_delta: float = 0.1,
    ellipsoid_hard_negative: bool = False,
    probe_sigma: float = 1.0,
    probe_negative_fraction: float = 0.7,
) -> PostTauAdmissionResult:
    """Propose a conservative export-only eta after the final tau is known."""

    device = gaussians.get_xyz.device
    z = gaussians.get_label.reshape(-1).detach()
    export_candidate = z >= float(export_tau)
    if base_eta is not None:
        base_eta_mask = base_eta.reshape(-1).bool().to(device)
        export_candidate = export_candidate & base_eta_mask
    else:
        base_eta_mask = None
    candidate_ids = torch.where(export_candidate)[0]
    n_all = int(z.numel())
    eta = torch.zeros(n_all, dtype=torch.bool, device=device)

    if candidate_ids.numel() == 0:
        return PostTauAdmissionResult(
            eta=eta,
            candidate_ids=candidate_ids,
            seed_local=torch.zeros(0, dtype=torch.bool, device=device),
            hard_negative_local=torch.zeros(0, dtype=torch.bool, device=device),
            admitted_local=torch.zeros(0, dtype=torch.bool, device=device),
            diagnostics={},
            stats={
                "enabled": True,
                "proposal_applied": False,
                "fallback_reason": "no_export_candidates",
                "num_candidates": 0,
                "num_admitted": 0,
            },
        )

    bridge_threshold = max(0.0, float(export_tau) - max(0.0, float(bridge_delta)))
    graph_candidate = export_candidate.clone()
    if bool(low_z_bridge):
        graph_candidate = graph_candidate | (z >= bridge_threshold)
        if base_eta_mask is not None:
            # A pre-existing high-z eta veto remains authoritative. Lower-z
            # nodes are transit-only and can never enter the export.
            graph_candidate = graph_candidate & (
                (z < float(export_tau)) | base_eta_mask
            )
    graph_ids = torch.where(graph_candidate)[0]
    export_local = export_candidate[graph_ids]

    evidence = _collect_candidate_evidence(
        gaussians,
        cameras,
        ref,
        pipe,
        background,
        graph_ids,
        a_min=float(a_min),
        depth_tol=float(depth_tol),
        min_view_support=float(min_view_support),
        ellipsoid_probes=bool(ellipsoid_hard_negative),
        probe_sigma=float(probe_sigma),
        probe_negative_fraction=float(probe_negative_fraction),
    )
    s_pos = evidence["s_pos"]
    s_neg = evidence["s_neg"]
    mass = s_pos + s_neg
    purity = torch.where(mass > 0, s_pos / mass.clamp_min(1e-8), torch.zeros_like(mass))
    joint = evidence["joint_fg_views"]
    rgb_views = evidence["rgb_views"]
    fg_surface = evidence["fg_surface_views"]
    bg_surface = evidence["bg_surface_views"]
    free = evidence["free_views"]

    reference_pool = (
        export_local
        & (joint >= float(min_seed_views))
        & (fg_surface >= float(min_surface_views))
        & (purity >= float(min_purity))
    )
    if reference_pool.any():
        support_reference = float(s_pos[reference_pool].median().item())
    else:
        positive = s_pos[export_local & (s_pos > 0)]
        support_reference = float(positive.median().item()) if positive.numel() else 0.0
    support_floor = max(
        float(support_abs_min),
        float(support_rel_median) * support_reference,
    )

    negative_views = bg_surface + free
    no_foreground_probe = (
        fg_surface < 1.0
        if bool(ellipsoid_hard_negative)
        else joint < 1.0
    )
    hard_negative = (
        (negative_views >= 2.0)
        & (negative_views > fg_surface + 1.0)
        & no_foreground_probe
    )
    seed = (
        export_local
        & (s_pos >= support_floor)
        & (joint >= float(min_seed_views))
        & (fg_surface >= float(min_surface_views))
        & (purity >= float(min_purity))
        & ~hard_negative
        & (negative_views <= fg_surface + 1.0)
    )

    weak_support_floor = max(
        float(support_abs_min),
        float(weak_support_rel) * float(support_floor),
    )
    rescue_strong_negative = (
        (negative_views >= 2.0)
        & (negative_views > fg_surface + 1.0)
    )
    bridge_local = ~export_local
    bridge_eligible = (
        bridge_local
        & (fg_surface >= 1.0)
        & (s_pos >= float(support_abs_min))
        & ~hard_negative
        & ~rescue_strong_negative
    )

    graph_xyz = gaussians.get_xyz[graph_ids].detach()
    graph_scales = gaussians.get_scaling[graph_ids].mean(dim=-1).detach()
    fallback_reason = None
    if not seed.any():
        # Fail open: absence of trustworthy evidence must not erase the asset.
        admitted_graph = export_local.clone()
        direct_admitted_graph = admitted_graph.clone()
        fallback_reason = "no_reliable_seed"
        radius = 0.0
    else:
        direct_connectable = export_local & ~hard_negative
        if bool(low_z_bridge):
            direct_admitted_graph, _ = _bounded_seed_connectivity(
                graph_xyz,
                graph_scales,
                seed,
                direct_connectable,
                radius_mult=float(radius_mult),
                max_hops=int(max_hops),
            )
        else:
            direct_admitted_graph = torch.zeros_like(seed)
        connectable = direct_connectable | bridge_eligible
        admitted_graph, radius = _bounded_seed_connectivity(
            graph_xyz,
            graph_scales,
            seed,
            connectable,
            radius_mult=float(radius_mult),
            max_hops=int(max_hops),
        )
        if not bool(low_z_bridge):
            direct_admitted_graph = admitted_graph.clone()

    bridge_recovered_graph = (
        admitted_graph
        & export_local
        & ~direct_admitted_graph
    )
    # A weak point may only attach to the established object for one hop.
    # It cannot override substantial background/free-space evidence and cannot
    # initiate or propagate a detached component.
    weak_fg = (
        (joint >= 1.0)
        & (fg_surface >= 1.0)
        & (s_pos >= weak_support_floor)
    )
    weak_rgb = (
        (rgb_views >= float(weak_rgb_views))
        & (s_pos >= weak_support_floor)
    )
    weak_candidate = (
        export_local
        & ~admitted_graph
        & ~hard_negative
        & ~rescue_strong_negative
        & (weak_fg | weak_rgb)
    )
    weak_rescued = torch.zeros_like(weak_candidate)
    weak_radius = 0.0
    admitted_export_graph = admitted_graph & export_local
    if bool(weak_rescue) and admitted_export_graph.any() and weak_candidate.any():
        rescued_graph, weak_radius = _bounded_seed_connectivity(
            graph_xyz,
            graph_scales,
            admitted_export_graph,
            admitted_export_graph | weak_candidate,
            radius_mult=float(weak_radius_mult),
            max_hops=1,
        )
        weak_rescued = rescued_graph & ~admitted_graph & weak_candidate
        admitted_graph = admitted_graph | weak_rescued

    candidate_ids = graph_ids[export_local]
    seed_local = seed[export_local]
    hard_negative_local = hard_negative[export_local]
    admitted_local = admitted_graph[export_local]
    weak_candidate_local = weak_candidate[export_local]
    weak_rescued_local = weak_rescued[export_local]
    eta[candidate_ids[admitted_local]] = True
    rejected = ~admitted_local
    export_evidence = {
        name: value[export_local]
        for name, value in evidence.items()
    }
    export_mass = mass[export_local]
    export_purity = purity[export_local]
    diagnostics = {
        **export_evidence,
        "support_mass": export_mass,
        "purity": export_purity,
        "weak_candidate": weak_candidate_local,
        "weak_rescued": weak_rescued_local,
        "z": z[candidate_ids],
        "opacity": gaussians.get_opacity.reshape(-1)[candidate_ids].detach(),
    }
    stats = {
        "enabled": True,
        "proposal_applied": fallback_reason is None,
        "fallback_reason": fallback_reason,
        "export_tau": float(export_tau),
        "num_views": int(len(cameras)),
        "camera_names": [c.image_name for c in cameras],
        "num_candidates": int(candidate_ids.numel()),
        "num_graph_nodes": int(graph_ids.numel()),
        "num_seeds": int(seed_local.sum().item()),
        "num_hard_negative": int(hard_negative_local.sum().item()),
        "num_admitted": int(admitted_local.sum().item()),
        "num_rejected": int(rejected.sum().item()),
        "keep_fraction": float(admitted_local.float().mean().item()),
        "low_z_bridge_enabled": bool(low_z_bridge),
        "bridge_delta": float(bridge_delta),
        "bridge_threshold": float(bridge_threshold),
        "num_bridge_nodes": int(bridge_local.sum().item()),
        "num_bridge_eligible": int(bridge_eligible.sum().item()),
        "num_bridge_nodes_used": int((admitted_graph & bridge_eligible).sum().item()),
        "num_export_recovered_by_bridge": int(bridge_recovered_graph.sum().item()),
        "ellipsoid_hard_negative_enabled": bool(ellipsoid_hard_negative),
        "probe_sigma": float(probe_sigma),
        "probe_negative_fraction": float(probe_negative_fraction),
        "probe_count": 7 if bool(ellipsoid_hard_negative) else 1,
        "support_abs_min": float(support_abs_min),
        "support_rel_median": float(support_rel_median),
        "support_reference_median": float(support_reference),
        "support_floor": float(support_floor),
        "min_view_support": float(min_view_support),
        "min_seed_views": int(min_seed_views),
        "min_surface_views": int(min_surface_views),
        "min_purity": float(min_purity),
        "radius_mult": float(radius_mult),
        "connectivity_radius": float(radius),
        "max_hops": int(max_hops),
        "weak_rescue_enabled": bool(weak_rescue),
        "weak_support_rel": float(weak_support_rel),
        "weak_support_floor": float(weak_support_floor),
        "weak_rgb_views": int(weak_rgb_views),
        "weak_radius_mult": float(weak_radius_mult),
        "weak_connectivity_radius": float(weak_radius),
        "num_weak_candidates": int(weak_candidate_local.sum().item()),
        "num_weak_rescued": int(weak_rescued_local.sum().item()),
        "candidate_s_pos": _distribution(export_evidence["s_pos"]),
        "candidate_mass": _distribution(export_mass),
        "candidate_purity": _distribution(export_purity),
        "rejected_s_pos": _distribution(export_evidence["s_pos"][rejected]),
        "rejected_mass": _distribution(export_mass[rejected]),
        "rejected_z": _distribution(z[candidate_ids][rejected]),
    }
    return PostTauAdmissionResult(
        eta=eta,
        candidate_ids=candidate_ids,
        seed_local=seed_local,
        hard_negative_local=hard_negative_local,
        admitted_local=admitted_local,
        diagnostics=diagnostics,
        stats=stats,
    )


@torch.no_grad()
def evaluate_hard_export(
    cameras: Sequence,
    gaussians,
    pipe,
    hard_mask: torch.Tensor,
    *,
    tau_alpha: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a proposed hard export without creating a temporary model."""

    device = gaussians.get_xyz.device
    hard = hard_mask.reshape(-1).to(device=device, dtype=gaussians.get_opacity.dtype)
    opacity = (hard * gaussians.get_opacity.reshape(-1)).clamp(0, 1)
    ones = torch.ones(hard.numel(), 3, device=device, dtype=torch.float32)
    bg0 = torch.zeros(3, device=device, dtype=torch.float32)
    ious: List[float] = []
    recalls: List[float] = []
    hallucinations: List[float] = []

    for cam in cameras:
        if not getattr(cam, "has_object_mask", False):
            continue
        pkg = render(
            cam,
            gaussians,
            pipe,
            bg0,
            override_color=ones,
            override_opacity=opacity[:, None],
        )
        alpha = pkg["render"][:1].clamp(0, 1)
        pred = alpha >= float(tau_alpha)
        gt = cam.object_mask.to(device) > 0.5
        valid = cam.mask_valid.to(device) > 0.5
        pred = pred & valid
        gt = gt & valid
        if not gt.any():
            hallucinations.append(float(pred.float().mean().item()))
            continue
        intersection = (pred & gt).sum().float()
        union = (pred | gt).sum().float().clamp_min(1.0)
        ious.append(float((intersection / union).item()))
        recalls.append(float((intersection / gt.sum().float().clamp_min(1.0)).item()))

    return {
        "mIoU": float(sum(ious) / len(ious)) if ious else 0.0,
        "mean_recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "hallucination_mean": (
            float(sum(hallucinations) / len(hallucinations)) if hallucinations else 0.0
        ),
        "num_fg_views": int(len(ious)),
        "num_empty_views": int(len(hallucinations)),
        "N_obj": int(hard_mask.sum().item()),
    }


def save_candidate_diagnostics(path: str, result: PostTauAdmissionResult, gaussians) -> None:
    """Save compact per-candidate evidence for component/floater inspection."""

    ids = result.candidate_ids.detach()
    xyz = gaussians.get_xyz[ids].detach()
    arrays = {
        "gaussian_id": ids.cpu().numpy(),
        "xyz": xyz.cpu().numpy(),
        "seed": result.seed_local.detach().cpu().numpy(),
        "hard_negative": result.hard_negative_local.detach().cpu().numpy(),
        "admitted": result.admitted_local.detach().cpu().numpy(),
    }
    for name, value in result.diagnostics.items():
        arrays[name] = value.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
