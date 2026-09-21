"""Default knobs: support_init, optional learn_tau, optional post_tau_admission."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EviConfig:
    # Schedule: P = max(V_fit, min_P); learn_tau steps = learn_export_tau_mult * P
    min_P: int = 64
    a_min: float = 0.05
    # Metadata / debug. Export tau is 0.5 without learn_tau, else the learned value.
    tau_z: float = 0.6
    export_tau_z: float = 0.5
    depth_tol: float = 0.05
    front_support_tau: float = 1e-3
    admit_radius_mult: float = 1.2
    calib_frac: float = 0.2

    # support_init: FlashSplat-style FG/BG support, then z_i = S+ / (S+ + S-)
    support_max_views: int = 8

    # Stage 2/3 default ON; --no_learn_export_tau / --no_post_tau_admission disable.
    # Valid combos: support_init | support_init+learn_tau | all three.
    enable_learn_export_tau: bool = True
    enable_post_tau_admission: bool = True

    # learn_tau: freeze Gaussians, optimize a scalar export label_z threshold
    learn_export_tau_init: float = 0.9
    learn_export_tau_lr: float = 0.01
    learn_export_tau_temp: float = 0.02
    learn_export_tau_min_views: int = 8
    learn_export_tau_mult: int = 1
    learn_export_tau_views: int = 8
    learn_export_tau_views_div: int = 8
    learn_export_tau_patience: int = 0
    learn_export_tau_max_steps: int = 0

    # post_tau_admission: read-only floater cleanup after the learned tau
    post_tau_admission_views: int = 32
    post_tau_interleaved_split: bool = False
    post_tau_calib_frac: float = 0.2
    post_tau_min_view_support: float = 1e-4
    post_tau_support_abs_min: float = 1e-3
    post_tau_support_rel_median: float = 0.05
    post_tau_min_seed_views: int = 2
    post_tau_min_surface_views: int = 2
    post_tau_min_purity: float = 0.6
    post_tau_radius_mult: float = 8.0
    post_tau_max_hops: int = 4
    post_tau_weak_rescue: bool = False
    post_tau_weak_support_rel: float = 0.1
    post_tau_weak_rgb_views: int = 2
    post_tau_weak_radius_mult: float = 12.0
    post_tau_low_z_bridge: bool = False
    post_tau_bridge_delta: float = 0.1
    post_tau_ellipsoid_hard_negative: bool = False
    post_tau_probe_sigma: float = 1.0
    post_tau_probe_negative_fraction: float = 0.7
    post_tau_min_keep_fraction: float = 0.5
    post_tau_max_calib_iou_drop: float = 0.01
    post_tau_max_calib_recall_drop: float = 0.02
