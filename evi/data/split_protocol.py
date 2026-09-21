"""Input / test camera split protocol with leakage guards."""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def _camera_azimuth_deg(R: np.ndarray, T: np.ndarray) -> float:
    """World-space camera forward azimuth in degrees (COLMAP convention)."""
    # C2W from R^T, T: camera center = -R @ T for COLMAP stored R (transposed w2c rot)
    w2c_R = R.T
    c2w_R = w2c_R.T
    center = -w2c_R @ T
    forward = c2w_R[:, 2]
    az = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    return az % 360.0


def build_sector_split(
    cam_infos: Sequence,
    holdout_ratio: float = 0.2,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """
    Split cameras by contiguous azimuth sector holdout.
    Returns dict with train_rgb / test image_name lists.
    """
    if not cam_infos:
        return {"train_rgb": [], "test": [], "holdout_ratio": holdout_ratio, "seed": seed}

    named = []
    for c in cam_infos:
        az = _camera_azimuth_deg(np.asarray(c.R), np.asarray(c.T))
        named.append((c.image_name, az))
    named.sort(key=lambda x: x[1])

    n = len(named)
    n_test = max(1, int(round(n * holdout_ratio))) if holdout_ratio > 0 else 0
    n_test = min(n_test, max(0, n - 1))  # keep at least one train view when possible

    rng = np.random.RandomState(seed)
    if n_test == 0:
        test_names: List[str] = []
    else:
        start = int(rng.randint(0, n))
        idxs = [(start + i) % n for i in range(n_test)]
        test_names = [named[i][0] for i in idxs]

    test_set = set(test_names)
    train_names = [name for name, _ in named if name not in test_set]
    return {
        "train_rgb": train_names,
        "test": test_names,
        "holdout_ratio": holdout_ratio,
        "seed": seed,
    }


def split_ids(split: Dict) -> Tuple[List[str], List[str]]:
    """Return (train_ids, test_ids) with backward-compatible keys."""
    train = split.get("train_ids") or split.get("train_rgb") or []
    test = split.get("test_ids") or split.get("test") or []
    return list(train), list(test)


def split_hash(train_ids: Sequence[str], test_ids: Sequence[str], extra: Optional[Dict] = None) -> str:
    payload = {
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_stride_split(
    image_names: Sequence[str],
    stride: int = 8,
    protocol: str = "uniform_stride",
) -> Dict[str, List[str]]:
    """
    Uniform sparse holdout: sort names, take every `stride`-th view as test.
    Matches COLMAP eval llffhold=8 used by this repo's dataset reader.
    """
    names = sorted({str(n) for n in image_names})
    if stride < 2:
        raise ValueError(f"stride must be >= 2, got {stride}")
    test_ids = [n for i, n in enumerate(names) if i % int(stride) == 0]
    test_set = set(test_ids)
    train_ids = [n for n in names if n not in test_set]
    extra = {"protocol": protocol, "stride": int(stride)}
    digest = split_hash(train_ids, test_ids, extra)
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_rgb": train_ids,
        "test": test_ids,
        "protocol": protocol,
        "stride": int(stride),
        "hash": digest,
        "num_train": len(train_ids),
        "num_test": len(test_ids),
        "num_total": len(names),
    }


def list_colmap_image_names(source_path: str) -> List[str]:
    """Sorted COLMAP image names (with extension), same order as Scene eval split."""
    bin_path = os.path.join(source_path, "sparse", "0", "images.bin")
    txt_path = os.path.join(source_path, "sparse", "0", "images.txt")
    if os.path.isfile(bin_path):
        from scene.colmap_loader import read_extrinsics_binary
        extr = read_extrinsics_binary(bin_path)
        return sorted(extr[k].name for k in extr)
    if os.path.isfile(txt_path):
        from scene.colmap_loader import read_extrinsics_text
        extr = read_extrinsics_text(txt_path)
        return sorted(extr[k].name for k in extr)
    raise FileNotFoundError(f"No COLMAP images.bin/txt under {source_path}")


def save_split(split: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train, test = split_ids(data)
    data["train_ids"] = train
    data["test_ids"] = test
    data.setdefault("train_rgb", train)
    data.setdefault("test", test)
    return data


class SplitGuard:
    """Assert that training never consumes test image names."""

    def __init__(self, split: Dict):
        train, test = split_ids(split)
        self.test_names: Set[str] = set(test)
        self.train_names: Set[str] = set(train)

    def assert_train_name(self, image_name: str) -> None:
        if image_name in self.test_names:
            raise RuntimeError(
                f"Test leakage: attempted to use test view '{image_name}' during training."
            )

    def assert_batch(self, image_names: Iterable[str]) -> None:
        for name in image_names:
            self.assert_train_name(name)
