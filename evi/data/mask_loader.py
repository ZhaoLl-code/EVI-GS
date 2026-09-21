"""Object mask loading and object_id selection."""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


def stem_of(path_or_name: str) -> str:
    base = os.path.basename(path_or_name)
    return os.path.splitext(base)[0]


def resolve_mask_path(masks_folder: str, image_name: str) -> str:
    stem = stem_of(image_name)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        candidate = os.path.join(masks_folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(masks_folder, stem + ".png")


def auto_select_object_id(masks_folder: str, image_names) -> int:
    """Pick non-zero id with largest total pixel area across frames."""
    areas: Counter = Counter()
    for name in image_names:
        path = resolve_mask_path(masks_folder, name)
        if not os.path.isfile(path):
            continue
        try:
            m = read_id_mask(path)
        except Exception:
            m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
        ids, counts = np.unique(m, return_counts=True)
        for uid, c in zip(ids.tolist(), counts.tolist()):
            if int(uid) != 0:
                areas[int(uid)] += int(c)
    if not areas:
        raise RuntimeError(f"No non-zero object ids found under {masks_folder}")
    best = areas.most_common(1)[0][0]
    print(f"[EVI] auto-selected object_id={best} (top areas={areas.most_common(5)})")
    return int(best)


def read_id_mask(mask_path: str) -> np.ndarray:
    """Read a single-channel object-id mask (uint8 or uint16). JPEG keeps raw values."""
    im = Image.open(mask_path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_binary_mask(
    mask_path: str,
    object_id: int,
    resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns:
      binary: float32 HxW in {0,1}
      valid: float32 HxW, 1 where annotation exists (file present)
      present: whether file existed
    """
    if not os.path.isfile(mask_path):
        if resolution is None:
            raise FileNotFoundError(mask_path)
        h, w = resolution[1], resolution[0]
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), False

    try:
        m = read_id_mask(mask_path)
    except Exception:
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise IOError(f"Failed to read mask: {mask_path}")
        if m.ndim == 3:
            m = m[..., 0]
    binary = (m == int(object_id)).astype(np.float32)
    valid = np.ones_like(binary, dtype=np.float32)
    if resolution is not None:
        wh = (resolution[0], resolution[1])
        binary = cv2.resize(binary, wh, interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, wh, interpolation=cv2.INTER_NEAREST)
        binary = (binary > 0.5).astype(np.float32)
        valid = (valid > 0.5).astype(np.float32)
    return binary, valid, True


def summarize_masks(masks_folder: str, image_names, object_id: int) -> Dict:
    n_present = 0
    n_pos = 0
    for name in image_names:
        path = resolve_mask_path(masks_folder, name)
        if not os.path.isfile(path):
            continue
        n_present += 1
        try:
            m = read_id_mask(path)
        except Exception:
            m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
        if np.any(m == int(object_id)):
            n_pos += 1
    return {
        "object_id": int(object_id),
        "num_images": len(list(image_names)),
        "masks_present": n_present,
        "masks_with_object": n_pos,
        "masks_folder": masks_folder,
    }


def _mask_files(masks_folder: str, image_names: Optional[Iterable[str]] = None) -> List[str]:
    if image_names is not None:
        paths = [resolve_mask_path(masks_folder, n) for n in image_names]
        return [p for p in paths if os.path.isfile(p)]
    if not os.path.isdir(masks_folder):
        return []
    out = []
    for name in sorted(os.listdir(masks_folder)):
        if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg"}:
            out.append(os.path.join(masks_folder, name))
    return out


def discover_object_ids(
    masks_folder: str,
    image_names: Optional[Sequence[str]] = None,
    min_frames: int = 1,
    min_pixels: int = 100,
) -> Dict:
    """
    Collect non-zero object IDs from dataset GT masks.
    JPEG instance maps with a dominant 255 channel are treated as a single object
    (LLFF-style compressed binary masks); ringing IDs are dropped.
    """
    files = _mask_files(masks_folder, image_names)
    areas: Counter = Counter()
    frames: Counter = Counter()
    jpeg_n = 0
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            jpeg_n += 1
        try:
            m = read_id_mask(path)
        except Exception:
            continue
        ids, counts = np.unique(m, return_counts=True)
        for uid, c in zip(ids.tolist(), counts.tolist()):
            uid = int(uid)
            if uid == 0:
                continue
            areas[uid] += int(c)
            if int(c) > 0:
                frames[uid] += 1

    ids = []
    skipped = []
    for uid, pix in areas.most_common():
        nfr = int(frames.get(uid, 0))
        if nfr < int(min_frames) or int(pix) < int(min_pixels):
            skipped.append({"object_id": int(uid), "pixels": int(pix), "frames": nfr, "reason": "too_small"})
            continue
        ids.append(int(uid))

    jpeg_collapsed = False
    if files and jpeg_n >= max(1, int(0.5 * len(files))) and 255 in areas:
        nz = sum(int(v) for k, v in areas.items())
        if nz > 0 and areas[255] / nz >= 0.8:
            jpeg_collapsed = True
            dropped = [i for i in ids if i != 255]
            ids = [255] if 255 in ids else ids[:1]
            for d in dropped:
                skipped.append({"object_id": int(d), "reason": "jpeg_ringing"})

    return {
        "object_ids": ids,
        "skipped": skipped,
        "num_mask_files": len(files),
        "jpeg_collapsed": jpeg_collapsed,
        "pixel_areas": {str(k): int(v) for k, v in areas.most_common()},
        "frames": {str(k): int(v) for k, v in frames.most_common()},
        "masks_folder": masks_folder,
    }
