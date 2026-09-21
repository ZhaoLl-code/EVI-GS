"""Attach binary object masks onto loaded Camera objects after Scene construction."""

from __future__ import annotations

import os
from typing import List

import torch

from evi.data.mask_loader import load_binary_mask, resolve_mask_path, summarize_masks


def attach_object_masks(cameras: List, masks_folder: str, object_id: int) -> dict:
    if not masks_folder:
        for cam in cameras:
            h, w = cam.image_height, cam.image_width
            cam.object_mask = torch.zeros((1, h, w), device=cam.data_device)
            cam.mask_valid = torch.zeros((1, h, w), device=cam.data_device)
            cam.has_object_mask = False
        return {"object_id": object_id, "masks_present": 0}

    names = [c.image_name for c in cameras]
    summary = summarize_masks(masks_folder, names, object_id)
    for cam in cameras:
        path = resolve_mask_path(masks_folder, cam.image_name)
        binary, valid, present = load_binary_mask(
            path, object_id, resolution=(cam.image_width, cam.image_height)
        )
        # Masks stay on CUDA even when RGB is on CPU (data_device=cpu).
        cam.object_mask = torch.from_numpy(binary)[None].cuda()
        cam.mask_valid = torch.from_numpy(valid)[None].cuda()
        cam.has_object_mask = bool(present and valid.any())
        cam.mask_path = path
    print(
        f"[EVI] masks: object_id={summary['object_id']} "
        f"present={summary['masks_present']}/{summary['num_images']} "
        f"with_object={summary['masks_with_object']}"
    )
    return summary
