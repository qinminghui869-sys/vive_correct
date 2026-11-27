"""Resolve depth metadata for rendering and measured-center origin calculation."""

import json
from pathlib import Path

import numpy as np

from ego_data.depth_metadata import load_metadata, metadata_path
from vive_correct.data.layout import (
    annotation_dir,
    calibration_file,
    dataset_root,
    timestamp_file,
    video_file,
)
from vive_correct.estimate.geometry import Camera, VIDEO_KEY

DEPTH_KEY = "observation.images.depth_headband_left_rectified"


def depth_input(episode):
    """Return validated rendering parameters, or None for missing depth inputs.

    An existing malformed metadata file is an error, never a fallback trigger.
    Missing masks are explicitly marked as previews and never create source annotations.
    """
    episode = Path(episode)
    ann = annotation_dir(episode, DEPTH_KEY)
    try:
        source = metadata_path(ann)
    except FileNotFoundError:
        source = dataset_root(episode) / "meta/info.json"
        if not source.is_file():
            return None
        feature = json.loads(source.read_text()).get("features", {}).get(DEPTH_KEY)
        if feature is None:
            return None
        storage = feature.get("depth_storage")
        if not storage or not calibration_file(episode).is_file():
            return None
        if storage.get("unit") != "millimeter":
            raise ValueError("Rendering requires millimeter depth storage")
        camera = Camera.load(calibration_file(episode))
        k = camera.K
        meta = dict(
            intrinsics=dict(left=dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])),
            max_depth_m=storage["max_depth_m"],
            invalid_value=storage.get("invalid_value", 0),
            resolution=list(camera.resolution),
        )
        shape = feature.get("shape", [])
        if shape[:2] != [camera.resolution[1], camera.resolution[0]]:
            raise ValueError("Depth and rectified camera resolutions differ")
    else:
        meta = load_metadata(ann)
    k = meta["intrinsics"]["left"]
    values = [k[n] for n in ("fx", "fy", "cx", "cy")] + [meta["max_depth_m"]]
    if not np.isfinite(values).all() or min(k["fx"], k["fy"], meta["max_depth_m"]) <= 0:
        raise ValueError("Invalid depth intrinsics or maximum distance")
    if not all(
        p.is_file()
        for p in (
            video_file(episode, DEPTH_KEY),
            video_file(episode, VIDEO_KEY),
            timestamp_file(episode, DEPTH_KEY),
        )
    ):
        return None
    mask = ann / "valid_mask_ffv1.mkv"
    return dict(
        metadata=meta,
        metadata_source=str(source),
        mask_path=str(mask) if mask.is_file() else None,
        mode="depth_masked" if mask.is_file() else "depth_unmasked_preview",
    )


def depth_available(episode):
    return depth_input(episode) is not None
