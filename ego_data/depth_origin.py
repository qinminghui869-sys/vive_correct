"""Measured center-pixel depth for reviewed origins; no PnP or plane-fit fallback."""

from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from ego_data.depth_targets import DEPTH_KEY
from ego_data.render_inputs import depth_input
from vive_correct.data.layout import timestamp_file, video_file
from vive_correct.runtime.runtime_cache import CACHE, file_stamp


def center_pixel(label):
    if "label_center_px" in label:
        center = np.asarray(label["label_center_px"], float)
    else:
        corners = np.asarray(label["image_corners_px"], float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("Invalid label corners")
        q = np.column_stack((corners, np.ones(4)))
        h = np.cross(np.cross(q[0], q[2]), np.cross(q[1], q[3]))
        if abs(h[2]) < 1e-12:
            raise ValueError("Label diagonals do not define a finite center")
        center = h[:2] / h[2]
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Invalid label center")
    return center


def backproject(depth, center, k, max_depth_m, *, mask=None, invalid_value=0):
    """Sample the nearest pixel, keeping the subpixel center ray; units mm -> m."""
    height, width = depth.shape
    u, v = center
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("Label center outside depth image")
    x, y = np.floor(np.asarray(center) + 0.5).astype(int)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Nearest depth pixel outside image")
    raw = float(depth[y, x])
    z = raw / 1000.0
    if not np.isfinite(z) or raw == invalid_value or not 0 < z <= max_depth_m:
        raise ValueError("Label center has invalid depth; PnP fallback is disabled")
    if mask is not None and (mask.shape != depth.shape or mask[y, x] <= 0):
        raise ValueError("Label center rejected by depth validity mask")
    point = np.array([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z])
    return point, [int(x), int(y)], raw


def decoded_frame(path, index):
    """Count decoded presentation frames, including streams with reordered B-frames."""
    with av.open(str(path)) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number == index:
                return frame.to_ndarray()
    raise ValueError("Depth video ended before the synchronized frame")


def measured_targets(episode, report):
    config = depth_input(episode)
    if config is None:
        raise ValueError(
            "Origin requires depth video, timestamps and calibrated intrinsics; PnP fallback is disabled"
        )
    meta = config["metadata"]
    k = meta["intrinsics"]["left"]
    expected = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    if not np.allclose(report["camera"]["K"], expected, atol=1e-4, rtol=0):
        raise ValueError("RGB and depth intrinsics differ")
    times = pq.read_table(timestamp_file(episode, DEPTH_KEY))["timestamp"].to_numpy()
    finite = np.flatnonzero(np.isfinite(times))
    result = {}
    for observation in report["observations"]:
        ts = observation["timestamp_s"]
        if not np.isfinite(ts) or not len(finite):
            raise ValueError("No finite RGB/depth timestamp")
        index = int(finite[np.argmin(abs(times[finite] - ts))])
        gap = float(times[index] - ts)
        if abs(gap) > 0.005:
            raise ValueError("RGB-depth timestamp gap exceeds 5 ms")
        paths = [video_file(episode, DEPTH_KEY)]
        if config["mask_path"]:
            paths.append(Path(config["mask_path"]))
        stamps = tuple(file_stamp(p) for p in paths)
        arrays = CACHE.get(
            ("origin_depth_pixels", stamps, index),
            lambda: [decoded_frame(p, index) for p in paths],
        )
        if stamps != tuple(file_stamp(p) for p in paths):
            raise ValueError("Depth inputs changed during decoding")
        depth = arrays[0]
        if depth.ndim != 2:
            raise ValueError("Expected single-channel millimeter depth")
        resolution = list(report["camera"].get("resolution", meta.get("resolution", [])))
        if resolution and list(depth.shape[::-1]) != resolution:
            raise ValueError("Depth resolution differs from RGB calibration")
        mask = arrays[1] if len(arrays) > 1 else None
        if mask is not None and mask.ndim == 3:
            mask = mask[:, :, 0]
        for side, label in observation["stickers"].items():
            center = center_pixel(label)
            try:
                point, pixel, raw = backproject(
                    depth,
                    center,
                    k,
                    meta["max_depth_m"],
                    mask=mask,
                    invalid_value=meta.get("invalid_value", 0),
                )
            except ValueError as error:
                raise ValueError(
                    f"RGB frame {observation['frame_index']} {side}: {error}"
                ) from error
            result[(observation["frame_index"], side)] = dict(
                status="depth_pixel_measured",
                origin_camera_m=point.tolist(),
                center_px=center.tolist(),
                sampled_pixel=pixel,
                depth_raw_mm=raw,
                depth_frame=index,
                depth_timestamp_s=float(times[index]),
                rgb_depth_delta_s=gap,
                validity_mask=config["mask_path"],
                metadata_source=config["metadata_source"],
                quality_status="masked_depth" if mask is not None else "unmasked_depth_preview",
                anatomical_accuracy_validated=False,
            )
    return result
