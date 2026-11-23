"""Validate detected label centers against registered depth, without changing detection."""

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

try:
    from .depth_metadata import load_metadata
except ImportError:
    from depth_metadata import load_metadata

DEPTH_KEY = "observation.images.depth_headband_left_rectified"


def plane_center(depth_m, valid_mask, corners, intrinsics, ray):
    """Intersect the detected center ray with a robust plane inside the label polygon."""
    corners = np.asarray(corners, float)
    ray = np.asarray(ray, float)
    if (
        corners.shape != (4, 2)
        or not np.isfinite(corners).all()
        or ray.shape != (3,)
        or not np.isfinite(ray).all()
        or ray[2] <= 0
    ):
        raise ValueError("Invalid label polygon or ray")
    polygon = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(polygon, np.rint(corners).astype(np.int32), 1)
    polygon = cv2.erode(polygon, np.ones((5, 5), np.uint8))
    yy, xx = np.where(
        (polygon > 0) & valid_mask & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 4)
    )
    coverage = len(xx) / max(1, np.count_nonzero(polygon))
    if len(xx) < 30 or coverage < 0.6:
        raise ValueError("Insufficient valid depth inside label")
    z = depth_m[yy, xx]
    xyz = np.column_stack(
        [
            (xx - intrinsics["cx"]) * z / intrinsics["fx"],
            (yy - intrinsics["cy"]) * z / intrinsics["fy"],
            z,
        ]
    )
    keep = np.abs(z - np.median(z)) <= max(0.01, 4 * np.median(np.abs(z - np.median(z))))
    for _ in range(3):
        if keep.sum() < 30:
            raise ValueError("Too few label plane inliers")
        center = xyz[keep].mean(0)
        _, singular, vt = np.linalg.svd(xyz[keep] - center, full_matrices=False)
        if singular[1] < 0.001:
            raise ValueError("Degenerate label depth patch")
        normal = vt[-1]
        distance = np.abs((xyz - center) @ normal)
        keep = distance <= 0.005
    fraction = float(keep.mean())
    if (
        fraction < 0.85
        or np.percentile(distance, 90) > 0.005
        or abs(normal @ (ray / np.linalg.norm(ray))) < 0.2
    ):
        raise ValueError("Depth patch is not a reliable label plane")
    target = ray * ((normal @ center) / (normal @ ray))
    u = intrinsics["fx"] * target[0] / target[2] + intrinsics["cx"]
    v = intrinsics["fy"] * target[1] / target[2] + intrinsics["cy"]
    if (
        target[2] <= 0
        or cv2.pointPolygonTest(corners.astype(np.float32), (float(u), float(v)), False) < 0
    ):
        raise ValueError("Center ray falls outside label")
    return target, dict(
        valid_pixels=len(xx),
        coverage=coverage,
        inlier_fraction=fraction,
        p90_plane_error_m=float(np.percentile(distance, 90)),
    )


def _packet_position(packet):
    """Timestamp used only to relocate a keyframe. Flush packets are ignored."""
    if type(getattr(packet, "pts", None)) is int:
        return packet.pts
    if type(getattr(packet, "dts", None)) is int:
        return packet.dts
    if packet.pts is None and packet.dts is None:
        return None
    return 0


def _frame_after_keyframe(container, index):
    """Decode the exact counted frame, starting at the preceding keyframe."""
    stream = container.streams.video[0]
    key_at, key_pts, count, found = 0, None, 0, False
    for packet in container.demux(stream):
        position = _packet_position(packet)
        if position is None:
            continue
        if packet.is_keyframe and type(position) is int:
            key_at, key_pts = count, position
        if count == index:
            found = True
            break
        count += 1
    if not found or type(key_pts) is not int:
        return None
    container.seek(key_pts, stream=stream, backward=True, any_frame=False)
    for number, frame in enumerate(container.decode(video=0)):
        if number == index - key_at:
            return frame.to_ndarray()
    return None


def _nth_decoded_frame(path, index):
    """Return decoded frame ``index`` without scanning from the start of a long GOP."""
    if type(index) is not int or index < 0:
        raise ValueError("Depth frame unavailable")
    with av.open(str(path)) as container:
        frame = _frame_after_keyframe(container, index)
    if frame is not None:
        return frame
    with av.open(str(path)) as container:
        for number, decoded in enumerate(container.decode(video=0)):
            if number == index:
                return decoded.to_ndarray()
    raise ValueError("Depth frame unavailable")


def depth_frame(episode, annotation, index):
    """Reuse exact decoded depth/mask pixels across corner edits and approval."""
    from vive_correct.runtime_cache import CACHE, file_stamp

    video = episode / "videos/chunk-000" / DEPTH_KEY / "episode_000000.mp4"
    mask_video = annotation / "valid_mask_ffv1.mkv"
    stamps = (file_stamp(video), file_stamp(mask_video))

    def read():
        # Frame numbers stay exact: packets are counted, then decoding starts at that keyframe.
        depth = _nth_decoded_frame(video, index).astype(float) / 1000
        mask = _nth_decoded_frame(mask_video, index)
        return depth, mask[:, :, 0] if mask.ndim == 3 else mask

    result = CACHE.get(("depth_frame", stamps, index), read)
    if stamps != (file_stamp(video), file_stamp(mask_video)):
        raise ValueError("Depth inputs changed during decoding")
    return result


def refine_targets(episode, report):
    data = episode / "data/chunk-000/episode_000000"
    ann = episode / "annotation/chunk-000" / DEPTH_KEY / "episode_000000"
    result = {}
    observations = report["observations"]
    try:
        meta = load_metadata(ann)
        k = meta["intrinsics"]["left"]
        expected = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
        if not np.allclose(report["camera"]["K"], expected, atol=1e-4):
            raise ValueError("RGB and depth intrinsics differ")
        times = pq.read_table(data / f"{DEPTH_KEY}.parquet")["timestamp"].to_numpy()
        selected = {}
        for o in observations:
            valid = np.flatnonzero(np.isfinite(times))
            if not len(valid):
                raise ValueError("No finite depth timestamps")
            index = int(valid[np.argmin(np.abs(times[valid] - o["timestamp_s"]))])
            if abs(times[index] - o["timestamp_s"]) > 0.005:
                raise ValueError("RGB-depth timestamp gap exceeds 5 ms")
            selected.setdefault(index, []).append(o)
        for index, items in selected.items():
            depth, mask = depth_frame(episode, ann, index)
            for o in items:
                for side, label in o["stickers"].items():
                    key = (o["frame_index"], side)
                    try:
                        original = np.asarray(label["origin_camera_m"], float)
                        target, quality = plane_center(
                            depth,
                            (mask > 0) & (depth <= meta["max_depth_m"]),
                            label["image_corners_px"],
                            k,
                            original,
                        )
                        result[key] = dict(
                            status="depth_plane_refined",
                            origin_camera_m=target.tolist(),
                            original_origin_camera_m=original.tolist(),
                            depth_shift_m=float(target[2] - original[2]),
                            quality=quality,
                            anatomical_accuracy_validated=False,
                        )
                    except (ValueError, KeyError) as error:
                        result[key] = dict(status="unverified_pnp_preview", reason=str(error))
    except (OSError, ValueError, KeyError) as error:
        for o in observations:
            for side in o["stickers"]:
                result[(o["frame_index"], side)] = dict(
                    status="unverified_pnp_preview", reason=str(error)
                )
    for o in observations:
        for side in o["stickers"]:
            result.setdefault(
                (o["frame_index"], side),
                dict(status="unverified_pnp_preview", reason="Depth frame unavailable"),
            )
    return result
