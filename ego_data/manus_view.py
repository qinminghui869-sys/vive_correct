"""RGB and original MANUS nodes, usable without depth or camera calibration."""

import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb

try:
    from .depth_metadata import metadata_path
    from .tracking_sync import normalize_tracking, sample_nearest
except ImportError:
    from depth_metadata import metadata_path
    from tracking_sync import normalize_tracking, sample_nearest

RGB_KEYS = tuple(
    "observation.images." + name
    for name in (
        "headband_left_rgb_rectified",
        "headband_left_rgb",
        "headband_right_rgb",
        "headband_bleft_rgb",
        "headband_bright_rgb",
        "third_rgb",
        "eye_rgb",
    )
)
CHAINS = [[0, 1, 2, 3, 4]] + [[0, i, i + 1, i + 2, i + 3, i + 4] for i in (5, 10, 15, 20)]


def rgb_source(episode):
    from vive_correct.data.layout import timestamp_file, video_file

    episode = Path(episode)
    for key in RGB_KEYS:
        timestamps = timestamp_file(episode, key)
        video = video_file(episode, key)
        if timestamps.is_file() and video.is_file():
            return timestamps, video
    raise ValueError("No supported RGB video with timestamps")


def export(episode, output, *, selected=None, start=0, end=None, labels=False, status=None):
    episode, output = Path(episode), Path(output)
    timestamps, video = rgb_source(episode)
    times = pq.read_table(timestamps)["timestamp"].to_numpy()
    end = len(times) - 1 if end is None else end
    wanted = set(selected) if selected else set(range(start, end + 1))
    if not wanted or min(wanted) < 0 or max(wanted) >= len(times):
        raise ValueError("Requested RGB frames outside stream")
    from vive_correct.data.layout import data_directory

    source = data_directory(episode) / "observation.hand_skeleton.parquet"
    raw = pq.read_table(source)
    (hand_times, points), diagnostics = normalize_tracking(
        (
            raw["timestamp"].to_numpy(),
            np.column_stack([raw[f"node{i:02d}_{a}"].to_numpy() for i in range(50) for a in "xyz"]),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(episode.name + "_manus_rgb")
    rr.save(str(output))
    message = f"RGB source: {timestamps.stem}.\nRGB + original MANUS local nodes. No depth registration or anatomical fit. Group indices are not hand-side labels."
    if status:
        message += "\n" + json.dumps(status, ensure_ascii=False)
    rr.log("notes", rr.TextDocument(message), static=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="rgb", name="RGB"),
                rrb.Vertical(
                    *[
                        rrb.Spatial3DView(
                            origin=f"manus/block_{b}", name=f"MANUS {b}-{b + 24} local"
                        )
                        for b in (0, 25)
                    ],
                    rrb.TextDocumentView(origin="notes", name="Status"),
                ),
            )
        )
    )
    seen, sync = [], []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index > max(wanted):
                break
            if index not in wanted:
                continue
            ts = float(times[index])
            rr.set_time_sequence("frame", index)
            # Frame indexing remains available even if a video timestamp is corrupt.
            rr.log("rgb", rr.Image(frame.to_ndarray(format="rgb24")).compress(jpeg_quality=85))
            value, delta = sample_nearest((hand_times, points), ts, 0.005)
            value = value.reshape(50, 3)
            log_nodes(value, labels=labels)
            delta = float(delta) if np.isfinite(delta) else None
            sync.append(
                dict(
                    frame=index,
                    delta_ms=None if delta is None else delta * 1000,
                    valid=bool(np.isfinite(value).all()),
                )
            )
            seen.append(index)
    rr.disconnect()
    if set(seen) != wanted:
        raise ValueError("RGB video ended before requested frames")
    result = dict(
        rgb_source=timestamps.stem,
        frames=len(seen),
        selected_frames=sorted(seen),
        mode="rgb_manus_only",
        registered=False,
        timestamp_diagnostics=diagnostics,
        hand_sync=sync,
        hand_side_verification=status,
        skipped_missing_head_frames=[],
        status=message,
    )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def depth_available(episode):
    from vive_correct.estimate.geometry import VIDEO_KEY
    from vive_correct.data.layout import annotation_dir, timestamp_file, video_file

    episode = Path(episode)
    key = "observation.images.depth_headband_left_rectified"
    ann = annotation_dir(episode, key)
    try:
        meta = metadata_path(ann)
    except FileNotFoundError:
        return False
    required = [
        meta,
        ann / "valid_mask_ffv1.mkv",
        video_file(episode, key),
        video_file(episode, VIDEO_KEY),
        timestamp_file(episode, key),
    ]
    return all(path.is_file() for path in required)


def log_nodes(nodes, *, labels=False):
    """Log the two source groups without assigning physical hand sides."""
    for block, color in ((0, [255, 180, 40]), (25, [140, 100, 255])):
        path = f"manus/block_{block}"
        rr.log(path, rr.Clear(recursive=True))
        points = nodes[block : block + 25]
        good = np.isfinite(points).all(axis=1)
        if not good.any():
            continue
        rr.log(
            path,
            rr.Points3D(
                points[good],
                colors=color,
                radii=0.002,
                labels=[str(block + i) for i in np.flatnonzero(good)],
                show_labels=labels,
            ),
        )
        strips = [points[chain] for chain in CHAINS if good[chain].all()]
        if strips:
            rr.log(path + "/bones", rr.LineStrips3D(strips, colors=color, radii=0.001))
