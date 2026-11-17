"""Rectified RGB plus corrected world-space wrists when registered depth is absent."""

import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb

from ego_data.tracking_sync import resample_tracking
from ego_data.vive_origin_alignment import pose_stream
from vive_correct.batch.hand_overlay import JOINTS, CHAINS
from vive_correct.estimate.geometry import VIDEO_KEY


def export_corrected_rgb(
    episode,
    output,
    alignment_path,
    *,
    start=0,
    end=None,
    selected=None,
    hand_points_dir=None,
    reviewed_data=None,
):
    episode, output = Path(episode), Path(output)
    alignment = json.loads(Path(alignment_path).read_text())
    if Path(alignment["episode"]).resolve() != episode.resolve():
        raise ValueError("Alignment belongs to another episode")
    data = episode / "data/chunk-000/episode_000000"
    times = pq.read_table(data / f"{VIDEO_KEY}.parquet")["timestamp"].to_numpy()
    end = len(times) - 1 if end is None else end
    wanted = set(selected) if selected else set(range(start, end + 1))
    if not wanted or min(wanted) < 0 or max(wanted) >= len(times):
        raise ValueError("Requested frames outside rectified RGB stream")
    from vive_correct.source_data import original_path

    wrists, hands, offsets, corrected_wrists, original_hands = {}, {}, {}, {}, {}
    for side in ("left", "right"):
        correction = alignment["wrists"][side]
        offset = np.asarray(correction["offset_m"], float)
        if correction["model"] != "world" or offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Expected finite fixed-world wrist offsets")
        offsets[side] = offset
        wrists[side] = resample_tracking(
            pose_stream(data / f"observation.{side}_wrist_pose.parquet"),
            times,
            pose=True,
            sequence=alignment["pose_euler_sequence"],
            degrees=alignment["pose_degrees"],
        )[1][:, :3]
        if reviewed_data:
            corrected_wrists[side] = resample_tracking(
                pose_stream(Path(reviewed_data) / f"observation.{side}_wrist_pose.parquet"),
                times,
                pose=True,
                sequence=alignment["pose_euler_sequence"],
                degrees=alignment["pose_degrees"],
            )[1][:, :3]
            raw = pq.read_table(original_path(data / f"observation.hand_{side}_points.parquet"))
            raw_points = np.column_stack(
                [raw[f"{j}_{a}"].to_numpy() for j in JOINTS for a in "xyz"]
            )
            original_hands[side] = resample_tracking(
                (raw["timestamp"].to_numpy(), raw_points), times
            )[1].reshape(-1, 21, 3)
        # Reviewed files already contain the final positions and retain the original hand shape.
        hand_directory = reviewed_data or hand_points_dir
        if hand_directory:
            table = pq.read_table(Path(hand_directory) / f"observation.hand_{side}_points.parquet")
            points = np.column_stack([table[f"{j}_{a}"].to_numpy() for j in JOINTS for a in "xyz"])
            hands[side] = resample_tracking((table["timestamp"].to_numpy(), points), times)[
                1
            ].reshape(-1, 21, 3)
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(episode.name + "_corrected_rgb", strict=True)
    rr.save(str(output))
    rr.log("world", getattr(rr.ViewCoordinates, alignment.get("world_axes", "LUF")), static=True)
    rr.log(
        "notes",
        rr.TextDocument(
            "User-confirmed label side mapping. Corrected wrists: fixed world translation from label centres. "
            "Not a calibrated VIVE internal origin or per-frame visual tracking. No depth cloud. "
            "RGB is rectified only. Gray: original; color: corrected.\n"
            + json.dumps(alignment, ensure_ascii=False)
        ),
        static=True,
    )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="rgb", name="Rectified RGB"),
                rrb.Spatial3DView(origin="world", name="Corrected wrists and verified hands"),
                rrb.TextDocumentView(origin="notes", name="Provenance"),
            )
        )
    )
    seen, invalid = [], {s: [] for s in wrists}
    try:
        with av.open(
            str(episode / "videos/chunk-000" / VIDEO_KEY / "episode_000000.mp4")
        ) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index > max(wanted):
                    break
                if index not in wanted:
                    continue
                rr.set_time_sequence("frame", index)
                rr.log("rgb", rr.Image(frame.to_ndarray(format="rgb24")).compress(jpeg_quality=85))
                for side, color in (("left", [255, 200, 30]), ("right", [180, 70, 255])):
                    path = f"world/{side}"
                    rr.log(path, rr.Clear(recursive=True))
                    point = wrists[side][index]
                    if not np.isfinite(point).all():
                        invalid[side].append(index)
                        continue
                    corrected = (
                        corrected_wrists[side][index] if reviewed_data else point + offsets[side]
                    )
                    rr.log(
                        path + "/original",
                        rr.Points3D([point], colors=[140, 140, 140], radii=0.008),
                    )
                    rr.log(path + "/corrected", rr.Points3D([corrected], colors=color, radii=0.01))
                    rr.log(
                        path + "/offset",
                        rr.LineStrips3D([[point, corrected]], colors=color, radii=0.001),
                    )
                    if side in hands:
                        points = (
                            hands[side][index]
                            if reviewed_data
                            else hands[side][index] + offsets[side]
                        )
                        if side in original_hands:
                            original = original_hands[side][index]
                            valid = np.isfinite(original).all(axis=1)
                            if valid.any():
                                rr.log(
                                    path + "/hand_original",
                                    rr.Points3D(
                                        original[valid], colors=[140, 140, 140], radii=0.003
                                    ),
                                )
                        if np.isfinite(points).all():
                            rr.log(path + "/hand", rr.Points3D(points, colors=color, radii=0.004))
                            rr.log(
                                path + "/bones",
                                rr.LineStrips3D(
                                    [points[list(c)] for c in CHAINS], colors=color, radii=0.001
                                ),
                            )
                seen.append(index)
        rr.get_global_data_recording().flush()
    finally:
        rr.disconnect()
    if set(seen) != wanted:
        raise ValueError("Incomplete rectified RGB frame coverage")
    result = dict(
        frames=len(seen),
        mode="rectified_rgb_corrected_wrists",
        registered=False,
        vive_correction_displayed=True,
        skipped_missing_head_frames=[],
        invalid_wrist_frames=invalid,
        verified_hand_sides=list(hands),
        source_alignment=str(alignment_path),
        frame_timeline="rectified_rgb",
    )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result
