"""Isolated Rerun export: fixed reference world, synchronized eye video."""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

from ego_data.tracking_sync import resample_tracking
from vive_correct.batch.hand_overlay import CHAINS, HandStream
from vive_correct.camera_input import rectify, timestamp_suffix, video_suffix
from vive_correct.eye_reference import POSE_COLUMNS, values
from vive_correct.human_review import decode_frame
from vive_correct.review_io import atomic_json, read_json


def render(episode, reviewed_data, alignment, report, output):
    context = alignment["camera_calibration"]
    times = pq.read_table(episode / timestamp_suffix(context))["timestamp"].to_numpy()
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("眼动视频时间戳必须有限且严格递增")
    hands, poses = {}, {}
    recording = rr.new_recording(episode.name + "_eye_reference")
    recording.save(str(output))
    stats = dict(
        mode="eye_reference_world",
        frames=0,
        skipped_missing_head_frames=[],
        reference_frame=alignment["reference_frame"],
        world_frame=alignment["world_frame"],
        valid_hand_frames={s: 0 for s in ("left", "right")},
        valid_vive_frames={s: 0 for s in ("left", "right")},
        per_frame_camera_registration=False,
        extrinsic_model=alignment.get("extrinsic_model", "legacy_per_hand"),
        shared_fit=alignment.get("shared_fit"),
        orientation_policy=alignment.get("orientation_policy", "transform_source"),
        world_axes=alignment.get("world_axes", "RDF"),
    )
    with recording:
        rr.log(
            "world",
            rr.ViewCoordinates.RUB
            if alignment.get("world_axes") == "RUB"
            else rr.ViewCoordinates.RDF,
            static=True,
        )
        rr.log(
            "world/reference_camera",
            rr.Points3D([[0, 0, 0]], labels=["reference eye camera"]),
            static=True,
        )
        rr.log(
            "notes",
            rr.TextDocument(
                f"World: {alignment['world_frame']} (metres).\n"
                f"Orientation policy: {alignment.get('orientation_policy', 'transform_source')}.\n"
                "Origin-only mode keeps Standing axes and source Vive rotations unchanged.\n"
                "Camera rotation is used only for image projection in this mode.\n"
                "Label centre approximates Vive origin; source Euler decoding remains unverified.\n"
                "Hand mounting [0, -0.085, -0.058] m is an uncalibrated hypothesis.\n"
                "Video is synchronized only: camera translation is unknown. No per-frame world-to-video overlay.\n"
                "双手物理侧别与设备轴向仍需人工确认。\n"
                + json.dumps(alignment.get("shared_fit", {}), ensure_ascii=False, indent=2)
            ),
            static=True,
        )
        reference, _ = decode_frame(episode, alignment["reference_frame"], context)
        for side, pose in report["observations"][0]["stickers"].items():
            quad = np.rint(pose.get("image_corners_px", [])).astype(np.int32)
            if quad.shape == (4, 2):
                cv2.polylines(reference, [quad], True, (0, 255, 255), 2)
            from vive_correct.eye_reference import draw_hand_preview

            draw_hand_preview(reference, pose, side)
            projection = pose.get("axis_projection", {})
            if projection.get("origin"):
                origin = tuple(np.rint(projection["origin"]).astype(int))
                for endpoint, color in zip(
                    projection["endpoints"], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                ):
                    if endpoint:
                        end = tuple(np.rint(np.clip(endpoint, -1e7, 1e7)).astype(int))
                        visible, a, b = cv2.clipLine(
                            (0, 0, reference.shape[1], reference.shape[0]), origin, end
                        )
                        if visible:
                            cv2.arrowedLine(reference, a, b, color, 3)
        rr.log(
            "reference",
            rr.Image(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90),
            static=True,
        )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="参考帧世界：Vive 与手点"),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="video", name="同步眼动视频"),
                        rrb.Spatial2DView(origin="reference", name="审核参考帧"),
                        rrb.TextDocumentView(origin="notes", name="坐标与标定说明"),
                    ),
                )
            )
        )
        for side in ("left", "right"):
            table = pq.read_table(reviewed_data / f"observation.{side}_wrist_pose.parquet")
            pose_times, pose_values = table["timestamp"].to_numpy(), values(table, POSE_COLUMNS)
            poses[side] = resample_tracking((pose_times, pose_values), times, pose=True)[1]
            hands[side] = HandStream(reviewed_data / f"observation.hand_{side}_points.parquet")
            adjacent = np.isfinite(pose_values[:-1]).all(1) & np.isfinite(pose_values[1:]).all(1)
            adjacent &= (np.diff(pose_times) > 0) & (np.diff(pose_times) <= 0.05)
            segments = np.stack([pose_values[:-1, :3], pose_values[1:, :3]], axis=1)[adjacent]
            color = [30, 210, 255] if side == "left" else [240, 90, 220]
            if len(segments):
                rr.log(
                    f"world/{side}/trajectory", rr.LineStrips3D(segments, colors=color), static=True
                )
        with av.open(str(episode / video_suffix(context))) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i >= len(times):
                    raise ValueError("视频帧多于眼动时间戳")
                rr.set_time_sequence("frame", i)
                rr.set_time_seconds("time", float(times[i]))
                rgb = rectify(frame.to_ndarray(format="bgr24"), context)
                rr.log(
                    "video",
                    rr.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=85),
                )
                for side in ("left", "right"):
                    color = [30, 210, 255] if side == "left" else [240, 90, 220]
                    pose = poses[side][i]
                    if np.isfinite(pose).all():
                        rr.log(
                            f"world/{side}/vive",
                            rr.Points3D(
                                [pose[:3]], colors=color, radii=0.008, labels=[side + " Vive"]
                            ),
                        )
                        rotation = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
                        rr.log(
                            f"world/{side}/vive/axes",
                            rr.Arrows3D(
                                origins=np.repeat(pose[None, :3], 3, axis=0),
                                vectors=0.04 * rotation.T,
                                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                                labels=["X", "Y", "Z"],
                            ),
                        )
                        stats["valid_vive_frames"][side] += 1
                    else:
                        rr.log(f"world/{side}/vive", rr.Clear(recursive=True))
                    try:
                        points, _ = hands[side].at(float(times[i]), tolerance=0.05)
                        valid = np.isfinite(points).all(1)
                        if valid.any():
                            rr.log(
                                f"world/{side}/hand/points",
                                rr.Points3D(points[valid], colors=color, radii=0.003),
                            )
                            segments = [
                                [points[a], points[b]]
                                for chain in CHAINS
                                for a, b in zip(chain, chain[1:])
                                if valid[a] and valid[b]
                            ]
                            rr.log(
                                f"world/{side}/hand/bones", rr.LineStrips3D(segments, colors=color)
                            )
                            stats["valid_hand_frames"][side] += 1
                        else:
                            rr.log(f"world/{side}/hand", rr.Clear(recursive=True))
                    except ValueError:
                        rr.log(f"world/{side}/hand", rr.Clear(recursive=True))
                stats["frames"] += 1
        if stats["frames"] != len(times):
            raise ValueError("眼动视频与时间戳帧数不一致")
        recording.flush()
    atomic_json(output.with_suffix(".json"), stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "reviewed-data", "alignment", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    render(
        args.episode,
        args.reviewed_data,
        read_json(args.alignment),
        read_json(args.report),
        args.output,
    )


if __name__ == "__main__":
    main()
