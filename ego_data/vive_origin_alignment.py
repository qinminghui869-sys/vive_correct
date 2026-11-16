"""Convert vive_correct label origins to explicit fixed-world wrist preview offsets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__:
    from .tracking_sync import resample_tracking
    from .wrist_alignment import pose_matrix, fit_samples
else:
    from tracking_sync import resample_tracking
    from wrist_alignment import pose_matrix, fit_samples


def pose_stream(path):
    t = pq.read_table(path)
    return t["timestamp"].to_numpy(), np.column_stack(
        [t[n].to_numpy() for n in ("x", "y", "z", "roll", "pitch", "yaw")]
    )


def build_alignment(
    episode, report_path, calibration_path, *, refine_depth=False, require_bilateral=True
):
    report = json.loads(report_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if Path(report["episode"]).resolve() != episode.resolve():
        raise ValueError("Origin report belongs to another episode")
    data = episode / "data/chunk-000/episode_000000"
    streams = {
        s: pose_stream(data / f"observation.{s}_pose.parquet")
        for s in ("head", "left_wrist", "right_wrist")
    }
    video_times = pq.read_table(data / f"{report['video_key']}.parquet")["timestamp"].to_numpy()
    depth_path = data / "observation.images.depth_headband_left_rectified.parquet"
    depth_times = (
        pq.read_table(depth_path)["timestamp"].to_numpy() if depth_path.is_file() else video_times
    )
    alignment_timeline = "depth" if depth_path.is_file() else "rectified_rgb"
    sequence = calibration["head_euler_sequence"]
    degrees = calibration["head_euler_degrees"]
    mount = np.asarray(calibration["T_head_rectified_camera"])
    if (
        mount.shape != (4, 4)
        or not np.isfinite(mount).all()
        or not np.allclose(mount[3], [0, 0, 0, 1])
        or not np.allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(mount[:3, :3]), 1)
    ):
        raise ValueError("Camera mounting must be a finite rigid 4x4 transform")
    refinements = {}
    if refine_depth:
        from ego_data.depth_targets import refine_targets

        refinements = refine_targets(episode, report)
    samples = []
    for observation in report["observations"]:
        ts = observation["timestamp_s"]
        frame = observation["frame_index"]
        if (
            type(frame) is not int
            or not 0 <= frame < len(video_times)
            or not np.isclose(video_times[frame], ts, rtol=0, atol=1e-6)
        ):
            raise ValueError("Observation timestamp does not match source video")
        head = resample_tracking(
            streams["head"], [ts], pose=True, sequence=sequence, degrees=degrees
        )[1][0]
        camera = pose_matrix(head, sequence, degrees) @ mount
        for side, item in observation["stickers"].items():
            if side not in ("left", "right"):
                raise ValueError(f"Unknown hand side: {side}")
            origin = np.asarray(item["origin_camera_m"], dtype=float)
            if origin.shape != (3,) or not np.isfinite(origin).all() or origin[2] <= 0:
                raise ValueError("Invalid camera origin")
            refinement = refinements.get((frame, side), {"status": "pnp_preview"})
            if refinement["status"] == "depth_plane_refined":
                origin = np.asarray(refinement["origin_camera_m"])
            wrist = resample_tracking(
                streams[side + "_wrist"], [ts], pose=True, sequence=sequence, degrees=degrees
            )[1][0]
            wrist_matrix = pose_matrix(wrist, sequence, degrees)
            target = camera[:3, :3] @ origin + camera[:3, 3]
            samples.append(
                dict(
                    side=side,
                    frame=int(np.argmin(abs(depth_times - ts))),
                    source_rgb_frame=observation["frame_index"],
                    timestamp_s=ts,
                    origin_camera_m=origin.tolist(),
                    target_world_m=target.tolist(),
                    wrist_world_m=wrist[:3].tolist(),
                    delta_world_m=(target - wrist[:3]).tolist(),
                    wrist_rotation=wrist_matrix[:3, :3].tolist(),
                    quality_status=item.get("quality_status"),
                    depth_refinement=refinement,
                )
            )
    wrists = {}
    for side in ("left", "right"):
        selected = [s for s in samples if s["side"] == side]
        if not selected:
            if require_bilateral:
                raise ValueError(f"No repaired origin for {side}")
            continue
        offset, residuals = fit_samples(selected, "world")
        wrists[side] = dict(
            model="world",
            offset_m=offset.tolist(),
            method="vive_correct_label_origin",
            sample_count=len(selected),
            residuals_m=residuals,
        )
    if not wrists:
        raise ValueError("No repaired wrist origins")
    return dict(
        depth_refinement_enabled=refine_depth,
        alignment_timeline=alignment_timeline,
        version=2,
        episode=str(episode.resolve()),
        camera_calibration=calibration,
        world_axes="LUF",
        pose_euler_sequence=sequence,
        pose_degrees=degrees,
        source_report=str(report_path.resolve()),
        source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        synchronization="linear positions + rotation SLERP; bracket gap <= 50 ms; no extrapolation",
        status="Label-center preview; fixed world translation applied to entire episode; not a calibrated Vive internal origin or per-frame visual repair",
        samples=samples,
        wrists=wrists,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "report", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build_alignment(args.episode, args.report, args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["wrists"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
