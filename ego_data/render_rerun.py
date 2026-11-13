"""Synchronized depth/pose/gaze recording; never invent cross-sensor calibration."""

import argparse
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation
from wrist_alignment import correction_world, hand_mount_summary
from cloud_visibility import local_hand_cloud_mask
from tracking_sync import resample_tracking

ROOT = Path(__file__).resolve().parent
CHAINS = [[0, 1, 2, 3, 4]] + [[0, s, s + 1, s + 2, s + 3] for s in (5, 9, 13, 17)]


def log_hand_projection(path, points, camera, intrinsic, color):
    rr.log(path, rr.Clear(recursive=True))
    points = (points - camera[:3, 3]) @ camera[:3, :3]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.01)
    uv = np.full((21, 2), np.nan)
    uv[valid, 0] = intrinsic["fx"] * points[valid, 0] / points[valid, 2] + intrinsic["cx"]
    uv[valid, 1] = intrinsic["fy"] * points[valid, 1] / points[valid, 2] + intrinsic["cy"]
    if valid.any():
        rr.log(path, rr.Points2D(uv[valid], colors=color, radii=3))
    strips = [uv[c] for c in CHAINS if valid[c].all()]
    if strips:
        rr.log(path + "/bones", rr.LineStrips2D(strips, colors=color, radii=1.5))


def table(path):
    t = pq.read_table(path)
    names = [n for n in t.column_names if n not in ("timestamp", "episode_index")]
    return t["timestamp"].to_numpy(), np.column_stack([t[n].to_numpy() for n in names])


def nearest(stream, timestamp, tolerance):
    times, values = stream
    i = np.searchsorted(times, timestamp)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
    j = min(candidates, key=lambda j: abs(times[j] - timestamp))
    return values[j] if abs(times[j] - timestamp) <= tolerance else np.full(values.shape[1], np.nan)


def frames(path, start, end, rgb=False):
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i > end:
                break
            if i >= start:
                yield frame.to_ndarray(format="rgb24") if rgb else frame.to_ndarray()


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "L3-001-2026-10-27-13-47-37")
    ap.add_argument("--start", type=int, default=360)
    ap.add_argument("--end", type=int, default=540)
    ap.add_argument("--frames", type=int, nargs="+", help="Only log these zero-based frames")
    ap.add_argument(
        "--hand-cloud-fade-radius",
        type=float,
        default=0,
        help="Meters around hand points to thin cloud; 0 disables",
    )
    ap.add_argument(
        "--hand-cloud-retain",
        type=float,
        default=0.2,
        help="Minimum local cloud retention; this is thinning, not alpha blending",
    )
    ap.add_argument(
        "--stride", type=int, default=4, help="Depth spatial subsampling; 1 keeps every pixel"
    )
    ap.add_argument(
        "--pose-euler-sequence",
        default="xyz",
        help="Preview assumption for stored roll/pitch/yaw; lowercase extrinsic, uppercase intrinsic",
    )
    ap.add_argument(
        "--wrist-calibration",
        type=Path,
        help="Wrist correspondence offsets exported by align_wrists.py",
    )
    ap.add_argument(
        "--interpolate-tracking",
        action="store_true",
        help="Interpolate tracking with <=50ms brackets, SLERP rotations, no extrapolation",
    )
    ap.add_argument(
        "--skip-missing-head",
        action="store_true",
        help="Skip frames whose camera cannot be registered; record skipped indices",
    )
    ap.add_argument(
        "--show-uncorrected-wrists",
        action="store_true",
        help="Show original wrist origins in gray and correction segments",
    )
    ap.add_argument(
        "--show-uncorrected-hands",
        action="store_true",
        help="Compare source hand skeletons (gray) against translated preview",
    )
    ap.add_argument(
        "--show-labels", action="store_true", help="Show 3D text labels (hidden by default)"
    )
    ap.add_argument(
        "--hide-axes",
        action="store_true",
        help="Hide world and camera reference axes (pose markers are controlled separately)",
    )
    ap.add_argument(
        "--pose-markers",
        choices=["axes", "spheres"],
        default="axes",
        help="Head and VIVE wrist markers",
    )
    ap.add_argument(
        "--pose-axis-length", type=float, default=0.05, help="Pose marker axis length in meters"
    )
    ap.add_argument(
        "--head-axis-length", type=float, default=0.03, help="Head pose axis length in meters"
    )
    ap.add_argument(
        "--frustum-depth",
        type=float,
        default=(0.10 * 2 / 9),
        help="Moving camera wireframe frustum depth in meters; 0 disables",
    )
    ap.add_argument("--pose-axis-radius", type=float, default=0.0025)
    ap.add_argument("--frustum-radius", type=float, default=0.0013)
    ap.add_argument("--hand-point-radius", type=float, default=0.006)
    ap.add_argument("--cloud-point-radius", type=float, default=0.0015)
    ap.add_argument("--bone-radius", type=float, default=0.002)
    ap.add_argument(
        "--pose-radians",
        action="store_true",
        help="Stored angles in radians; default preview uses degrees",
    )
    ap.add_argument(
        "--world-axes",
        choices=["RUB", "LUF"],
        default="LUF",
        help="Source world-axis convention; values are not changed",
    )
    ap.add_argument("--application-id", help="Name shown in the Rerun recording browser")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/frames_360_540.rrd")
    ap.add_argument(
        "--calibration",
        type=Path,
        help="JSON: T_head_rectified_camera (camera to tracker), head_euler_sequence, head_euler_degrees, eye_to_rectified_homography",
    )
    args = ap.parse_args()
    if args.frames:
        args.frames = sorted(set(args.frames))
        args.start, args.end = min(args.frames), max(args.frames)
    if any(
        v <= 0 or not np.isfinite(v)
        for v in [
            args.pose_axis_radius,
            args.frustum_radius,
            args.hand_point_radius,
            args.cloud_point_radius,
            args.bone_radius,
        ]
    ):
        ap.error("Display radii must be positive finite values")
    if args.hand_cloud_fade_radius < 0 or not 0 <= args.hand_cloud_retain <= 1:
        ap.error("Invalid local cloud thinning parameters")
    if (
        args.stride < 1
        or not 0 <= args.start <= args.end
        or args.pose_axis_length <= 0
        or args.head_axis_length <= 0
        or args.frustum_depth < 0
    ):
        ap.error("Require stride >= 1 and 0 <= start <= end")
    data = args.dataset / "data/chunk-000/episode_000000"
    video = args.dataset / "videos/chunk-000"
    key = "observation.images.depth_headband_left_rectified"
    ann = args.dataset / "annotation/chunk-000" / key / "episode_000000"
    meta = json.loads((ann / "episode_000000_meta.json").read_text())
    k = meta["intrinsics"]["left"]
    with av.open(str(video / key / "episode_000000.mp4")) as container:
        width, height = container.streams.video[0].width, container.streams.video[0].height
    times = pq.read_table(data / f"{key}.parquet")["timestamp"].to_numpy()
    if args.end >= len(times):
        ap.error(f"Last depth frame is {len(times) - 1}")
    streams = {
        n: table(data / f"observation.{n}.parquet")
        for n in (
            "head_pose",
            "left_wrist_pose",
            "right_wrist_pose",
            "hand_left_points",
            "hand_right_points",
            "eye_gaze",
        )
    }
    eye_times = pq.read_table(data / "observation.images.eye_rgb.parquet")["timestamp"].to_numpy()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else {}
    wrist_calibration = (
        json.loads(args.wrist_calibration.read_text()) if args.wrist_calibration else {}
    )
    if wrist_calibration:
        if (
            wrist_calibration.get("episode")
            and Path(wrist_calibration["episode"]).resolve() != args.dataset.resolve()
        ):
            ap.error("Wrist calibration belongs to a different episode")
        if wrist_calibration.get("camera_calibration") != calibration:
            ap.error("Wrist samples were collected with a different camera calibration")
        if wrist_calibration.get("world_axes", args.world_axes) != args.world_axes:
            ap.error("Wrist calibration uses a different world-axis convention")
        if wrist_calibration.get(
            "pose_euler_sequence"
        ) != args.pose_euler_sequence or wrist_calibration.get("pose_degrees") != (
            not args.pose_radians
        ):
            ap.error("Wrist calibration uses a different pose angle convention")
    registered = "T_head_rectified_camera" in calibration
    if args.hand_cloud_fade_radius and not registered:
        ap.error("Local hand/cloud thinning requires registered camera calibration")
    mapped_gaze = "eye_to_rectified_homography" in calibration
    hand_diagnostics = {}
    if args.interpolate_tracking:
        for name, stream in streams.items():
            if name == "eye_gaze":
                continue
            is_head = name == "head_pose"
            streams[name] = resample_tracking(
                stream,
                times,
                pose="pose" in name,
                sequence=calibration.get("head_euler_sequence", args.pose_euler_sequence)
                if is_head
                else args.pose_euler_sequence,
                degrees=calibration.get("head_euler_degrees", not args.pose_radians)
                if is_head
                else not args.pose_radians,
            )
        for side in ("left", "right"):
            hand_diagnostics[side] = hand_mount_summary(
                streams[f"hand_{side}_points"][1],
                streams[f"{side}_wrist_pose"][1],
                args.pose_euler_sequence,
                not args.pose_radians,
            )
    if registered:
        for name in ("head_euler_sequence", "head_euler_degrees"):
            if name not in calibration:
                ap.error(f"Calibration must explicitly specify {name}")
        mount = np.asarray(calibration["T_head_rectified_camera"], dtype=float)
        if mount.shape != (4, 4) or not np.allclose(mount[3], [0, 0, 0, 1]):
            ap.error("T_head_rectified_camera must be a homogeneous 4x4 matrix")
    depth_root = "world/camera" if registered else "depth_camera"
    status = (
        "Depth camera registered to tracking world."
        if registered
        else "UNREGISTERED: depth camera and OpenVR world are shown separately. Missing tracker-camera extrinsic."
    )
    if calibration.get("assumptions"):
        status += "\nCamera mounting is approximate, not independently calibrated."
    status = (
        "Hand skeleton: translation preview only; orientation and wrist mounting are NOT corrected.\n"
        "Cyan: left; pink: right; gray: source (when comparison is enabled).\n" + status
    )
    status += "\n" + (
        "Gaze mapping supplied (2D homography approximation)."
        if mapped_gaze
        else "Gaze shown on eye camera only: missing eye-to-depth calibration."
    )
    status += (
        f"\nTracker local axes preview: Euler {args.pose_euler_sequence}, "
        + ("radians" if args.pose_radians else "degrees")
        + "; angle convention not yet verified from acquisition code. X red, Y green, Z blue."
    )
    status += "\nHand landmarks use the dataset mounting hypothesis; not independently calibrated.\nFrame indices are zero-based, inclusive. Distances: meters.\nNearest timestamps, max gap: poses/hands 5ms, gaze 15ms, eye video 25ms."
    if wrist_calibration:
        status += (
            "\nWrist alignment applied to each wrist and all landmarks of that hand: "
            + json.dumps(wrist_calibration.get("wrists", {}))
        )
        status += "\n" + wrist_calibration.get("status", "")
    if args.interpolate_tracking:
        status += "\nTracking synchronization: linear positions + rotation SLERP, brackets <=50ms, no extrapolation."
        status = status.replace("poses/hands 5ms, ", "")
    if args.hide_axes:
        status += "\nWorld/camera reference axes hidden."
    status += f"\nHead and wrist markers: {args.pose_markers}; local axis length {args.pose_axis_length}m. Moving camera frustum depth {args.frustum_depth}m; wireframe only."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(args.application_id or "ego_depth_360_540")
    rr.save(str(args.output))
    rr.log("notes", rr.TextDocument(status), static=True)
    rr.log("world", getattr(rr.ViewCoordinates, args.world_axes), static=True)
    rr.log(depth_root, rr.ViewCoordinates.RDF, static=True)
    axis_colors = [[240, 60, 60], [50, 210, 80], [60, 130, 255]]
    if not args.hide_axes:
        rr.log(
            "world/reference_axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * 0.15,
                colors=axis_colors,
                labels=(
                    [
                        f"World +X {'left' if args.world_axes == 'LUF' else 'right'}",
                        "World +Y up",
                        f"World +Z {'forward' if args.world_axes == 'LUF' else 'back'}",
                    ]
                    if args.show_labels
                    else None
                ),
                show_labels=args.show_labels,
                radii=0.002,
            ),
            static=True,
        )
    if not args.hide_axes:
        rr.log(
            depth_root + "/reference_axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * 0.10,
                colors=axis_colors,
                labels=["Camera +X right", "Camera +Y down", "Camera +Z forward"]
                if args.show_labels
                else None,
                show_labels=args.show_labels,
                radii=0.0015,
            ),
            static=True,
        )
    if args.frustum_depth > 0:
        d = args.frustum_depth
        corners = np.array(
            [
                [(u - k["cx"]) * d / k["fx"], (v - k["cy"]) * d / k["fy"], d]
                for u, v in [(0, 0), (width, 0), (width, height), (0, height)]
            ]
        )
        strips = [np.vstack([corners, corners[0]])] + [
            np.vstack([np.zeros(3), corner]) for corner in corners
        ]
        rr.log(
            depth_root + "/fov_wireframe",
            rr.LineStrips3D(strips, colors=[255, 200, 70], radii=args.frustum_radius),
            static=True,
        )
    rr.log(
        depth_root + "/image",
        rr.Pinhole(
            focal_length=[k["fx"], k["fy"]],
            principal_point=[k["cx"], k["cy"]],
            resolution=[width, height],
        ),
        static=True,
    )
    views = [
        rrb.Spatial3DView(
            origin="world" if registered else depth_root,
            contents=["+ $origin/**", "- " + depth_root + "/image/**"],
            name="Depth point cloud",
        )
    ]
    if not registered:
        views.append(rrb.Spatial3DView(origin="world", name="OpenVR tracking (unregistered)"))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                *views,
                rrb.Vertical(
                    rrb.Spatial2DView(origin=depth_root + "/image", name="Rectified RGB"),
                    rrb.Spatial2DView(origin="eye", name="Eye camera + gaze"),
                    rrb.TextDocumentView(origin="notes", name="Calibration status"),
                ),
            )
        )
    )
    # Eye video uses its own timestamps, not the depth frame number.
    eye_iter = iter(
        frames(video / "observation.images.eye_rgb/episode_000000.mp4", 0, len(eye_times) - 1, True)
    )
    eye_idx = -1
    eye_image = None
    sources = [
        frames(video / key / "episode_000000.mp4", args.start, args.end),
        frames(ann / "valid_mask_ffv1.mkv", args.start, args.end),
        frames(
            video / "observation.images.headband_left_rgb_rectified/episode_000000.mp4",
            args.start,
            args.end,
            True,
        ),
    ]
    counts = []
    skipped_frames = []
    local_thinning_removed = {}
    selected_frames = set(args.frames) if args.frames else None
    gaze_hits = 0
    for idx, (depth, mask, rgb) in enumerate(zip(*sources, strict=True), args.start):
        if selected_frames is not None and idx not in selected_frames:
            continue
        ts = float(times[idx])
        rr.set_time_sequence("frame", idx)
        rr.set_time_seconds("time", ts)
        head = nearest(streams["head_pose"], ts, 0.005)
        if registered:
            if not np.isfinite(head).all():
                if args.skip_missing_head:
                    skipped_frames.append(idx)
                    continue
                raise ValueError(f"No valid head pose for registered frame {idx}")
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler(
                calibration["head_euler_sequence"],
                head[3:],
                degrees=calibration["head_euler_degrees"],
            ).as_matrix()
            pose[:3, 3] = head[:3]
            camera = pose @ mount
            rr.log(depth_root, rr.Transform3D(translation=camera[:3, 3], mat3x3=camera[:3, :3]))
        z = depth[:: args.stride, :: args.stride].astype(np.float32) / 1000
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        valid = (mask[:: args.stride, :: args.stride] > 0) & (z > 0) & (z <= meta["max_depth_m"])
        yy, xx = np.mgrid[0 : depth.shape[0] : args.stride, 0 : depth.shape[1] : args.stride]
        points = np.stack(((xx - k["cx"]) * z / k["fx"], (yy - k["cy"]) * z / k["fy"], z), axis=-1)
        if args.hand_cloud_fade_radius:
            corrected_hands = []
            for side in ("left", "right"):
                wrist_pose = nearest(streams[side + "_wrist_pose"], ts, 0.005)
                hand_points = nearest(streams["hand_" + side + "_points"], ts, 0.005).reshape(21, 3)
                hand_points = hand_points + correction_world(
                    wrist_calibration,
                    side,
                    wrist_pose,
                    args.pose_euler_sequence,
                    not args.pose_radians,
                )
                corrected_hands.append((hand_points - camera[:3, 3]) @ camera[:3, :3])
            valid_flat = np.flatnonzero(valid)
            keep = local_hand_cloud_mask(
                points[valid],
                np.concatenate(corrected_hands),
                valid_flat,
                args.hand_cloud_fade_radius,
                args.hand_cloud_retain,
            )
            local_thinning_removed[str(idx)] = int((~keep).sum())
            valid.reshape(-1)[valid_flat[~keep]] = False
        rr.log(
            depth_root + "/cloud",
            rr.Points3D(
                points[valid],
                colors=rgb[:: args.stride, :: args.stride][valid],
                radii=args.cloud_point_radius,
            ),
        )
        rr.log(depth_root + "/image/rgb", rr.Image(rgb).compress(jpeg_quality=85))
        for name, color in [
            ("head_pose", [255, 220, 30]),
            ("left_wrist_pose", [30, 220, 255]),
            ("right_wrist_pose", [255, 70, 140]),
            ("hand_left_points", [30, 220, 255]),
            ("hand_right_points", [255, 70, 140]),
        ]:
            value = nearest(streams[name], ts, 0.005)
            pts = value.reshape(-1, 3) if "points" in name else value[:3].reshape(1, 3)
            if "head" not in name:
                side = "left" if "left" in name else "right"
                wrist_pose = nearest(streams[side + "_wrist_pose"], ts, 0.005)
                pts = pts + correction_world(
                    wrist_calibration,
                    side,
                    wrist_pose,
                    args.pose_euler_sequence,
                    not args.pose_radians,
                )
            rr.log("world/" + name, rr.Clear(recursive=True))
            if "wrist_pose" in name and registered and wrist_calibration:
                image_path = depth_root + "/image/" + name
                rr.log(image_path, rr.Clear(recursive=True))
                camera_point = (pts[0] - camera[:3, 3]) @ camera[:3, :3]
                if np.isfinite(camera_point).all() and camera_point[2] > 0:
                    u = k["fx"] * camera_point[0] / camera_point[2] + k["cx"]
                    v = k["fy"] * camera_point[1] / camera_point[2] + k["cy"]
                    rr.log(image_path, rr.Points2D([[u, v]], colors=color, radii=7))
                if args.show_uncorrected_wrists and np.isfinite(value[:3]).all():
                    rr.log(
                        "world/" + name + "/original",
                        rr.Points3D([value[:3]], colors=[160, 160, 160], radii=0.008),
                    )
                    if np.isfinite(pts[0]).all():
                        rr.log(
                            "world/" + name + "/correction",
                            rr.LineStrips3D([[value[:3], pts[0]]], colors=color, radii=0.001),
                        )
            good = np.isfinite(pts).all(axis=1)
            if good.any() and ("points" in name or args.pose_markers == "spheres"):
                label = "head" if name == "head_pose" else "vive_target"
                rr.log(
                    "world/" + name,
                    rr.Points3D(
                        pts[good],
                        colors=color,
                        radii=args.hand_point_radius if "points" in name else 0.015,
                        labels=[label] if args.show_labels and "points" not in name else None,
                        show_labels=args.show_labels and "points" not in name,
                    ),
                )
            if (
                args.pose_markers == "axes"
                and "pose" in name
                and np.isfinite(value).all()
                and np.isfinite(pts[0]).all()
            ):
                orientation = Rotation.from_euler(
                    args.pose_euler_sequence, value[3:], degrees=not args.pose_radians
                ).as_matrix()
                label = {
                    "head_pose": "Head VIVE",
                    "left_wrist_pose": "Left wrist VIVE",
                    "right_wrist_pose": "Right wrist VIVE",
                }[name]
                rr.log(
                    "world/" + name + "/axes",
                    rr.Arrows3D(
                        origins=np.tile(pts[0], (3, 1)),
                        vectors=orientation.T
                        * (args.head_axis_length if name == "head_pose" else args.pose_axis_length),
                        colors=axis_colors,
                        labels=[label + " +X", label + " +Y", label + " +Z"]
                        if args.show_labels
                        else None,
                        show_labels=args.show_labels,
                        radii=args.pose_axis_radius,
                    ),
                )
            if "points" in name:
                if registered:
                    log_hand_projection(depth_root + "/image/" + name, pts, camera, k, color)
                if args.show_uncorrected_hands:
                    original = value.reshape(21, 3)
                    raw_path = "world/" + name + "/source"
                    raw_good = np.isfinite(original).all(axis=1)
                    if raw_good.any():
                        rr.log(
                            raw_path,
                            rr.Points3D(
                                original[raw_good],
                                colors=[150, 150, 150],
                                radii=args.hand_point_radius / 2,
                            ),
                        )
                    raw_strips = [original[c] for c in CHAINS if raw_good[c].all()]
                    if raw_strips:
                        rr.log(
                            raw_path + "/bones",
                            rr.LineStrips3D(
                                raw_strips, colors=[150, 150, 150], radii=args.bone_radius / 2
                            ),
                        )
                    if registered:
                        log_hand_projection(
                            depth_root + "/image/" + name + "/source",
                            original,
                            camera,
                            k,
                            [150, 150, 150],
                        )
                if args.show_labels and np.isfinite(pts[0]).all():
                    rr.log(
                        "world/" + name + "/wrist_label",
                        rr.Points3D(
                            [pts[0]],
                            colors=color,
                            radii=args.hand_point_radius,
                            labels=["wrist"],
                            show_labels=True,
                        ),
                    )
                strips = [pts[c] for c in CHAINS if np.isfinite(pts[c]).all()]
                if strips:
                    rr.log(
                        "world/" + name + "/bones",
                        rr.LineStrips3D(strips, colors=color, radii=args.bone_radius),
                    )
        if wrist_calibration:
            rr.log("world/wrist_alignment_targets", rr.Clear(recursive=True))
            for sample_index, sample in enumerate(wrist_calibration.get("samples", [])):
                if sample["frame"] == idx:
                    rr.log(
                        f"world/wrist_alignment_targets/{sample_index}",
                        rr.Points3D(
                            [sample["target_world_m"]],
                            colors=[255, 255, 0],
                            radii=0.008,
                            labels=[sample["side"] + " wrist target"] if args.show_labels else None,
                            show_labels=args.show_labels,
                        ),
                    )
        target = int(np.argmin(abs(eye_times - ts)))
        while eye_idx < target:
            eye_image = next(eye_iter)
            eye_idx += 1
        rr.log("eye", rr.Clear(recursive=True))
        if abs(eye_times[target] - ts) <= 0.025:
            rr.log("eye/image", rr.Image(eye_image).compress(jpeg_quality=85))
            gaze_eye = nearest(streams["eye_gaze"], float(eye_times[target]), 0.015)
            if np.isfinite(gaze_eye).all():
                rr.log("eye/gaze", rr.Points2D([gaze_eye], colors=[255, 255, 0], radii=10))
        rr.log(depth_root + "/gaze", rr.Clear(recursive=True))
        if mapped_gaze:
            gaze = nearest(streams["eye_gaze"], ts, 0.015)
            uvh = np.asarray(calibration["eye_to_rectified_homography"]) @ np.r_[gaze, 1]
            if np.isfinite(uvh).all() and abs(uvh[2]) > 1e-9:
                u, v = uvh[:2] / uvh[2]
                u, v = int(round(u)), int(round(v))
                if (
                    0 <= u < depth.shape[1]
                    and 0 <= v < depth.shape[0]
                    and mask[v, u] > 0
                    and 0 < depth[v, u] <= 4000
                ):
                    d = float(depth[v, u]) / 1000
                    p = [(u - k["cx"]) * d / k["fx"], (v - k["cy"]) * d / k["fy"], d]
                    rr.log(
                        depth_root + "/gaze", rr.Points3D([p], colors=[255, 255, 0], radii=0.015)
                    )
                    gaze_hits += 1
        counts.append(int(valid.sum()))
        if len(counts) % 30 == 0:
            print(f"frame {idx}, points {counts[-1]}", flush=True)
    if len(counts) + len(skipped_frames) != (
        len(selected_frames) if selected_frames is not None else args.end - args.start + 1
    ):
        raise ValueError("Video ended before requested frame range")
    rr.disconnect()
    if not counts:
        raise ValueError("No frames have valid synchronized head tracking")
    report = dict(
        display_radii_m={
            "pose_axes": args.pose_axis_radius,
            "frustum": args.frustum_radius,
            "hand_points": args.hand_point_radius,
            "cloud_points": args.cloud_point_radius,
            "bones": args.bone_radius,
        },
        world_axes=args.world_axes,
        start=args.start,
        end=args.end,
        frames=len(counts),
        selected_frames=args.frames,
        stride=args.stride,
        local_thinning_removed=local_thinning_removed,
        hand_cloud_fade_radius_m=args.hand_cloud_fade_radius,
        min_points=min(counts),
        max_points=max(counts),
        registered=registered,
        gaze_mapped=mapped_gaze,
        gaze_hits=gaze_hits,
        reference_axes_visible=not args.hide_axes,
        pose_markers=args.pose_markers,
        pose_axis_length_m=args.pose_axis_length,
        head_axis_length_m=args.head_axis_length,
        moving_frustum_depth_m=args.frustum_depth,
        labels_visible=args.show_labels,
        pose_axes_euler_sequence=args.pose_euler_sequence,
        pose_axes_degrees=not args.pose_radians,
        status=status,
    )
    report.update(
        skipped_missing_head_frames=skipped_frames,
        tracking_interpolated=args.interpolate_tracking,
        image_resolution=[width, height],
        hand_alignment="translation_preview_not_rigid_calibration",
        source_hand_mount=hand_diagnostics,
        wrist_calibration=str(args.wrist_calibration) if args.wrist_calibration else None,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
