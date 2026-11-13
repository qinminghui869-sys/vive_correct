"""Audit synchronization and attempt episode-specific mounting calibration."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from vive_correct.batch.pose_alignment import fit_mount, load_streams, propagate
from vive_correct.estimate.find_stable_frame import Episode
from vive_correct.estimate.geometry import MAX_RMS_PX
from vive_correct.estimate.label_detection import detect_labels
from vive_correct.batch.report_io import digest


def fixed_pose(corners, object_points, camera, previous):
    _, rotations, translations, _ = cv2.solvePnPGeneric(
        object_points, corners, camera.K, None, flags=cv2.SOLVEPNP_IPPE
    )
    candidates = []
    for rv, tv in zip(rotations, translations):
        rv, tv = cv2.solvePnPRefineLM(object_points, corners, camera.K, None, rv.copy(), tv.copy())
        rotation = cv2.Rodrigues(rv)[0]
        translation = tv.ravel()
        if (
            not np.isfinite(translation).all()
            or (object_points @ rotation.T + translation)[:, 2].min() <= 0.05
        ):
            continue
        error = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (camera.project(object_points, rotation, translation) - corners) ** 2,
                        axis=1,
                    )
                )
            )
        )
        if error > MAX_RMS_PX:
            continue
        angle = Rotation.from_matrix(previous[:3, :3].T @ rotation).magnitude()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        candidates.append((angle, error, matrix))
    if not candidates:
        raise ValueError("Tracked corners fail the fixed-geometry check")
    return min(candidates, key=lambda c: (c[0], c[1]))[2]


def track_reference(episode, anchor, side, radius=45):
    """Keep anchor corner identities; stop at failed flow or boundary agreement."""
    frame = anchor["frame_index"]
    pose = anchor["stickers"][side]
    objects = np.asarray(pose["object_corners_local_m"], float)
    initial = np.asarray(pose["image_corners_px"], np.float32)
    cap = cv2.VideoCapture(str(episode.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    if not ok:
        raise ValueError("Cannot decode calibration anchor")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows = [
        {
            "frame_index": frame,
            "timestamp_s": float(episode.timestamps[frame]),
            "camera_reference": pose["T_camera_local"],
            "physical_axes_verified": bool(pose.get("physical_axes_verified", False)),
            "side": side,
        }
    ]
    stops = []
    for direction in [-1, 1]:
        previous_gray = gray
        points = initial.copy().reshape(4, 1, 2)
        previous_pose = np.asarray(pose["T_camera_local"])
        for step in range(1, radius + 1):
            index = frame + direction * step
            if not 0 <= index < len(episode.timestamps):
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                break
            current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            moved, valid, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current, points, None, winSize=(21, 21), maxLevel=4
            )
            if moved is None or not valid.all() or not np.isfinite(moved).all():
                stops.append({"frame_index": index, "reason": "flow_failed"})
                break
            back, valid, _ = cv2.calcOpticalFlowPyrLK(
                current, previous_gray, moved, None, winSize=(21, 21), maxLevel=4
            )
            if back is None or not valid.all() or np.linalg.norm(back - points, axis=2).max() > 1.5:
                stops.append({"frame_index": index, "reason": "forward_backward_error"})
                break
            predicted = moved.reshape(4, 2)
            lo = np.maximum(0, np.floor(predicted.min(0) - 20)).astype(int)
            hi = np.minimum(episode.camera.resolution, np.ceil(predicted.max(0) + 20)).astype(int)
            matches = []
            if np.all(hi > lo):
                for quad in detect_labels(
                    image, {side: [*lo.tolist(), *hi.tolist()]}, preserve_variants=True
                )[side]:
                    aligned = min(
                        (np.roll(q, s, axis=0) for q in (quad, quad[::-1]) for s in range(4)),
                        key=lambda q: np.sum((q - predicted) ** 2),
                    )
                    distance = float(np.linalg.norm(aligned - predicted, axis=1).max())
                    if distance <= 3:
                        matches.append((distance, aligned))
            if not matches:
                stops.append(
                    {"frame_index": index, "reason": "independent_label_boundary_not_found"}
                )
                break
            corners = min(matches, key=lambda pair: pair[0])[1]
            try:
                matrix = fixed_pose(corners, objects, episode.camera, previous_pose)
            except (ValueError, cv2.error):
                stops.append({"frame_index": index, "reason": "fixed_geometry_failed"})
                break
            angle = np.degrees(
                Rotation.from_matrix(previous_pose[:3, :3].T @ matrix[:3, :3]).magnitude()
            )
            if angle > 15 or np.linalg.norm(matrix[:3, 3] - previous_pose[:3, 3]) > 0.03:
                stops.append({"frame_index": index, "reason": "pose_jump"})
                break
            if step % 3 == 0:
                rows.append(
                    {
                        "frame_index": index,
                        "timestamp_s": float(episode.timestamps[index]),
                        "camera_reference": matrix.tolist(),
                        "physical_axes_verified": bool(pose.get("physical_axes_verified", False)),
                        "side": side,
                        "corners_anchor_order_px": corners.tolist(),
                    }
                )
            previous_gray, points, previous_pose = (
                current,
                corners.astype(np.float32).reshape(4, 1, 2),
                matrix,
            )
    cap.release()
    rows.sort(key=lambda row: row["frame_index"])
    return {
        "side": side,
        "anchor_frame": frame,
        "observations": rows,
        "stops": stops,
        "physical_axes_scope": "Within this uninterrupted anchor track only; no cross-anchor axis equivalence assumed.",
    }


def audit_sync(episode, streams):
    result = {}
    for side, stream in streams.items():
        right = np.searchsorted(stream.times, episode.timestamps)
        inside = (right > 0) & (right < len(stream.times))
        idx = np.clip(right, 1, len(stream.times) - 1)
        gap = stream.times[idx] - stream.times[idx - 1]
        nearest = np.minimum(
            abs(episode.timestamps - stream.times[idx]),
            abs(episode.timestamps - stream.times[idx - 1]),
        )
        valid = inside & (gap <= 0.05)
        invalid = np.searchsorted(stream.invalid_times, stream.times[idx]) - np.searchsorted(
            stream.invalid_times, stream.times[idx - 1], side="right"
        )
        valid &= invalid == 0
        result[side] = {
            "samples": len(stream.times),
            "video_frames_bracketed": int(valid.sum()),
            "invalid_pose_samples": len(stream.invalid_times),
            "video_frame_count": len(episode.timestamps),
            "nearest_timestamp_p95_ms": float(np.percentile(nearest, 95) * 1000),
            "max_pose_gap_ms": float(np.diff(stream.times).max() * 1000),
        }
    return result


def calibrate(report_path, initial_mount_path, output):
    report = json.loads(Path(report_path).read_text())
    episode = Episode.load(report["episode"])
    streams = load_streams(episode.path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initial = json.loads(Path(initial_mount_path).read_text())
    head_camera = np.asarray(initial["T_head_raw_camera"], float).copy()
    rectification = np.asarray(
        yaml.safe_load((episode.path / "calibrations/rectify_params.yaml").read_text())["R1"]
    )
    head_camera[:3, :3] = head_camera[:3, :3] @ rectification.T
    result = {
        "episode": str(episode.path),
        "synchronization": audit_sync(episode, streams),
        "source_report_sha256": digest(report_path),
        "initial_mount_sha256": digest(initial_mount_path),
        "pose_convention": "xyz extrinsic Euler, degrees; evaluated as a model, acquisition convention not independently certified",
        "initial_head_camera": head_camera.tolist(),
        "initial_mount_status": "installation approximation, not validated",
        "tracks": {},
        "accepted": False,
        "propagation": None,
    }
    observations = []
    for side in ["left", "right"]:
        available = [
            o
            for o in report["observations"]
            if side in o["stickers"]
            and o["stickers"][side].get("axes_generated", True)
            and not o["stickers"][side].get("low_resolution")
        ]
        if not available:
            continue
        anchor = max(
            available,
            key=lambda o: abs(
                cv2.contourArea(np.asarray(o["stickers"][side]["image_corners_px"], np.float32))
            ),
        )
        track = track_reference(episode, anchor, side)
        synchronized = []
        for row in track["observations"]:
            try:
                row.update(
                    world_head=streams["head"].at(row["timestamp_s"]).tolist(),
                    world_wrist=streams[side].at(row["timestamp_s"]).tolist(),
                )
            except ValueError:
                continue
            synchronized.append(row)
        # Hold out the ends of a track; fitting never sees these observations.
        for n, row in enumerate(synchronized):
            row["split"] = "validation" if n < 2 or n >= len(synchronized) - 2 else "train"
        track["observations"] = synchronized
        result["tracks"][side] = track
        observations.extend(synchronized)
    if set(result["tracks"]) != {"left", "right"}:
        result["reason"] = "missing_clear_anchor_for_one_or_both_wrists"
    else:
        try:
            fitted = fit_mount(observations, head_camera)
            result["fit"] = fitted
            result["accepted"] = fitted["accepted"]
            result["reason"] = fitted["status"]
            if fitted["accepted"]:
                result["propagation"] = propagate(streams, episode.timestamps, fitted)
        except ValueError as error:
            result["reason"] = str(error)
    (output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def calibrate_manifest(manifest_path, initial_mount, output, indices=None):
    """Calibrate only reports matching the manifest's recorded content."""
    manifest = json.loads(Path(manifest_path).read_text())
    for entry in manifest["entries"]:
        if indices and entry["index"] not in indices:
            continue
        if digest(entry["report"]) != entry["report_sha256"]:
            raise ValueError(f"Report changed: {entry['report']}")
        result = calibrate(entry["report"], initial_mount, Path(output) / f"{entry['index']:02d}")
        counts = {side: len(track["observations"]) for side, track in result["tracks"].items()}
        print(entry["index"], result["accepted"], result["reason"], counts, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-mount", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*")
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    calibrate_manifest(args.manifest, args.initial_mount, args.out, args.indices)


if __name__ == "__main__":
    main()
