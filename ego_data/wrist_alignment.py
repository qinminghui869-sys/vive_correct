"""Shared geometry for manual wrist correspondence alignment (meters)."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose, sequence="xyz", degrees=True):
    pose = np.asarray(pose)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Missing synchronized pose")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler(sequence, pose[3:], degrees=degrees).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def backproject(u, v, depth_m, intrinsics):
    return np.array(
        [
            (u - intrinsics["cx"]) * depth_m / intrinsics["fx"],
            (v - intrinsics["cy"]) * depth_m / intrinsics["fy"],
            depth_m,
        ]
    )


def project_world(point, world_camera, intrinsics):
    point = np.asarray(point)
    camera_point = world_camera[:3, :3].T @ (point - world_camera[:3, 3])
    if not np.isfinite(camera_point).all() or camera_point[2] <= 0:
        return None
    x, y, z = camera_point
    return (
        (intrinsics["fx"] * x / z + intrinsics["cx"]),
        (intrinsics["fy"] * y / z + intrinsics["cy"]),
    )


def fit_samples(samples, model):
    if not samples:
        return np.zeros(3), []
    deltas = np.array([s["delta_world_m"] for s in samples])
    rotations = np.array([s["wrist_rotation"] for s in samples])
    candidates = np.einsum("nji,nj->ni", rotations, deltas) if model == "wrist_local" else deltas
    # Coordinate median reduces sensitivity to a single inaccurate depth click.
    offset = np.median(candidates, axis=0)
    predicted = (
        np.einsum("nij,j->ni", rotations, offset)
        if model == "wrist_local"
        else np.tile(offset, (len(samples), 1))
    )
    return offset, np.linalg.norm(predicted - deltas, axis=1).tolist()


def correction_world(config, side, wrist_pose, sequence="xyz", degrees=True):
    entry = config.get("wrists", {}).get(side)
    if not entry:
        return np.zeros(3)
    offset = np.asarray(entry["offset_m"], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid wrist offset")
    if entry["model"] == "world":
        return offset
    if entry["model"] != "wrist_local":
        raise ValueError("Unknown wrist alignment model")
    if not np.isfinite(wrist_pose).all():
        return np.full(3, np.nan)
    return pose_matrix(wrist_pose, sequence, degrees)[:3, :3] @ offset


def hand_mount_summary(points, poses, sequence="xyz", degrees=True):
    """Recover the source wrist landmark offset; this is not a physical calibration."""
    points, poses = np.asarray(points).reshape(-1, 21, 3), np.asarray(poses)
    valid = np.isfinite(points[:, 0]).all(axis=1) & np.isfinite(poses).all(axis=1)
    if not valid.any():
        return {"status": "no_synchronized_samples"}
    rotation = Rotation.from_euler(sequence, poses[valid, 3:], degrees=degrees).as_matrix()
    local = np.einsum("nji,nj->ni", rotation, points[valid, 0] - poses[valid, :3])
    offset = np.median(local, axis=0)
    return {
        "status": "source_mount_hypothesis_not_calibrated",
        "wrist_landmark_in_tracker_m": offset.tolist(),
        "offset_consistency_rms_m": float(np.sqrt(np.mean(np.sum((local - offset) ** 2, axis=1)))),
        "sample_count": int(valid.sum()),
    }
