"""Rectified camera geometry and the agreed wrist reference axes."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import yaml

from vive_correct.config import estimate_settings

VIDEO_KEY = "observation.images.headband_left_rgb_rectified"
_GEOMETRY = estimate_settings()["geometry"]
LABEL_SIZE_M = (_GEOMETRY["label_length_m"], _GEOMETRY["label_width_m"])
_LENGTH, _WIDTH = LABEL_SIZE_M
LABEL_CORNERS = np.array(
    [
        [-_LENGTH / 2, _WIDTH / 2, 0],
        [_LENGTH / 2, _WIDTH / 2, 0],
        [_LENGTH / 2, -_WIDTH / 2, 0],
        [-_LENGTH / 2, -_WIDTH / 2, 0],
    ]
)
LOCAL_AXES_CONVENTION = "X_left_Y_up_Z_forward"
REFERENCE_FROM_LOCAL = np.eye(3)
MAX_RMS_PX = _GEOMETRY["max_rms_px"]
MIN_EDGE_PX = _GEOMETRY["min_edge_px"]
MAX_X_FROM_LEFT_DEG = _GEOMETRY["max_x_from_left_deg"]
MAX_Y_FROM_UP_DEG = _GEOMETRY["max_y_from_up_deg"]
MAX_Z_FROM_FORWARD_DEG = _GEOMETRY["max_z_from_forward_deg"]
CORNER_NOISE_SIGMA_PX = _GEOMETRY["corner_noise_sigma_px"]
CORNER_NOISE_TRIALS = _GEOMETRY["corner_noise_trials"]


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    resolution: tuple[int, int]
    source: str

    @classmethod
    def load(cls, path: Path):
        doc = yaml.safe_load(path.read_text())
        cameras = doc["cameras"]
        cam = cameras.get(VIDEO_KEY.removeprefix("observation.images."), cameras.get(VIDEO_KEY))
        if cam is None:
            raise ValueError(f"{path}: missing {VIDEO_KEY} intrinsics")
        if (
            cam.get("camera_model") != "pinhole"
            or cam.get("distortion_model") != "none"
            or np.any(cam.get("distortion_coeffs", []))
            or cam.get("feature_key", VIDEO_KEY) != VIDEO_KEY
        ):
            raise ValueError("Expected rectified headband-left pinhole calibration")
        fx, fy, cx, cy = map(float, cam["intrinsics"])
        if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
            raise ValueError("Invalid camera intrinsics")
        resolution = tuple(map(int, cam["resolution"]))
        if len(resolution) != 2 or min(resolution) <= 0:
            raise ValueError("Invalid calibrated resolution")
        return cls(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), resolution, str(path))

    def project(self, points, R, t):
        return cv2.projectPoints(
            np.asarray(points, float),
            cv2.Rodrigues(R)[0],
            t,
            self.K,
            None,
        )[0].reshape(-1, 2)


def order_quad(points, min_edge_px=MIN_EDGE_PX):
    q = np.asarray(points, float)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("Expected four finite image corners")
    q = q[np.argsort(np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean()))]
    if not cv2.isContourConvex(q.astype(np.float32)):
        raise ValueError("Corners must form a convex quadrilateral")
    if cv2.contourArea(q.astype(np.float32), oriented=True) < 0:
        q = q[::-1]
    edges = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
    if edges.min() < min_edge_px:
        raise ValueError(f"Projected label edge is smaller than {min_edge_px:g} px")
    if edges[1] + edges[3] > edges[0] + edges[2]:
        q = np.roll(q, -1, axis=0)
    return np.ascontiguousarray(q)


def pose_candidates(quad, camera, min_edge_px=MIN_EDGE_PX):
    """Keep both planar solutions and both possible long-edge assignments."""
    q = order_quad(quad, min_edge_px)
    candidates = []
    for shift in (0, 1):
        image_points = np.roll(q, shift, axis=0).copy()
        _, rotations, translations, _ = cv2.solvePnPGeneric(
            LABEL_CORNERS,
            image_points,
            camera.K,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        for rvec, tvec in zip(rotations, translations):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rvec, tvec = cv2.solvePnPRefineLM(
                LABEL_CORNERS,
                image_points,
                camera.K,
                None,
                rvec.copy(),
                tvec.copy(),
            )
            R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
            depth = (LABEL_CORNERS @ R.T + t)[:, 2]
            if not np.isfinite(depth).all() or depth.min() <= 0.05 or depth.max() > 3:
                continue
            projected = camera.project(LABEL_CORNERS, R, t)
            rms = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
            candidates.append(dict(R=R, t=t, corners=image_points, rms_px=rms))
    if not candidates:
        raise ValueError("No finite positive-depth label pose")
    return sorted(candidates, key=lambda item: item["rms_px"])


def _angle_deg(vector, reference):
    vector = np.asarray(vector, float)
    reference = np.asarray(reference, float)
    scale = np.linalg.norm(vector) * np.linalg.norm(reference)
    if scale < 1e-12:
        return None
    cosine = np.clip(float(np.dot(vector, reference)) / scale, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def final_axis_angles(rotation, t, camera):
    """Local X/Y project left/up; local Z points forward away from the camera."""
    length = _GEOMETRY["axis_length_m"]
    points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0]], float)
    projected = camera.project(points, rotation, t)
    return (
        _angle_deg(projected[1] - projected[0], [-1, 0]),
        _angle_deg(projected[2] - projected[0], [0, -1]),
        _angle_deg(rotation[:, 2], np.asarray(t, float)),
    )


def _axes_within_limits(angles):
    x_deg, y_deg, z_deg = angles
    if None in angles:
        return False
    return (
        x_deg <= MAX_X_FROM_LEFT_DEG
        and y_deg <= MAX_Y_FROM_UP_DEG
        and z_deg <= MAX_Z_FROM_FORWARD_DEG
    )


def local_basis(R, t, camera):
    """Choose a right-handed left/up/forward reference without moving the origin."""
    facing = np.diag([1.0, -1, -1]) if R[:, 2] @ t > 0 else np.eye(3)
    candidates = []
    for swap in (False, True):
        for a, b in product((-1, 1), repeat=2):
            reference = np.zeros((3, 3))
            reference[2, 2] = -1
            reference[int(swap), 0] = a
            reference[1 - int(swap), 1] = b
            if np.linalg.det(reference) < 0:
                continue
            final = facing @ reference @ REFERENCE_FROM_LOCAL
            rotation = R @ final
            angles = final_axis_angles(rotation, t, camera)
            if not _axes_within_limits(angles):
                continue
            x_deg, y_deg, z_deg = angles
            score = (
                np.cos(np.radians(x_deg)) + np.cos(np.radians(y_deg)) + np.cos(np.radians(z_deg))
            )
            candidates.append((score, final, angles))
    if not candidates:
        raise ValueError("Label axes exceed the configured direction limits")
    return max(candidates, key=lambda item: item[0])[1]


def estimate_pose(
    quad, camera, min_edge_px=MIN_EDGE_PX, *, allow_origin_only=False, device_fixed=True
):
    candidates = [
        item for item in pose_candidates(quad, camera, min_edge_px) if item["rms_px"] <= MAX_RMS_PX
    ]
    if not candidates:
        raise ValueError(f"Label reprojection RMS exceeds {MAX_RMS_PX:g} px")
    best = basis = None
    for candidate in [] if device_fixed else candidates:
        try:
            basis = local_basis(candidate["R"], candidate["t"], camera)
        except ValueError:
            continue
        best = candidate
        break
    if best is None:
        if not (allow_origin_only or device_fixed):
            raise ValueError("Label axes exceed the configured direction limits")
        best = candidates[0]
        return {
            "T_camera_local": None,
            "origin_camera_m": best["t"].tolist(),
            "local_axes_convention": "unresolved",
            "axes_generated": False,
            "axes_failure_reason": "installation_reference_missing"
            if device_fixed
            else "direction_limits",
            "orientation_status": "axes_not_generated",
            "physical_axes_verified": False,
            "image_corners_px": best["corners"].tolist(),
            "rms_px": best["rms_px"],
            "low_resolution": bool(
                min(np.linalg.norm(best["corners"] - np.roll(best["corners"], 1, axis=0), axis=1))
                < MIN_EDGE_PX
            ),
            "plane_ambiguous": bool(
                len(candidates) > 1 and candidates[1]["rms_px"] - best["rms_px"] < 0.5
            ),
            "orientation_180_unresolved": True,
            "true_vive_origin_calibrated": False,
            "T_label_vive": None,
        }

    R, t = best["R"], best["t"]
    local_R = R @ basis
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = local_R, t
    label_from_local = np.eye(4)
    label_from_local[:3, :3] = basis
    direction = final_axis_angles(local_R, t, camera)
    alternatives = []
    for candidate in candidates:
        if candidate is best:
            continue
        angles = []
        for flip in (np.eye(3), np.diag([-1.0, -1, 1])):
            relative = R.T @ candidate["R"] @ flip
            angles.append(
                float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
            )
        if min(angles) > 10:
            alternatives.append(
                {
                    "R_label_to_camera": candidate["R"].tolist(),
                    "t_camera_m": candidate["t"].tolist(),
                    "rms_px": candidate["rms_px"],
                    "rms_gap_px": candidate["rms_px"] - best["rms_px"],
                }
            )
    shortest = float(
        np.linalg.norm(best["corners"] - np.roll(best["corners"], -1, axis=0), axis=1).min()
    )
    return {
        "local_axes_convention": LOCAL_AXES_CONVENTION,
        "axes_generated": True,
        "axes_failure_reason": None,
        "origin_camera_m": t.tolist(),
        "projected_min_edge_px": shortest,
        "low_resolution": shortest < MIN_EDGE_PX,
        "image_corners_px": best["corners"].tolist(),
        "object_corners_local_m": (LABEL_CORNERS @ basis).tolist(),
        "T_camera_local": transform.tolist(),
        "T_label_local": label_from_local.tolist(),
        "R_object_to_camera": local_R.tolist(),
        "t_camera_m": t.tolist(),
        "range_m": float(np.linalg.norm(t)),
        "rms_px": best["rms_px"],
        "axis_direction_deg": {
            "x_from_left": direction[0],
            "y_from_up": direction[1],
            "z_from_forward": direction[2],
        },
        "alternatives": alternatives,
        "plane_ambiguous": any(item["rms_gap_px"] < 0.5 for item in alternatives),
        "orientation_180_unresolved": True,
        "true_vive_origin_calibrated": False,
        "T_label_vive": None,
    }


def corner_noise_sensitivity(
    pose, camera, sigma_px=CORNER_NOISE_SIGMA_PX, trials=CORNER_NOISE_TRIALS
):
    """Conditional pixel-noise sensitivity, not calibrated metric accuracy.

    Plane normals are compared without sign; the separate planar ambiguity
    and 180-degree identity flags remain authoritative.
    """
    rng = np.random.default_rng(0)
    corners = np.array(pose["image_corners_px"])
    transform = np.array(pose["T_camera_local"])
    normal, origin = transform[:3, 1], transform[:3, 3]
    angles, translations = [], []
    for noise in rng.normal(0, sigma_px, (trials, 4, 2)):
        try:
            candidate = pose_candidates(corners + noise, camera, min_edge_px=1)[0]
        except (ValueError, cv2.error):
            continue
        cosine = abs(normal @ candidate["R"][:, 2])
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, 0, 1)))))
        translations.append(float(np.linalg.norm(candidate["t"] - origin)))
    return {
        "method": "iid Gaussian corner perturbation; no calibration or mounting error",
        "assumed_sigma_px": sigma_px,
        "trials": trials,
        "valid_trials": len(angles),
        "plane_normal_p95_deg": float(np.percentile(angles, 95)) if angles else None,
        "origin_displacement_p95_m": float(np.percentile(translations, 95)) if angles else None,
    }
