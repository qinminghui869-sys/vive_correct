"""Deterministic selection and short-track checks, independent of model confidence."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from vive_correct.batch.mount_calibration import fixed_pose
from vive_correct.batch.report_io import digest


def load_constraints(manifest_path, *, retain_observations=False):
    """Load hash-bound observations; importing a baseline does not verify its identity."""
    if manifest_path is None:
        return {}
    manifest_path = Path(manifest_path).resolve()
    result = {}
    for entry in json.loads(manifest_path.read_text())["entries"]:
        path = Path(entry["report"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if digest(path) != entry["report_sha256"]:
            raise ValueError(f"Axis baseline report changed: {path}")
        report = json.loads(path.read_text())
        episode = str(Path(entry["episode"]).resolve())
        if str(Path(report["episode"]).resolve()) != episode:
            raise ValueError("Axis baseline episode mismatch")
        for observation in report["observations"]:
            for side, pose in observation["stickers"].items():
                pose = deepcopy(pose)
                pose.setdefault(
                    "quality_status", observation.get("quality_status", "conditional_motion")
                )
                review = pose.get("axis_review", {})
                constrained = (
                    pose.get("orientation_status") == "user_constraint_applied"
                    and review.get("reviewer_type") == "user_direction_feedback"
                    and review.get("input_direction_feedback_confirmed") is True
                )
                if not retain_observations and not constrained:
                    continue
                result.setdefault(episode, []).append(
                    {
                        "id": f"constraint_{observation['frame_index']}_{side}",
                        "frame_index": observation["frame_index"],
                        "side_hint": side,
                        "physical_side": side,
                        "quality": 1e9,
                        "pose": deepcopy(pose),
                        "source_axis_convention": report.get(
                            "axis_convention_before_review", report.get("axis_convention", "")
                        ),
                        "baseline_source": str(path),
                        "baseline_sha256": entry["report_sha256"],
                        **(
                            {
                                "constraint_source": str(path),
                                "constraint_sha256": entry["report_sha256"],
                            }
                            if constrained
                            else {}
                        ),
                    }
                )
    return result


def temporal_check(frames, pose, camera, max_fb_px=1.5):
    """Follow ordered corners through both halves; reject jumps, retain planar ambiguity."""
    if pose.get("axes_generated") is False:
        return {"passed": None, "reason": "axes_not_generated"}
    if len(frames) < 3 or len(frames) % 2 == 0:
        return {"passed": False, "reason": "incomplete_window"}
    center = len(frames) // 2
    scale = np.array(frames[0].shape[::-1]) / np.asarray(camera.resolution)
    initial = np.asarray(pose["image_corners_px"], np.float32)
    objects = np.asarray(pose["object_corners_local_m"], float)
    max_angle = max_translation = 0.0
    try:
        for direction in (-1, 1):
            points = (initial * scale).astype(np.float32).reshape(4, 1, 2)
            previous = np.asarray(pose["T_camera_local"], float)
            gray = frames[center]
            for index in range(center + direction, len(frames) if direction > 0 else -1, direction):
                current = frames[index]
                moved, valid, _ = cv2.calcOpticalFlowPyrLK(
                    gray, current, points, None, winSize=(21, 21), maxLevel=3
                )
                if moved is None or not valid.all() or not np.isfinite(moved).all():
                    raise ValueError("tracking_failed")
                back, valid, _ = cv2.calcOpticalFlowPyrLK(
                    current, gray, moved, None, winSize=(21, 21), maxLevel=3
                )
                if (
                    back is None
                    or not valid.all()
                    or not np.isfinite(back).all()
                    or np.linalg.norm((back - points).reshape(4, 2) / scale, axis=1).max()
                    > max_fb_px
                ):
                    raise ValueError("tracking_failed")
                corners = moved.reshape(4, 2) / scale
                if np.any(corners < 2) or np.any(corners > np.asarray(camera.resolution) - 3):
                    raise ValueError("tracked_label_clipped")
                matrix = fixed_pose(corners, objects, camera, previous)
                angle = float(
                    np.degrees(
                        Rotation.from_matrix(previous[:3, :3].T @ matrix[:3, :3]).magnitude()
                    )
                )
                translation = float(np.linalg.norm(previous[:3, 3] - matrix[:3, 3]))
                max_angle, max_translation = (
                    max(max_angle, angle),
                    max(max_translation, translation),
                )
                if angle > 15 or translation > 0.03:
                    raise ValueError("pose_jump")
                gray, points, previous = current, moved, matrix
    except (ValueError, cv2.error) as error:
        return {"passed": False, "reason": str(error)}
    return {
        "passed": True,
        "reason": None,
        "frames": len(frames),
        "max_rotation_step_deg": max_angle,
        "max_translation_step_m": max_translation,
        "absolute_axes_verified": False,
    }


def visual_quality(image, pose):
    q = np.asarray(pose["image_corners_px"])
    lo = np.maximum(np.floor(q.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(q.max(0)).astype(int), image.shape[1::-1])
    crop = image[lo[1] : hi[1], lo[0] : hi[0]]
    sharpness = (
        float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if crop.size
        else 0.0
    )
    edges = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
    return {"sharpness": sharpness, "min_edge_px": float(edges.min())}


def rank(item):
    p = item["pose"]
    return (
        p.get("orientation_status") == "user_constraint_applied",
        p["stability"]["passed"],
        not p.get("low_resolution", False),
        p["visual_quality"]["min_edge_px"],
        p["visual_quality"]["sharpness"],
        -p["rms_px"],
    )


def improves_baseline(candidate, baseline):
    """Require measurable improvement without discarding orientation knowledge."""
    new, old = candidate["pose"], baseline["pose"]
    if old.get("orientation_status") != new.get("orientation_status"):
        return False
    if old.get("orientation_status") == "user_constraint_applied":
        return False
    return (
        all(a >= b for a, b in zip(rank(candidate)[1:3], rank(baseline)[1:3]))
        and new["visual_quality"]["min_edge_px"] >= old["visual_quality"]["min_edge_px"]
        and new["visual_quality"]["sharpness"] > old["visual_quality"]["sharpness"] * 1.1
        and new["rms_px"] <= old["rms_px"]
    )


def choose_pair(accepted):
    choices = {side: max(items, key=rank) for side, items in accepted.items() if items}
    for side, items in accepted.items():
        baselines = [i for i in items if i["pose"].get("baseline_retention")]
        if baselines:
            baseline = max(baselines, key=rank)
            eligible = [i for i in items if improves_baseline(i, baseline)]
            choices[side] = max(eligible, key=rank) if eligible else baseline
    if len(choices) != 2 or any(i["pose"].get("baseline_retention") for i in choices.values()):
        return choices
    pairs = [
        (a, b)
        for a in accepted["left"]
        for b in accepted["right"]
        if a["frame_index"] == b["frame_index"]
    ]
    for a, b in sorted(pairs, key=lambda pair: min(rank(pair[0]), rank(pair[1])), reverse=True):
        pair = {"left": a, "right": b}
        # Same-time coverage cannot displace a user constraint or a stable result.
        if all(rank(pair[s])[:3] >= rank(choices[s])[:3] for s in choices):
            return pair
    return choices


def good_pair(accepted):
    return all(
        any(
            item["pose"]["stability"]["passed"] and not item["pose"].get("low_resolution")
            for item in accepted[side]
        )
        for side in ("left", "right")
    )
