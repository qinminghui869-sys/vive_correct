"""Synchronize tracked poses and fit rigid camera/wrist mounting transforms.

All transforms map column vectors from the frame named on the right to the
frame on the left. Positions are metres; Euler decoding is explicit. Calibration
requires observations with consistent physical axes, not per-image axis naming.
"""

from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from vive_correct.config import batch_settings
from vive_correct.data.layout import data_directory

_CALIBRATION = batch_settings()["calibration"]


def transform(vector):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    matrix[:3, 3] = vector[3:]
    return matrix


def parameters(matrix):
    return np.r_[Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]


def inverse(matrix):
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


@dataclass
class PoseStream:
    times: np.ndarray
    matrices: np.ndarray
    invalid_times: np.ndarray = field(default_factory=lambda: np.empty(0))

    @classmethod
    def load(
        cls, path, sequence=_CALIBRATION["euler_sequence"], degrees=_CALIBRATION["euler_degrees"]
    ):
        table = pq.read_table(path, columns=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"])
        values = np.column_stack([table[key].to_numpy() for key in table.column_names]).astype(
            float
        )
        if (
            len(values) < 2
            or not np.isfinite(values[:, 0]).all()
            or np.any(np.diff(values[:, 0]) < 0)
        ):
            raise ValueError(f"Invalid or nonmonotonic pose stream: {path}")
        times, counts = np.unique(values[:, 0], return_counts=True)
        valid = np.isfinite(values[:, 1:]).all(axis=1) & ~np.isin(values[:, 0], times[counts > 1])
        # Conflicting samples at the same timestamp are not silently averaged.
        invalid_times = np.unique(values[~valid, 0])
        values = values[valid]
        if len(values) < 2:
            raise ValueError(f"Insufficient finite tracked poses: {path}")
        matrices = np.repeat(np.eye(4)[None], len(values), axis=0)
        matrices[:, :3, 3] = values[:, 1:4]
        matrices[:, :3, :3] = Rotation.from_euler(
            sequence, values[:, 4:7], degrees=degrees
        ).as_matrix()
        return cls(values[:, 0], matrices, invalid_times)

    def at(self, time, max_gap_s=_CALIBRATION["max_sync_gap_s"]):
        if not np.isfinite(time) or time < self.times[0] or time > self.times[-1]:
            raise ValueError("Pose interpolation would extrapolate")
        right = int(np.searchsorted(self.times, time))
        if right < len(self.times) and self.times[right] == time:
            return self.matrices[right].copy()
        left = right - 1
        if self.times[right] - self.times[left] > max_gap_s or np.any(
            (self.invalid_times > self.times[left]) & (self.invalid_times < self.times[right])
        ):
            raise ValueError("Pose interpolation crosses a tracking gap")
        weight = (time - self.times[left]) / (self.times[right] - self.times[left])
        matrix = np.eye(4)
        matrix[:3, 3] = (1 - weight) * self.matrices[left, :3, 3] + weight * self.matrices[
            right, :3, 3
        ]
        matrix[:3, :3] = Slerp(
            self.times[[left, right]], Rotation.from_matrix(self.matrices[[left, right], :3, :3])
        )([time]).as_matrix()[0]
        return matrix


def load_streams(
    episode, sequence=_CALIBRATION["euler_sequence"], degrees=_CALIBRATION["euler_degrees"]
):
    root = data_directory(episode)
    return {
        side: PoseStream.load(root / f"observation.{key}.parquet", sequence, degrees)
        for side, key in [
            ("head", "head_pose"),
            ("left", "left_wrist_pose"),
            ("right", "right_wrist_pose"),
        ]
    }


def camera_wrist(world_head, world_wrist, head_camera, wrist_reference):
    return inverse(world_head @ head_camera) @ world_wrist @ wrist_reference


def fit_mount(observations, initial_head_camera, *, fit_camera=True):
    """Fit on training observations and assess independent held-out observations.

    Each observation includes side, world_head, world_wrist, camera_reference,
    split ('train'/'validation') and physical_axes_verified. A single-frame fit,
    or a fit with unobservable parameters, cannot authorize propagation.
    """
    if not observations or not all(o.get("physical_axes_verified") for o in observations):
        raise ValueError("Consistent physical axes must be verified before mounting calibration")
    sides = sorted({o["side"] for o in observations})
    if not set(sides) <= {"left", "right"}:
        raise ValueError("Unknown wrist side")
    train = [o for o in observations if o["split"] == "train"]
    validation = [o for o in observations if o["split"] == "validation"]
    for side in sides:
        if (
            sum(o["side"] == side for o in train) < _CALIBRATION["min_train_observations"]
            or sum(o["side"] == side for o in validation) < _CALIBRATION["min_holdout_observations"]
        ):
            raise ValueError(
                "Each wrist needs at least "
                f"{_CALIBRATION['min_train_observations']} training and "
                f"{_CALIBRATION['min_holdout_observations']} held-out observations"
            )
    initial_head_camera = np.asarray(initial_head_camera, float)
    initial_mounts = {}
    for side in sides:
        row = next(o for o in train if o["side"] == side)
        initial_mounts[side] = (
            inverse(np.asarray(row["world_wrist"]))
            @ np.asarray(row["world_head"])
            @ initial_head_camera
            @ np.asarray(row["camera_reference"])
        )
    vectors = ([parameters(initial_head_camera)] if fit_camera else []) + [
        parameters(initial_mounts[s]) for s in sides
    ]

    def unpack(values):
        offset = 6 if fit_camera else 0
        camera = transform(values[:6]) if fit_camera else initial_head_camera
        mounts = {
            side: transform(values[offset + 6 * i : offset + 6 * (i + 1)])
            for i, side in enumerate(sides)
        }
        return camera, mounts

    def residual(values):
        camera, mounts = unpack(values)
        errors = []
        for row in train:
            predicted = camera_wrist(
                np.asarray(row["world_head"]),
                np.asarray(row["world_wrist"]),
                camera,
                mounts[row["side"]],
            )
            delta = inverse(np.asarray(row["camera_reference"])) @ predicted
            errors.extend(
                np.r_[Rotation.from_matrix(delta[:3, :3]).as_rotvec() / 0.05, delta[:3, 3] / 0.01]
            )
        return np.asarray(errors)

    fit = least_squares(residual, np.concatenate(vectors), loss="soft_l1", max_nfev=500)
    camera, mounts = unpack(fit.x)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-6))
    condition = float(singular[0] / max(singular[-1], 1e-15))
    rows = []
    for row in validation:
        predicted = camera_wrist(
            np.asarray(row["world_head"]),
            np.asarray(row["world_wrist"]),
            camera,
            mounts[row["side"]],
        )
        delta = inverse(np.asarray(row["camera_reference"])) @ predicted
        rows.append(
            {
                "side": row["side"],
                "translation_error_m": float(np.linalg.norm(delta[:3, 3])),
                "rotation_error_deg": float(
                    np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
                ),
            }
        )
    # Generous installation bounds, not measured device dimensions. Short tracks
    # can fit very small residuals with physically implausible mounting offsets.
    mounting_distances = {
        side: float(np.linalg.norm(matrix[:3, 3])) for side, matrix in mounts.items()
    }
    camera_distance = float(np.linalg.norm(camera[:3, 3]))
    plausible = camera_distance <= _CALIBRATION["max_head_camera_offset_m"] and all(
        d <= _CALIBRATION["max_wrist_label_offset_m"] for d in mounting_distances.values()
    )
    accepted = bool(
        fit.success
        and rank == len(fit.x)
        and condition < _CALIBRATION["max_jacobian_condition"]
        and plausible
        and all(
            r["translation_error_m"] < _CALIBRATION["holdout_translation_m"]
            and r["rotation_error_deg"] < _CALIBRATION["holdout_rotation_deg"]
            for r in rows
        )
    )
    return {
        "accepted": accepted,
        "status": "held_out_consistent" if accepted else "calibration_not_validated",
        "head_camera": camera.tolist(),
        "wrist_reference": {s: t.tolist() for s, t in mounts.items()},
        "fit_camera": fit_camera,
        "parameter_count": len(fit.x),
        "jacobian_rank": rank,
        "jacobian_condition": condition,
        "validation": rows,
        "installation_plausible": plausible,
        "mounting_distances_m": mounting_distances,
        "camera_distance_m": camera_distance,
        "installation_limits_m": {
            "head_camera": _CALIBRATION["max_head_camera_offset_m"],
            "wrist_label": _CALIBRATION["max_wrist_label_offset_m"],
        },
        "limits_note": "Conservative engineering screening bounds; not measured installation dimensions.",
        "note": "Consistency with visual observations is not independent metric ground truth.",
    }


def propagate(streams, times, calibration):
    if not calibration.get("accepted"):
        raise ValueError("Unvalidated mounting calibration cannot authorize propagation")
    if set(calibration["wrist_reference"]) != {"left", "right"}:
        raise ValueError("Both wrists need their own mounting calibration")
    camera = np.asarray(calibration["head_camera"])
    rows = []
    for time in times:
        row = {"timestamp_s": float(time), "wrists": {}, "errors": {}}
        try:
            head = streams["head"].at(time)
        except ValueError as error:
            row["errors"]["head"] = str(error)
            rows.append(row)
            continue
        for side in ("left", "right"):
            try:
                row["wrists"][side] = camera_wrist(
                    head,
                    streams[side].at(time),
                    camera,
                    np.asarray(calibration["wrist_reference"][side]),
                ).tolist()
            except ValueError as error:
                row["errors"][side] = str(error)
        rows.append(row)
    return rows
