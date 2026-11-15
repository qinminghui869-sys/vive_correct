"""Fit and validate a rigid hand-mount correction from independent correspondences."""

import argparse
import json
from pathlib import Path

import numpy as np


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
    ):
        raise ValueError(
            "Hand correction must be a finite rigid transform without scale/reflection"
        )
    return matrix


def fit_rigid(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 6
        or not np.isfinite([source, target]).all()
    ):
        raise ValueError("Need at least six finite 3D point correspondences")
    a, b = source - source.mean(0), target - target.mean(0)
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Collinear correspondences cannot determine a hand rotation")
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ sign @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = target.mean(0) - rotation @ source.mean(0)
    return rigid_matrix(matrix)


def fit_calibration(document, max_error_m=0.02):
    """Hold out whole frames: fitting residual alone never certifies a calibration."""
    if not np.isfinite(max_error_m) or not 0 < max_error_m <= 0.05:
        raise ValueError("Validation threshold must be in (0, 0.05] meters")
    if not document.get("reference_method"):
        raise ValueError("Describe independent anatomical reference points in reference_method")
    result = {
        k: document[k]
        for k in ("episode", "camera_calibration", "wrist_calibration_sha256", "reference_method")
    }
    result.update(
        version=1,
        accepted=False,
        hands={},
        max_error_m=max_error_m,
        convention="tracker-local rigid correction before Vive translation",
    )
    for side, rows in document["hands"].items():
        if side not in ("left", "right"):
            raise ValueError("Unknown hand side")
        train = [r for r in rows if r["split"] == "train"]
        valid = [r for r in rows if r["split"] == "validation"]
        if len(train) + len(valid) != len(rows):
            raise ValueError("Each correspondence needs train or validation split")
        train_frames, valid_frames = {r["frame"] for r in train}, {r["frame"] for r in valid}
        if (
            len(train_frames) < 3
            or len(valid_frames) < 2
            or train_frames & valid_frames
            or len(valid) < 6
        ):
            raise ValueError(
                "Use >=3 training and >=2 disjoint validation frames, >=6 validation points"
            )
        src = np.asarray([r["source_tracker_m"] for r in train])
        dst = np.asarray([r["target_tracker_m"] for r in train])
        matrix = fit_rigid(src, dst)
        errors = {}
        for name, group in (("train", train), ("validation", valid)):
            source = np.asarray([r["source_tracker_m"] for r in group], float)
            target = np.asarray([r["target_tracker_m"] for r in group], float)
            if (
                source.shape != target.shape
                or source.shape != (len(group), 3)
                or not np.isfinite([source, target]).all()
            ):
                raise ValueError("Invalid correspondence coordinates")
            distance = np.linalg.norm(source @ matrix[:3, :3].T + matrix[:3, 3] - target, axis=1)
            errors[name] = dict(
                rms_m=float(np.sqrt(np.mean(distance**2))),
                max_m=float(distance.max()),
                points=len(group),
            )
        result["hands"][side] = dict(
            matrix=matrix.tolist(),
            errors=errors,
            accepted=errors["validation"]["max_m"] <= max_error_m,
            training_frames=sorted(train_frames),
            validation_frames=sorted(valid_frames),
        )
    result["accepted"] = bool(result["hands"]) and all(
        h["accepted"] for h in result["hands"].values()
    )
    return result


def apply_mount(points, wrist_matrix, correction):
    """Correct in tracker coordinates, retaining the tracked orientation and bone lengths."""
    matrix = rigid_matrix(correction)
    rotation, translation = wrist_matrix[:3, :3], wrist_matrix[:3, 3]
    local = (np.asarray(points) - translation) @ rotation
    local = local @ matrix[:3, :3].T + matrix[:3, 3]
    return local @ rotation.T + translation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-error-mm", type=float, default=20)
    args = parser.parse_args()
    result = fit_calibration(json.loads(args.samples.read_text()), args.max_error_mm / 1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Calibration accepted: {result['accepted']}")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
