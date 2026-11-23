"""Source wrist axes, reviewed local rotations and safe image projections."""

import numpy as np
import warnings
from scipy.spatial.transform import Rotation

CONVENTION = "intrinsic_XYZ_degrees"
AXIS_LENGTH_M = 0.04


def axis_angles(choice):
    values = np.asarray(choice.get("axis_angles_deg", [0, 0, 0]), dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("XYZ 旋转角度必须为三个有限数值")
    if choice.get("axis_convention", CONVENTION) != CONVENTION:
        raise ValueError("不支持的坐标轴旋转约定")
    return values


def local_rotation(choice):
    return Rotation.from_euler("XYZ", axis_angles(choice), degrees=True).as_matrix()


def corner_correction(pose, camera, convention="label_z_normal"):
    """Use the same PnP plane as the center; resolve rectangle signs by image axes."""
    from vive_correct.estimate.geometry import local_basis, LOCAL_AXES_CONVENTION

    rotation = np.asarray(pose["rotation_camera_label"])
    position = np.asarray(pose["pnp_origin_camera_m"])
    target = rotation @ local_basis(rotation, position, camera, require_direction_limits=False)
    if convention not in ("label_z_normal", "wrist_y_normal"):
        raise ValueError(f"不支持的 PnP 轴定义: {convention}")
    conversion = np.eye(3)
    if convention == "wrist_y_normal":
        conversion = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    target = target @ conversion
    source = np.asarray(pose["calibration_sample"]["rotation_camera_source"])
    # At gimbal lock SciPy chooses a zero third angle, preserving the rotation.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)
        angles = Rotation.from_matrix(source.T @ target).as_euler("XYZ", degrees=True)
    return dict(
        axis_angles_deg=angles.tolist(),
        axis_convention=CONVENTION,
        rotation_camera_target=target.tolist(),
        geometry_convention=(
            LOCAL_AXES_CONVENTION
            if convention == "label_z_normal"
            else "X_left_Y_negative_label_normal_Z_tangent"
        ),
        pnp_axis_convention=convention,
        baseline_local_rotation=conversion.tolist(),
        physical_axes_verified=False,
        plane_ambiguous=pose.get("plane_ambiguous", False),
    )


def prepare_pose(pose, camera, choice):
    """Anchor the PnP depth on the measured quadrilateral's center ray."""
    pose["pnp_origin_camera_m"] = pose["origin_camera_m"]
    ray = np.linalg.solve(camera.K, [*pose["label_center_px"], 1.0])
    pose.update(
        origin_camera_m=(ray * (pose["origin_camera_m"][2] / ray[2])).tolist(),
        axis_angles_deg=axis_angles(choice).tolist(),
        axis_convention=CONVENTION,
    )


def project_axes(origin, rotation, K):
    """Hide axes crossing the camera plane; clip remaining lines in the renderer."""
    origin, rotation, K = map(np.asarray, (origin, rotation, K))

    def project(p):
        if not np.isfinite(p).all() or p[2] <= 1e-4:
            return None
        uv = K @ p
        return (uv[:2] / uv[2]).tolist()

    return dict(
        origin=project(origin),
        endpoints=[project(origin + AXIS_LENGTH_M * rotation[:, i]) for i in range(3)],
    )


def enrich_pose(pose, sample, K):
    rotation = np.asarray(sample["rotation_camera_corrected"])
    origin = np.asarray(sample["origin_camera_m"])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, origin
    pose.update(
        origin_camera_m=origin.tolist(),
        T_camera_local=transform.tolist(),
        axes_generated=True,
        axes_failure_reason=None,
        local_axes_convention="source_wrist_with_local_calibration",
        orientation_status="human_axis_review",
        physical_axes_verified=False,
        axis_projection=project_axes(origin, rotation, K),
        axis_length_m=AXIS_LENGTH_M,
        camera_K=np.asarray(K).tolist(),
        calibration_sample=sample,
    )
    return pose


def preview(episode, side, choice, calibration):
    from pathlib import Path
    import pyarrow.parquet as pq
    from ego_data.vive_origin_alignment import build_alignment
    from vive_correct.dataset import TIMESTAMP_SUFFIX
    from vive_correct.estimate.geometry import VIDEO_KEY
    from vive_correct.human_review import checked_pose

    if side not in ("left", "right"):
        raise ValueError("未知物理手侧")
    axis_angles(choice)
    pose, camera = checked_pose(Path(episode), choice, side=side)
    times = pq.read_table(Path(episode) / TIMESTAMP_SUFFIX)["timestamp"].to_numpy()
    frame = choice["frame_index"]
    if frame >= len(times) or not np.isfinite(times[frame]):
        raise ValueError("参考帧缺少有效时间戳")
    prepare_pose(pose, camera, choice)
    report = dict(
        episode=str(episode),
        video_key=VIDEO_KEY,
        camera=dict(K=camera.K.tolist(), resolution=list(camera.resolution)),
        observations=[
            dict(frame_index=frame, timestamp_s=float(times[frame]), stickers={side: pose})
        ],
    )
    alignment = build_alignment(
        Path(episode), report, calibration, refine_depth=True, require_bilateral=False
    )
    result = enrich_pose(pose, alignment["samples"][0], camera.K)
    try:
        from vive_correct.review_io import read_json

        convention = read_json(Path(calibration)).get("pnp_axis_convention", "label_z_normal")
        result["corner_correction"] = corner_correction(result, camera, convention)
        result["corner_correction_error"] = None
    except ValueError as error:
        result["corner_correction"] = None
        result["corner_correction_error"] = str(error)
    return result
