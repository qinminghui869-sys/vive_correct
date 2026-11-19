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


def corner_correction(pose, camera):
    """Use the same PnP plane as the center; resolve rectangle signs by image axes."""
    from vive_correct.estimate.geometry import local_basis, LOCAL_AXES_CONVENTION

    rotation = np.asarray(pose["rotation_camera_label"])
    position = np.asarray(pose["pnp_origin_camera_m"])
    target = rotation @ local_basis(rotation, position, camera, require_direction_limits=False)
    source = np.asarray(pose["calibration_sample"]["rotation_camera_source"])
    # At gimbal lock SciPy chooses a zero third angle, preserving the rotation.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)
        angles = Rotation.from_matrix(source.T @ target).as_euler("XYZ", degrees=True)
    return dict(
        axis_angles_deg=angles.tolist(),
        axis_convention=CONVENTION,
        rotation_camera_target=target.tolist(),
        geometry_convention=LOCAL_AXES_CONVENTION,
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
        local_axes_convention="source_vive_camera_projection"
        if sample.get("orientation_policy") == "preserve_source"
        else "source_vive_shared_extrinsic"
        if sample.get("shared_extrinsic")
        else "reviewed_label_basis"
        if "T_reference_source" in sample
        else "source_wrist_with_local_calibration",
        orientation_status="human_axis_review",
        physical_axes_verified=False,
        axis_projection=project_axes(origin, rotation, K),
        axis_length_m=AXIS_LENGTH_M,
        camera_K=np.asarray(K).tolist(),
        calibration_sample=sample,
    )
    if sample.get("orientation_policy") == "preserve_source":
        pose["T_world_local"] = sample["reference_pose"]
        pose["orientation_policy"] = "preserve_source"
    return pose


def preview(episode, side, choice, calibration, context=None, selected=None):
    from pathlib import Path
    import pyarrow.parquet as pq
    from ego_data.vive_origin_alignment import build_alignment
    from vive_correct.dataset import TIMESTAMP_SUFFIX
    from vive_correct.estimate.geometry import VIDEO_KEY
    from vive_correct.human_review import checked_pose

    if side not in ("left", "right"):
        raise ValueError("未知物理手侧")
    axis_angles(choice)
    if context:
        from vive_correct.eye_reference import shared_preview

        choices = dict(selected or {})
        choices[side] = choice
        return shared_preview(episode, side, choices, context)
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
        result["corner_correction"] = corner_correction(result, camera)
        result["corner_correction_error"] = None
    except ValueError as error:
        result["corner_correction"] = None
        result["corner_correction_error"] = str(error)
    return result
