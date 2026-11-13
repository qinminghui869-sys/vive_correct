"""Apply explicit local-axis reviews without changing the observed tag plane."""

from copy import deepcopy

import numpy as np

from vive_correct.estimate.geometry import final_axis_angles


def reorient_pose(pose, old_from_new, camera):
    """R_new = R_old @ old_from_new; keep camera-space corners unchanged."""
    basis = np.asarray(old_from_new, dtype=float)
    if (
        basis.shape != (3, 3)
        or not np.isfinite(basis).all()
        or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(basis), 1, atol=1e-8)
    ):
        raise ValueError("Axis review must be a proper right-handed rotation")
    result = deepcopy(pose)
    rotation = np.asarray(pose["R_object_to_camera"]) @ basis
    transform = np.asarray(pose["T_camera_local"]).copy()
    transform[:3, :3] = rotation
    label_transform = np.asarray(pose["T_label_local"]).copy()
    label_transform[:3, :3] = label_transform[:3, :3] @ basis
    result.update(
        R_object_to_camera=rotation.tolist(),
        T_camera_local=transform.tolist(),
        T_label_local=label_transform.tolist(),
        object_corners_local_m=(np.asarray(pose["object_corners_local_m"]) @ basis).tolist(),
        axis_direction_deg=dict(
            zip(
                ("x_from_left", "y_from_up", "z_from_forward"),
                final_axis_angles(rotation, np.asarray(pose["t_camera_m"]), camera),
            )
        ),
    )
    return result


def normalize_reference_axes(pose, camera, source_convention):
    """Convert the known reference convention once; never guess an unknown basis."""
    from vive_correct.estimate.geometry import LOCAL_AXES_CONVENTION

    if pose.get("local_axes_convention") == LOCAL_AXES_CONVENTION:
        return deepcopy(pose)
    if pose.get("axes_generated") is False:
        return dict(deepcopy(pose), local_axes_convention=LOCAL_AXES_CONVENTION)
    if "Y=-Zref" not in source_convention or "Z=Yref" not in source_convention:
        raise ValueError("Baseline local-axis convention is unknown")
    basis = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    result = reorient_pose(pose, basis, camera)
    result.update(
        local_axes_convention=LOCAL_AXES_CONVENTION,
        axis_convention_change={"old_from_new": basis.tolist(), "source": source_convention},
    )
    return result
