"""Device-local axes come from a bound reference, never from image directions."""

from copy import deepcopy

import numpy as np

CONVENTION = "device_fixed_anchor"


def origin_only(pose, reason="installation_reference_missing"):
    result = deepcopy(pose)
    transform = result.get("T_camera_local")
    if result.get("origin_camera_m") is None and transform is not None:
        result["origin_camera_m"] = np.asarray(transform)[:3, 3].tolist()
    result.update(
        T_camera_local=None,
        axes_generated=False,
        axes_failure_reason=reason,
        local_axes_convention="unresolved",
        orientation_status="axes_not_generated",
        physical_axes_verified=False,
    )
    for key in ("R_object_to_camera", "T_label_local", "axis_direction_deg", "device_anchor"):
        result.pop(key, None)
    return result


def bind_anchor(pose, side, frame, source):
    """Keep a user-specified reference exactly; an unreviewed basis supplies only origin."""
    review = pose.get("axis_review", {})
    explicit = (
        pose.get("orientation_status") == "user_constraint_applied"
        and review.get("reviewer_type") == "user_direction_feedback"
        and review.get("input_direction_feedback_confirmed") is True
    )
    prior = pose.get("device_anchor", {})
    bound = (
        pose.get("local_axes_convention") == CONVENTION
        and prior.get("side") == side
        and prior.get("frame_index") == frame
    )
    if prior and not bound:
        return origin_only(pose, "reference_scope_mismatch")
    if not (explicit or bound) or pose.get("T_camera_local") is None:
        return origin_only(pose)
    # A display-basis conversion must not silently become a physical mounting reference.
    if pose.get("local_axes_convention") == "X_left_Y_up_Z_forward" and not bound:
        return origin_only(pose, "use_original_device_reference")
    result = deepcopy(pose)
    result.update(
        axes_generated=True,
        axes_failure_reason=None,
        local_axes_convention=CONVENTION,
        origin_camera_m=np.asarray(pose["T_camera_local"])[:3, 3].tolist(),
        device_anchor={
            "side": side,
            "frame_index": frame,
            "source_report": source["baseline_source"],
            "source_report_sha256": source["baseline_sha256"],
            "scope": "same_episode_side_contiguous_track",
        },
    )
    return result
