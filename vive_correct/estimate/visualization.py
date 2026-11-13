"""Render axes with the same transforms exported in the pose report."""

from pathlib import Path

import cv2
import numpy as np

from vive_correct.config import estimate_settings

AXIS_LENGTH_M = estimate_settings()["geometry"]["axis_length_m"]
AXIS_COLORS_BGR = ((40, 40, 245), (30, 200, 30), (255, 125, 25))


def draw_poses(image, labels, camera, axis_length_m=AXIS_LENGTH_M):
    result = image.copy()
    axes = np.vstack([np.zeros(3), np.eye(3) * axis_length_m])
    for side, pose in labels.items():
        if pose.get("axes_generated") is False:
            t = np.asarray(pose["origin_camera_m"])
            if t[2] <= 0:
                continue
            origin = tuple(np.round(camera.project(np.zeros((1, 3)), np.eye(3), t)[0]).astype(int))
            cv2.circle(result, origin, 5, (255, 255, 255), -1)
            cv2.putText(
                result,
                f"O_{side[0].upper()} axes unavailable",
                (origin[0] + 8, origin[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            continue
        if "T_camera_local" in pose:
            transform = np.asarray(pose["T_camera_local"], dtype=float)
            R, t = transform[:3, :3], transform[:3, 3]
        else:
            R, t = np.array(pose["R_object_to_camera"]), np.array(pose["t_camera_m"])
        uv = camera.project(axes, R, t)
        if np.any((axes @ R.T + t)[:, 2] <= 0):
            continue
        corners = np.round(pose["image_corners_px"]).astype(int)
        cv2.polylines(result, [corners], True, (20, 220, 245), 2, cv2.LINE_AA)
        origin = tuple(np.round(uv[0]).astype(int))
        for i, color in enumerate(AXIS_COLORS_BGR, 1):
            end = tuple(np.round(uv[i]).astype(int))
            cv2.arrowedLine(result, origin, end, (20, 20, 20), 5, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(result, origin, end, color, 3, cv2.LINE_AA, tipLength=0.2)
            position = (end[0] + 5, end[1] - 5)
            cv2.putText(
                result,
                "XYZ"[i - 1],
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                result,
                "XYZ"[i - 1],
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.circle(result, origin, 5, (255, 255, 255), -1)
        cv2.putText(
            result,
            f"O_{side[0].upper()}",
            (origin[0] + 8, origin[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result


def save_image(path: Path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")
