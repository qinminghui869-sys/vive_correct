"""Explicit, serializable camera input shared by scan, review and export."""

from pathlib import Path
import json

import cv2
import numpy as np

from vive_correct.batch.report_io import digest
from vive_correct.runtime_cache import cached_file

EYE_KEY = "observation.images.eye_rgb"
DEFAULT_EYE_CALIBRATION = (
    Path(__file__).resolve().parents[1] / "calibrations/neon_scene_camera_917093.json"
)


def from_args(args):
    if getattr(args, "coordinate_mode", "headband") != "eye-reference":
        return None
    path = Path(getattr(args, "eye_calibration", None) or DEFAULT_EYE_CALIBRATION).resolve()
    doc = json.loads(path.read_text())
    K = np.asarray(doc["camera_matrix"], float)
    dist = np.asarray(doc["dist_coefs"], float)
    size = doc["resolution"]
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or not np.allclose(K[2], [0, 0, 1])
        or dist.ndim != 1
        or dist.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(dist).all()
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
    ):
        raise ValueError("Invalid Neon camera calibration")
    return dict(
        coordinate_mode="eye-reference",
        reference_model="eye_origin_standing_axes_v1",
        video_key=EYE_KEY,
        calibration_path=str(path),
        calibration_sha256=digest(path),
        K=K.tolist(),
        distortion=dist.tolist(),
        resolution=size,
        rectification="cv2.initUndistortRectifyMap; R=I; new_K=K; original resolution",
        offset_local_m=[0.0, -0.085, -0.058],
    )


def verify(context):
    if context and digest(Path(context["calibration_path"])) != context["calibration_sha256"]:
        raise ValueError("眼动标定已变化，请使用新输出目录重新选帧")


def video_suffix(context=None):
    from vive_correct.dataset import VIDEO_SUFFIX

    return Path("videos/chunk-000") / EYE_KEY / "episode_000000.mp4" if context else VIDEO_SUFFIX


def timestamp_suffix(context=None):
    from vive_correct.dataset import TIMESTAMP_SUFFIX

    return (
        Path("data/chunk-000/episode_000000") / f"{EYE_KEY}.parquet"
        if context
        else TIMESTAMP_SUFFIX
    )


def camera(context):
    from vive_correct.estimate.geometry import Camera

    verify(context)
    return Camera(
        np.asarray(context["K"], float), tuple(context["resolution"]), context["calibration_path"]
    )


def rectify(image, context=None):
    if not context:
        return image
    cam = camera(context)
    if image.shape[1::-1] != cam.resolution:
        raise ValueError("眼动视频分辨率与标定不一致")

    def maps():
        return cv2.initUndistortRectifyMap(
            cam.K, np.asarray(context["distortion"]), None, cam.K, cam.resolution, cv2.CV_32FC1
        )

    mapping = cached_file(
        ("eye_maps", json.dumps(context, sort_keys=True)), Path(context["calibration_path"]), maps
    )
    return cv2.remap(image, *mapping, interpolation=cv2.INTER_LINEAR)


def effective_calibration(store, index, legacy):
    context = store.context(index)
    if context:
        verify(context)
        return context
    return json.loads(Path(legacy).read_text())
