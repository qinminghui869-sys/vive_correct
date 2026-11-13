"""Select same-frame stable labels first; retain independently observed fallbacks."""

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from vive_correct.config import estimate_settings
from vive_correct.estimate.geometry import Camera, MIN_EDGE_PX, VIDEO_KEY, estimate_pose, order_quad
from vive_correct.estimate.label_detection import refine_corners

_SELECTION = estimate_settings()["selection"]


@dataclass(frozen=True)
class SelectionConfig:
    window: int = _SELECTION["window"]
    stride: int = _SELECTION["stride"]
    max_frames: int | None = _SELECTION["max_frames"]
    max_fb_px: float = _SELECTION["max_fb_px"]
    max_median_motion_px: float = _SELECTION["max_median_motion_px"]
    max_p95_motion_px: float = _SELECTION["max_p95_motion_px"]
    max_excursion_px: float = _SELECTION["max_excursion_px"]
    min_projected_edge_px: float = _SELECTION["min_projected_edge_px"]

    def __post_init__(self):
        if self.window < 3 or self.window % 2 != 1 or self.stride < 1:
            raise ValueError("Window must be odd and >=3; stride must be positive")
        if self.max_frames is not None and self.max_frames < self.window:
            raise ValueError("Frame limit must include a complete stability window")
        gates = (
            self.max_fb_px,
            self.max_median_motion_px,
            self.max_p95_motion_px,
            self.max_excursion_px,
        )
        if not np.isfinite(gates).all() or min(gates) <= 0:
            raise ValueError("Stability thresholds must be finite and positive")
        if not np.isfinite(self.min_projected_edge_px) or self.min_projected_edge_px < 8:
            raise ValueError("Projected edge threshold must be at least 8 pixels")


@dataclass(frozen=True)
class Episode:
    path: Path
    video: Path
    camera: Camera
    timestamps: np.ndarray

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        video = path / f"videos/chunk-000/{VIDEO_KEY}/episode_000000.mp4"
        timestamps = path / f"data/chunk-000/episode_000000/{VIDEO_KEY}.parquet"
        for file in (video, timestamps):
            if not file.is_file():
                raise FileNotFoundError(file)
        camera = Camera.load(path / "calibrations/rectify_params.yaml")
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video}")
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if size != camera.resolution:
            raise ValueError(
                f"Video resolution {size} != calibrated resolution {camera.resolution}"
            )
        times = (
            pq.read_table(timestamps, columns=["timestamp"])["timestamp"].to_numpy().astype(float)
        )
        if count != len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(
                "Video and timestamps must have equal counts and strictly increasing times"
            )
        return cls(path, video, camera, times)


def frame_windows(episode, config, frame_index=None):
    """Decode once with bounded buffers; frames and quads stay in native pixels."""
    half = config.window // 2
    limit = min(len(episode.timestamps), config.max_frames or len(episode.timestamps))
    if frame_index is not None and not half <= frame_index < limit - half:
        raise ValueError("Requested frame does not have a complete stability window")
    start = 0 if frame_index is None else frame_index - half
    stop = limit if frame_index is None else frame_index + half + 1
    gray_buffer = deque(maxlen=config.window)
    color_buffer = deque(maxlen=half + 1)
    cap = cv2.VideoCapture(str(episode.video))
    try:
        if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError("Cannot seek to requested frame")
        width, height = episode.camera.resolution
        for index in range(start, stop):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at frame {index}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_buffer.append(cv2.resize(gray, (width // 2, height // 2)))
            color_buffer.append(image)
            center = index - half
            if len(gray_buffer) == config.window and (
                frame_index is not None or center % config.stride == 0
            ):
                yield center, color_buffer[0], list(gray_buffer)
    finally:
        cap.release()


def measure_stability(frames, quad, resolution, config):
    if len(frames) != config.window:
        return {"passed": False, "reason": "incomplete_window"}
    center = config.window // 2
    scale = np.array([frames[0].shape[1], frames[0].shape[0]]) / resolution
    points = (np.asarray(quad) * scale).astype(np.float32).reshape(4, 1, 2)
    tracks = {}
    fb_max = excursion_max = 0.0
    # Check distant frames first; reject moving windows before the full LK pass.
    order = dict.fromkeys([0, len(frames) - 1, center, *range(len(frames))])
    for index in order:
        target, valid, _ = cv2.calcOpticalFlowPyrLK(
            frames[center],
            frames[index],
            points,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(3, 30, 0.01),
        )
        if target is None or not valid.all() or not np.isfinite(target).all():
            return {"passed": False, "reason": "tracking_failed"}
        back, valid_back, _ = cv2.calcOpticalFlowPyrLK(
            frames[index],
            frames[center],
            target,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(3, 30, 0.01),
        )
        if back is None or not valid_back.all() or not np.isfinite(back).all():
            return {"passed": False, "reason": "tracking_failed"}
        fb_max = max(
            fb_max, float(np.linalg.norm((back - points).reshape(4, 2) / scale, axis=1).max())
        )
        excursion_max = max(
            excursion_max,
            float(np.linalg.norm((target - points).reshape(4, 2) / scale, axis=1).max()),
        )
        if fb_max > config.max_fb_px or excursion_max > config.max_excursion_px:
            return {"passed": False, "reason": "tracking_or_excursion_gate"}
        tracks[index] = target.reshape(4, 2) / scale
    speed = np.linalg.norm(np.diff([tracks[i] for i in range(len(frames))], axis=0), axis=2)
    median, p95 = float(np.median(speed)), float(np.percentile(speed, 95))
    passed = median <= config.max_median_motion_px and p95 <= config.max_p95_motion_px
    return {
        "passed": passed,
        "reason": None if passed else "motion_gate",
        "median_motion_px_per_frame": median,
        "p95_motion_px_per_frame": p95,
        "fb_max_px": fb_max,
        "max_corner_excursion_px": excursion_max,
    }


def fit_candidate(image, candidate, camera, refine=True, min_edge_px=MIN_EDGE_PX):
    """Fit a visible quadrilateral, preserving native-pixel bounds and RMS gates."""
    q = order_quad(candidate, min_edge_px)
    if np.any(q < 2) or np.any(q > np.array(camera.resolution) - 3):
        raise ValueError("Corners touch image boundary; complete label required")
    try:
        pose = estimate_pose(q, camera, min_edge_px, allow_origin_only=True, device_fixed=True)
    except ValueError:
        if not refine:
            raise
        # A noisy threshold contour must get the same edge-refinement opportunity.
        pose = estimate_pose(
            refine_corners(image, q), camera, min_edge_px, allow_origin_only=True, device_fixed=True
        )
    if refine:
        try:
            refined = estimate_pose(
                refine_corners(image, q),
                camera,
                min_edge_px,
                allow_origin_only=True,
                device_fixed=True,
            )
            if refined["rms_px"] <= pose["rms_px"]:
                pose = refined
        except (ValueError, cv2.error):
            pass
    corners = np.array(pose["image_corners_px"])
    if np.any(corners < 2) or np.any(corners > np.array(camera.resolution) - 3):
        raise ValueError("Refined corners touch image boundary")
    return pose


def select_frames(episode, candidate_provider, config=None, frame_index=None, refine=True):
    """Provider returns {side: [quad,...]}; missing hands are never synthesized."""
    config = config or SelectionConfig()
    singles, rejected = {}, Counter()
    examined = 0
    for index, image, grays in frame_windows(episode, config, frame_index):
        examined += 1
        labels = {}
        for side, candidates in candidate_provider(image).items():
            if side not in {"left", "right"}:
                raise ValueError(f"Unknown wrist side: {side}")
            for candidate in candidates:
                try:
                    pose = fit_candidate(
                        image, candidate, episode.camera, refine, config.min_projected_edge_px
                    )
                    stability = measure_stability(
                        grays, pose["image_corners_px"], episode.camera.resolution, config
                    )
                    if not stability["passed"]:
                        rejected[stability["reason"]] += 1
                        continue
                    pose["stability"] = stability
                    if side not in labels or pose["rms_px"] < labels[side]["rms_px"]:
                        labels[side] = pose
                except (ValueError, cv2.error):
                    rejected["geometry_gate"] += 1
        if not labels:
            continue
        observation = {
            "frame_index": index,
            "timestamp_s": float(episode.timestamps[index]),
            "window": [index - config.window // 2, index + config.window // 2],
            "window_duration_s": float(
                episode.timestamps[index + config.window // 2]
                - episode.timestamps[index - config.window // 2]
            ),
            "labels": labels,
            "image": image,
        }
        if set(labels) == {"left", "right"}:
            return {
                "mode": "both_hands_same_frame",
                "observations": [observation],
                "examined_windows": examined,
                "rejected": dict(rejected),
            }
        for side in labels:
            if (
                side not in singles
                or labels[side]["rms_px"] < singles[side]["labels"][side]["rms_px"]
            ):
                singles[side] = observation
    observations = sorted(singles.values(), key=lambda obs: obs["frame_index"])
    mode = {0: "no_usable_label", 1: "single_hand", 2: "separate_frames"}[len(observations)]
    return {
        "mode": mode,
        "observations": observations,
        "examined_windows": examined,
        "rejected": dict(rejected),
    }
