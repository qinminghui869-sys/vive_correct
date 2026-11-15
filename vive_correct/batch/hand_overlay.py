"""Project timestamp-matched hand joints; never invent a MANO surface or alignment."""

from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from vive_correct.batch.pose_alignment import PoseStream, inverse

JOINTS = (
    "wrist",
    "thumbCMC",
    "thumbMCP",
    "thumbIP",
    "thumbTip",
    "indexMCP",
    "indexPIP",
    "indexDIP",
    "indexTip",
    "middleMCP",
    "middlePIP",
    "middleDIP",
    "middleTip",
    "ringMCP",
    "ringPIP",
    "ringDIP",
    "ringTip",
    "pinkyMCP",
    "pinkyPIP",
    "pinkyDIP",
    "pinkyTip",
)
CHAINS = tuple((0, start, start + 1, start + 2, start + 3) for start in (1, 5, 9, 13, 17))
COLORS = {"left": (255, 220, 30), "right": (180, 70, 255)}


class HandStream:
    def __init__(self, path):
        columns = ["timestamp"] + [f"{joint}_{axis}" for joint in JOINTS for axis in "xyz"]
        table = pq.read_table(path, columns=columns)
        from ego_data.tracking_sync import normalize_tracking

        (self.times, values), self.timestamp_report = normalize_tracking(
            (
                table["timestamp"].to_numpy(),
                np.column_stack([table[c].to_numpy() for c in columns[1:]]),
            )
        )
        self.points = values.reshape(-1, 21, 3)

    def at(self, timestamp, tolerance=0.005):
        if (
            not len(self.times)
            or not np.isfinite(timestamp)
            or not self.times[0] <= timestamp <= self.times[-1]
        ):
            raise ValueError("Hand timestamp outside stream")
        right = int(np.searchsorted(self.times, timestamp))
        indices = [i for i in (right - 1, right) if 0 <= i < len(self.times)]
        index = min(indices, key=lambda i: abs(self.times[i] - timestamp))
        delta = float(self.times[index] - timestamp)
        if abs(delta) > tolerance:
            raise ValueError("No hand sample within 5 ms")
        return self.points[index], delta


def project(points, camera_world, intrinsic, shape):
    camera_points = points @ camera_world[:3, :3].T + camera_world[:3, 3]
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.01)
    pixels = np.full((len(points), 2), np.nan)
    projected = camera_points[valid] @ intrinsic.T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    height, width = shape[:2]
    valid &= (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return pixels, valid


class HandOverlay:
    def __init__(self, episode, calibration, camera):
        self.streams, self.errors = {}, {}
        self.head = None
        self.intrinsic = np.asarray(camera.get("K", []), float)
        accepted = calibration.get("accepted", False)
        mount = (
            calibration.get("fit", {}).get("head_camera")
            if accepted
            else calibration.get("initial_head_camera")
        )
        self.mount = np.asarray(mount, float) if mount is not None else None
        self.status = "calibrated_projection" if accepted else "unverified_preview"
        root = Path(episode) / "data/chunk-000/episode_000000"
        try:
            self.head = PoseStream.load(root / "observation.head_pose.parquet")
        except (OSError, ValueError, KeyError) as error:
            self.errors["head"] = str(error)
        for side in COLORS:
            try:
                self.streams[side] = HandStream(root / f"observation.hand_{side}_points.parquet")
            except (OSError, ValueError, KeyError) as error:
                self.errors[side] = str(error)

    def draw(self, image, timestamp):
        result = image.copy()
        record = {
            "timestamp_s": timestamp,
            "representation": "named_21_joint_skeleton_not_MANO_mesh",
            "hands": {},
        }
        try:
            if timestamp is None or not np.isfinite(timestamp):
                raise ValueError("Observation timestamp unavailable")
            if (
                self.head is None
                or self.mount is None
                or self.mount.shape != (4, 4)
                or not np.isfinite(self.mount).all()
                or self.intrinsic.shape != (3, 3)
                or not np.isfinite(self.intrinsic).all()
            ):
                raise ValueError("Head pose or camera mounting unavailable")
            camera_world = inverse(self.head.at(timestamp) @ self.mount)
        except ValueError as error:
            record.update(status="unavailable", reason=str(error))
            return result, record
        record["status"] = self.status
        for side, color in COLORS.items():
            try:
                if side not in self.streams:
                    raise ValueError(self.errors.get(side, "Hand stream unavailable"))
                points, delta = self.streams[side].at(timestamp)
                pixels, valid = project(points, camera_world, self.intrinsic, image.shape)
                for chain in CHAINS:
                    for a, b in zip(chain, chain[1:]):
                        if valid[a] and valid[b]:
                            cv2.line(
                                result,
                                tuple(np.rint(pixels[a]).astype(int)),
                                tuple(np.rint(pixels[b]).astype(int)),
                                color,
                                3,
                                cv2.LINE_AA,
                            )
                for i in np.flatnonzero(valid):
                    cv2.circle(
                        result, tuple(np.rint(pixels[i]).astype(int)), 5, color, -1, cv2.LINE_AA
                    )
                record["hands"][side] = {
                    "visible_joints": int(valid.sum()),
                    "sample_delta_s": delta,
                }
            except ValueError as error:
                record["hands"][side] = {"status": "unavailable", "reason": str(error)}
        return result, record
