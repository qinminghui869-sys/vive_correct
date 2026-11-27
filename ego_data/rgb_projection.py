"""Project world hand joints onto rectified RGB using the renderer's head convention."""

import json
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from ego_data.tracking_sync import resample_tracking
from ego_data.vive_origin_alignment import pose_stream
from vive_correct.batch.hand_overlay import CHAINS, HandStream, project
from vive_correct.data.layout import calibration_file, data_directory
from vive_correct.estimate.geometry import Camera


class RGBProjection:
    def __init__(self, episode, times, calibration=None, *, rectified=True):
        self.reason = None
        self.frames = []
        self.streams = {}
        self.camera = None
        try:
            if not rectified:
                raise ValueError("Projection requires rectified left RGB")
            config = (
                json.loads(Path(calibration).read_text())
                if isinstance(calibration, (str, Path))
                else calibration or {}
            )
            if "T_head_rectified_camera" not in config:
                raise ValueError("Head-to-camera extrinsic unavailable")
            self.mount = np.asarray(config["T_head_rectified_camera"], float)
            if self.mount.shape != (4, 4) or not np.isfinite(self.mount).all():
                raise ValueError("Invalid head-to-camera extrinsic")
            self.camera = Camera.load(calibration_file(episode))
            self.sequence = config["head_euler_sequence"]
            self.degrees = config["head_euler_degrees"]
            self.heads = resample_tracking(
                pose_stream(data_directory(episode) / "observation.head_pose.parquet"),
                times,
                pose=True,
                sequence=self.sequence,
                degrees=self.degrees,
            )[1]
        except (OSError, ValueError, KeyError) as error:
            self.reason = str(error)

    def log(self, side, points, index, shape, color):
        path = f"rgb/hands/{side}"
        rr.log(path, rr.Clear(recursive=True))
        record = dict(frame=index, side=side, visible_joints=0)
        try:
            if self.reason:
                raise ValueError(self.reason)
            if tuple(shape[:2][::-1]) != self.camera.resolution:
                raise ValueError("RGB resolution differs from camera calibration")
            head = self.heads[index]
            if not np.isfinite(head).all():
                raise ValueError("Synchronized head pose unavailable")
            world_head = np.eye(4)
            world_head[:3, :3] = Rotation.from_euler(
                self.sequence, head[3:], degrees=self.degrees
            ).as_matrix()
            world_head[:3, 3] = head[:3]
            pixels, valid = project(
                points, np.linalg.inv(world_head @ self.mount), self.camera.K, shape
            )
            if valid.any():
                rr.log(path, rr.Points2D(pixels[valid], colors=color, radii=3))
            segments = [
                pixels[[a, b]]
                for chain in CHAINS
                for a, b in zip(chain, chain[1:])
                if valid[a] and valid[b]
            ]
            if segments:
                rr.log(path + "/bones", rr.LineStrips2D(segments, colors=color, radii=1.5))
            record.update(status="projected", visible_joints=int(valid.sum()))
        except ValueError as error:
            record.update(status="unavailable", reason=str(error))
        self.frames.append(record)

    def load_hands(self, directory):
        self.errors = {}
        for side in ("left", "right"):
            try:
                self.streams[side] = HandStream(
                    Path(directory) / f"observation.hand_{side}_points.parquet"
                )
            except (OSError, ValueError, KeyError) as error:
                self.errors[side] = str(error)

    def log_source(self, index, timestamp, shape):
        for side, color in (("left", [30, 220, 255]), ("right", [255, 100, 180])):
            try:
                if side not in self.streams:
                    raise ValueError(self.errors[side])
                points, _ = self.streams[side].at(timestamp)
            except ValueError as error:
                rr.log(f"rgb/hands/{side}", rr.Clear(recursive=True))
                self.frames.append(
                    dict(frame=index, side=side, status="unavailable", reason=str(error))
                )
                continue
            self.log(side, points, index, shape, color)

    def report(self):
        return dict(
            status="unavailable" if self.reason else "camera_mount_projection_preview",
            reason=self.reason,
            frames=self.frames,
        )
