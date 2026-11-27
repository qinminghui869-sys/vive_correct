import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import yaml

from ego_data.corrected_rgb import export_corrected_rgb
from vive_correct.batch.hand_overlay import JOINTS
from vive_correct.estimate.geometry import VIDEO_KEY


class CorrectedRGBTests(unittest.TestCase):
    def test_reviewed_rigid_wrists_and_hands_are_logged_without_second_correction(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "双手肌电 + MANUS"
            data = root / "data/chunk-000/episode_000001"
            data.mkdir(parents=True)
            (data.parent / "episode_000000").mkdir()
            reviewed = root / "reviewed"
            reviewed.mkdir()
            times = [0.0, 0.03]
            pq.write_table(pa.table({"timestamp": times}), data / f"{VIDEO_KEY}.parquet")
            for side in ("left", "right"):
                original = dict(timestamp=times)
                original.update({k: [0.0, 0.0] for k in ("x", "y", "z", "roll", "pitch", "yaw")})
                original["yaw"] = [0.0, 90.0]
                corrected = dict(original, x=[0.1, 0.0], y=[0.0, 0.1], yaw=[30.0, 120.0])
                name = f"observation.{side}_wrist_pose.parquet"
                pq.write_table(pa.table(original), data / name)
                pq.write_table(pa.table(corrected), reviewed / name)
                for destination, point in ((data, [0.01, 0.02, 0.03]), (reviewed, [0.4, 0.5, 0.6])):
                    hand = dict(timestamp=times)
                    hand.update(
                        {f"{j}_{a}": [point[i]] * 2 for j in JOINTS for i, a in enumerate("xyz")}
                    )
                    pq.write_table(
                        pa.table(hand), destination / f"observation.hand_{side}_points.parquet"
                    )
            video = root / "videos/chunk-000" / VIDEO_KEY / f"{data.name}.mp4"
            video.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in times:
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            alignment = root / "alignment.json"
            alignment.write_text(
                json.dumps(
                    dict(
                        episode=str(data),
                        pose_euler_sequence="xyz",
                        pose_degrees=True,
                        wrists={
                            s: dict(model="wrist_local_rigid", offset_m=[0.1, 0, 0])
                            for s in ("left", "right")
                        },
                    )
                )
            )
            camera_file = root / "calibrations/chunk-000" / data.name / "rectify_params.yaml"
            camera_file.parent.mkdir(parents=True)
            camera_file.write_text(
                yaml.safe_dump(
                    dict(
                        cameras={
                            VIDEO_KEY: dict(
                                camera_model="pinhole",
                                distortion_model="none",
                                intrinsics=[20, 20, 32, 24],
                                resolution=[64, 48],
                            )
                        }
                    )
                )
            )
            head = dict(timestamp=times)
            head.update({k: [0.0, 0.0] for k in ("x", "y", "z", "roll", "pitch", "yaw")})
            pq.write_table(pa.table(head), data / "observation.head_pose.parquet")
            mount = dict(
                T_head_rectified_camera=np.eye(4).tolist(),
                head_euler_sequence="xyz",
                head_euler_degrees=True,
            )
            with patch("ego_data.corrected_rgb.rr.log", wraps=rr.log) as log:
                result = export_corrected_rgb(
                    data,
                    root / "reviewed.rrd",
                    alignment,
                    reviewed_data=reviewed,
                    camera_calibration=mount,
                )
            self.assertEqual(result["verified_hand_sides"], ["left", "right"])
            for side in ("left", "right"):
                wrist = [
                    c.args[1].positions.as_arrow_array().to_pylist()
                    for c in log.call_args_list
                    if c.args[0] == f"world/{side}/corrected"
                ]
                np.testing.assert_allclose(np.asarray(wrist)[:, 0], [[0.1, 0, 0], [0, 0.1, 0]])
                hand = [
                    c.args[1].positions.as_arrow_array().to_pylist()
                    for c in log.call_args_list
                    if c.args[0] == f"world/{side}/hand"
                ]
                np.testing.assert_allclose(hand, np.tile([0.4, 0.5, 0.6], (2, 21, 1)))
                projected = [
                    c.args[1].positions.as_arrow_array().to_pylist()
                    for c in log.call_args_list
                    if c.args[0] == f"rgb/hands/{side}" and isinstance(c.args[1], rr.Points2D)
                ]
                np.testing.assert_allclose(
                    projected, np.tile([32 + 20 * 0.4 / 0.6, 24 + 20 * 0.5 / 0.6], (2, 21, 1))
                )
                bones = [c for c in log.call_args_list if c.args[0] == f"world/{side}/bones"]
                self.assertEqual(len(bones), 2)

    def test_depthless_recording_keeps_correction_and_full_frame_coverage(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            episode = root / "episode"
            data = episode / "data/chunk-000/episode_000000"
            data.mkdir(parents=True)
            times = [0.0, 0.03, 0.06]
            pq.write_table(pa.table({"timestamp": times}), data / f"{VIDEO_KEY}.parquet")
            for side in ("left", "right"):
                values = dict(timestamp=times)
                values.update({k: [0.0] * 3 for k in ("x", "y", "z", "roll", "pitch", "yaw")})
                pq.write_table(pa.table(values), data / f"observation.{side}_wrist_pose.parquet")
            video = episode / "videos/chunk-000" / VIDEO_KEY / "episode_000000.mp4"
            video.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in times:
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            alignment = root / "alignment.json"
            alignment.write_text(
                json.dumps(
                    dict(
                        episode=str(episode),
                        pose_euler_sequence="xyz",
                        pose_degrees=True,
                        wrists={
                            s: dict(model="world", offset_m=[0.1, 0.2, 0.3])
                            for s in ("left", "right")
                        },
                    )
                )
            )
            output = root / "test.rrd"
            result = export_corrected_rgb(episode, output, alignment)
            self.assertEqual(result["frames"], 3)
            self.assertTrue(result["vive_correction_displayed"])
            self.assertEqual(result["mode"], "rectified_rgb_corrected_wrists")
            self.assertEqual(result["invalid_wrist_frames"], {"left": [], "right": []})
            self.assertGreater(output.stat().st_size, 100)
