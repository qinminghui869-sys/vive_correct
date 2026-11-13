"""Check the camera-to-world chain and reject mismatched origin reports."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego_data.vive_origin_alignment import build_alignment


class OriginAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.episode = root / "episode"
        data = self.episode / "data/chunk-000/episode_000000"
        data.mkdir(parents=True)
        for name, positions in [
            ("head", [1.0, 3.0]),
            ("left_wrist", [10.0, 30.0]),
            ("right_wrist", [30.0, 50.0]),
        ]:
            columns = dict(timestamp=[0.0, 0.02], x=positions)
            columns.update({name: [0.0, 0.0] for name in ("y", "z", "roll", "pitch", "yaw")})
            pq.write_table(pa.table(columns), data / f"observation.{name}_pose.parquet")
        key = "observation.images.headband_left_rgb_rectified"
        for name in (key, "observation.images.depth_headband_left_rectified"):
            pq.write_table(pa.table({"timestamp": [0.0, 0.01, 0.02]}), data / f"{name}.parquet")
        self.report = root / "report.json"
        self.payload = dict(
            episode=str(self.episode),
            video_key=key,
            observations=[
                dict(
                    frame_index=1,
                    timestamp_s=0.01,
                    stickers={s: {"origin_camera_m": [0.1, 0.2, 0.4]} for s in ("left", "right")},
                )
            ],
        )
        self.report.write_text(json.dumps(self.payload))
        mount = np.eye(4)
        mount[1, 3] = 1.0
        self.calibration = root / "mount.json"
        self.calibration.write_text(
            json.dumps(
                dict(
                    T_head_rectified_camera=mount.tolist(),
                    head_euler_sequence="xyz",
                    head_euler_degrees=True,
                )
            )
        )

    def build(self):
        return build_alignment(self.episode, self.report, self.calibration)

    def test_interpolated_origins_transform_to_world_in_meters(self):
        result = self.build()
        np.testing.assert_allclose(result["wrists"]["left"]["offset_m"], [-17.9, 1.2, 0.4])
        np.testing.assert_allclose(result["wrists"]["right"]["offset_m"], [-37.9, 1.2, 0.4])
        for sample in result["samples"]:
            np.testing.assert_allclose(sample["target_world_m"], [2.1, 1.2, 0.4])

    def test_other_episode_is_rejected(self):
        self.payload["episode"] = str(self.episode.parent / "other")
        self.report.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, "another episode"):
            self.build()

    def test_missing_hand_is_not_replaced_with_zero(self):
        del self.payload["observations"][0]["stickers"]["right"]
        self.report.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, "No repaired origin for right"):
            self.build()

    def test_timestamp_mismatch_is_rejected(self):
        self.payload["observations"][0]["timestamp_s"] = 0.015
        self.report.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.build()
