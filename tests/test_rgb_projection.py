from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ego_data.rgb_projection import RGBProjection
from vive_correct.estimate.geometry import VIDEO_KEY


class RGBProjectionTests(unittest.TestCase):
    def test_world_to_camera_clipping_and_missing_head_clear(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data/chunk-000/episode_000000"
            data.mkdir(parents=True)
            values = dict(
                timestamp=[0.0], x=[1.0], y=[0.0], z=[0.0], roll=[0.0], pitch=[0.0], yaw=[0.0]
            )
            pq.write_table(pa.table(values), data / "observation.head_pose.parquet")
            calibration = root / "calibrations/rectify_params.yaml"
            calibration.parent.mkdir()
            calibration.write_text(
                yaml.safe_dump(
                    dict(
                        cameras={
                            VIDEO_KEY: dict(
                                camera_model="pinhole",
                                distortion_model="none",
                                intrinsics=[100, 100, 32, 24],
                                resolution=[64, 48],
                            )
                        }
                    )
                )
            )
            config = dict(
                T_head_rectified_camera=np.eye(4).tolist(),
                head_euler_sequence="xyz",
                head_euler_degrees=True,
            )
            projection = RGBProjection(root, np.array([0.0, 1.0]), config)
            points = np.tile([1.0, 0.0, 1.0], (21, 1))
            points[1] = [1, 0, -1]
            points[2] = [10, 0, 1]
            with patch("ego_data.rgb_projection.rr.log") as log:
                projection.log("left", points, 0, (48, 64, 3), [255, 0, 0])
                emitted = log.call_args_list[1].args[1].positions.as_arrow_array().to_pylist()
                np.testing.assert_allclose(emitted, np.tile([32, 24], (19, 1)))
                log.reset_mock()
                projection.log("left", points, 1, (48, 64, 3), [255, 0, 0])
                self.assertEqual(log.call_count, 1)  # Clear stale projected joints.
            self.assertEqual(projection.frames[0]["visible_joints"], 19)
            self.assertEqual(projection.frames[1]["status"], "unavailable")
