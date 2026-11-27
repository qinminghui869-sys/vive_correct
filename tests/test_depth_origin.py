import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from test_human_review import make_fixture

from ego_data.depth_origin import backproject, measured_targets
from ego_data.vive_origin_alignment import build_alignment
from vive_correct.estimate.geometry import VIDEO_KEY
from vive_correct.runtime.runtime_cache import source_stamp


class DepthOriginTests(unittest.TestCase):
    def test_center_depth_not_pnp_or_world_z(self):
        with TemporaryDirectory() as folder:
            episode, _, calibration = make_fixture(Path(folder))
            mount = json.loads(calibration.read_text())
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("x", 60, degrees=True).as_matrix()
            transform[:3, 3] = [0, 0, 2]
            mount["T_head_rectified_camera"] = transform.tolist()
            calibration.write_text(json.dumps(mount))
            report = dict(
                episode=str(episode),
                video_key=VIDEO_KEY,
                camera=dict(K=[[600, 0, 320], [0, 600, 200], [0, 0, 1]], resolution=[640, 400]),
                observations=[
                    dict(
                        frame_index=0,
                        timestamp_s=0.0,
                        stickers={
                            s: dict(
                                label_center_px=[380, 260],
                                origin_camera_m=[999, 999, 999],
                                axis_angles_deg=[0, 0, 0],
                            )
                            for s in ["left", "right"]
                        },
                    )
                ],
            )
            del report["observations"][0]["stickers"]["left"]["origin_camera_m"]
            result = build_alignment(episode, report, calibration)
            for sample in result["samples"]:
                np.testing.assert_allclose(sample["origin_camera_m"], [0.05, 0.05, 0.5])
                np.testing.assert_allclose(
                    sample["target_world_m"],
                    transform[:3, :3] @ [0.05, 0.05, 0.5] + transform[:3, 3],
                )
                self.assertNotAlmostEqual(sample["target_world_m"][2], 0.5)
                self.assertEqual(
                    sample["depth_refinement"]["quality_status"], "unmasked_depth_preview"
                )
            report["observations"][0]["timestamp_s"] = 0.01
            with self.assertRaisesRegex(ValueError, "5 ms"):
                measured_targets(episode, report)
            before = source_stamp(episode)
            (episode / "meta/info.json").write_text("{}")
            self.assertNotEqual(source_stamp(episode), before)
            with self.assertRaisesRegex(ValueError, "PnP fallback"):
                measured_targets(episode, report)

    def test_single_pixel_validity_and_subpixel_ray(self):
        depth = np.full((3, 3), 900, np.uint16)
        depth[1, 1] = 600
        k = dict(fx=100, fy=100, cx=0, cy=0)
        point, pixel, raw = backproject(depth, [1.1, 1.2], k, 4)
        np.testing.assert_allclose(point, [0.0066, 0.0072, 0.6])
        self.assertEqual(pixel, [1, 1])
        self.assertEqual(raw, 600)
        with self.assertRaisesRegex(ValueError, "mask"):
            backproject(depth, [1, 1], k, 4, mask=np.zeros_like(depth))
        for value in (0, 5000):
            depth[1, 1] = value
            with self.assertRaisesRegex(ValueError, "invalid depth"):
                backproject(depth, [1, 1], k, 4)
        with self.assertRaisesRegex(ValueError, "outside"):
            backproject(depth, [-1, 1], k, 4)
