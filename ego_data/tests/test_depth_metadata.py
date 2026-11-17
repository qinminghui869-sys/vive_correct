import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego_data.depth_metadata import METADATA_NAMES, load_metadata, metadata_path
from ego_data.depth_targets import DEPTH_KEY, refine_targets
from ego_data.manus_view import depth_available


class DepthMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.episode = Path(self.temp.name)
        self.ann = self.episode / "annotation/chunk-000" / DEPTH_KEY / "episode_000000"
        self.ann.mkdir(parents=True)
        self.k = dict(fx=100, fy=100, cx=50, cy=40)
        self.meta = dict(intrinsics=dict(left=self.k), max_depth_m=4)

    def test_both_names_enable_depth_but_other_assets_are_still_required(self):
        required = [
            self.ann / "valid_mask_ffv1.mkv",
            self.episode / "videos/chunk-000" / DEPTH_KEY / "episode_000000.mp4",
            self.episode
            / "videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000000.mp4",
            self.episode / "data/chunk-000/episode_000000" / f"{DEPTH_KEY}.parquet",
        ]
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.assertFalse(depth_available(self.episode))
        with self.assertRaisesRegex(FileNotFoundError, "Depth metadata missing"):
            load_metadata(self.ann)
        for name in METADATA_NAMES:
            with self.subTest(name=name):
                path = self.ann / name
                path.write_text(json.dumps(self.meta))
                self.assertEqual(load_metadata(self.ann), self.meta)
                self.assertTrue(depth_available(self.episode))
                for asset in required:
                    asset.unlink()
                    self.assertFalse(depth_available(self.episode))
                    asset.touch()
                path.unlink()

    def test_precedence_and_invalid_json_do_not_silently_change_calibration(self):
        primary, alternate = [self.ann / name for name in METADATA_NAMES]
        alternate.write_text(json.dumps(self.meta))
        primary.write_text(json.dumps(dict(self.meta, max_depth_m=2)))
        self.assertEqual(metadata_path(self.ann), primary)
        self.assertEqual(load_metadata(self.ann)["max_depth_m"], 2)
        primary.write_text("invalid json")
        with self.assertRaises(json.JSONDecodeError):
            load_metadata(self.ann)

    def test_refinement_uses_both_names(self):
        data = self.episode / "data/chunk-000/episode_000000"
        data.mkdir(parents=True)
        pq.write_table(pa.table({"timestamp": [1.0]}), data / f"{DEPTH_KEY}.parquet")
        report = dict(
            camera=dict(K=[[100, 0, 50], [0, 100, 40], [0, 0, 1]]),
            observations=[
                dict(
                    frame_index=0,
                    timestamp_s=1.0,
                    stickers=dict(
                        left=dict(
                            origin_camera_m=[0, 0, 0.8],
                            image_corners_px=[[30, 20], [70, 20], [70, 60], [30, 60]],
                        )
                    ),
                )
            ],
        )
        for name in METADATA_NAMES:
            with self.subTest(name=name):
                path = self.ann / name
                path.write_text(json.dumps(self.meta))
                containers = []
                for value in (400, 255):
                    frame = MagicMock()
                    frame.to_ndarray.return_value = np.full((80, 100), value, np.uint16)
                    container = MagicMock()
                    container.__enter__.return_value.decode.return_value = [frame]
                    containers.append(container)
                with patch("ego_data.depth_targets.av.open", side_effect=containers):
                    result = refine_targets(self.episode, report)[(0, "left")]
                self.assertEqual(result["status"], "depth_plane_refined")
                np.testing.assert_allclose(result["origin_camera_m"], [0, 0, 0.4])
                path.unlink()
