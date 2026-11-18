import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import numpy as np
from ego_data.depth_targets import DEPTH_KEY, depth_frame, plane_center
from vive_correct.runtime_cache import CACHE


class DepthTargetTests(unittest.TestCase):
    def test_depth_cache_reuses_exact_pixels_and_invalidates_both_inputs(self):
        with TemporaryDirectory() as directory:
            episode = Path(directory)
            ann = episode / "annotation"
            ann.mkdir()
            video = episode / "videos/chunk-000" / DEPTH_KEY / "episode_000000.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            mask = ann / "valid_mask_ffv1.mkv"
            mask.touch()
            CACHE.clear()
            self.addCleanup(CACHE.clear)

            def open_video(path):
                container = MagicMock()
                frames = []
                for value in (100, 400):
                    frame = MagicMock()
                    frame.to_ndarray.return_value = np.full(
                        (8, 10), 255 if path == str(mask) else value, np.uint16
                    )
                    frames.append(frame)
                container.__enter__.return_value.decode.return_value = frames
                return container

            with patch("ego_data.depth_targets.av.open", side_effect=open_video) as opened:
                depth, valid = depth_frame(episode, ann, 1)
                np.testing.assert_array_equal(depth, np.full((8, 10), 0.4))
                np.testing.assert_array_equal(valid, np.full((8, 10), 255))
                depth[:] = 99
                self.assertEqual(depth_frame(episode, ann, 1)[0][0, 0], 0.4)
                self.assertEqual(opened.call_count, 2)
                for source in (video, mask):
                    source.write_bytes(b"changed")
                    depth_frame(episode, ann, 1)
                self.assertEqual(opened.call_count, 6)
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    depth_frame(episode, ann, 2)

    def test_refines_wrong_scale_without_changing_image_ray(self):
        depth = np.full((80, 100), 0.4)
        k = dict(fx=100, fy=100, cx=50, cy=40)
        corners = [[30, 20], [70, 20], [70, 60], [30, 60]]
        target, quality = plane_center(depth, np.ones_like(depth, bool), corners, k, [0, 0, 0.8])
        np.testing.assert_allclose(target, [0, 0, 0.4], atol=1e-12)
        self.assertGreater(quality["inlier_fraction"], 0.99)

    def test_missing_depth_and_depth_edge_rejected(self):
        depth = np.full((80, 100), 0.4)
        k = dict(fx=100, fy=100, cx=50, cy=40)
        corners = [[30, 20], [70, 20], [70, 60], [30, 60]]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            plane_center(depth, np.zeros_like(depth, bool), corners, k, [0, 0, 0.8])
        depth[:, 50:] = 0.8
        with self.assertRaises(ValueError):
            plane_center(depth, np.ones_like(depth, bool), corners, k, [0, 0, 0.8])
