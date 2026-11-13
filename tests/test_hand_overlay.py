"""Hand rendering must respect field names, synchronization and camera depth."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vive_correct.batch.hand_overlay import JOINTS, HandStream, HandOverlay, project


class HandOverlayTests(unittest.TestCase):
    def test_named_columns_and_nearest_sample_tolerance(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hand.parquet"
            data = {"timestamp": [0.0, 0.01, 0.03]}
            for i, joint in enumerate(JOINTS):
                for axis in "zyx":
                    data[f"{joint}_{axis}"] = [float(i)] * 3
            pq.write_table(pa.table(data), path)
            stream = HandStream(path)
            points, delta = stream.at(0.011)
            np.testing.assert_allclose(points[:, 0], np.arange(21))
            self.assertAlmostEqual(delta, -0.001)
            for timestamp in (-0.001, 0.02, 0.04):
                with self.assertRaises(ValueError):
                    stream.at(timestamp)

    def test_projection_rejects_behind_camera_invalid_and_offscreen(self):
        k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
        points = np.array([[0.0, 0, 2], [0, 0, -1], [np.nan, 0, 1], [10, 0, 1]])
        pixels, valid = project(points, np.eye(4), k, (100, 100, 3))
        np.testing.assert_allclose(pixels[0], [50, 50])
        self.assertEqual(valid.tolist(), [True, False, False, False])

    def test_missing_calibration_does_not_invent_registration(self):
        with TemporaryDirectory() as tmp:
            overlay = HandOverlay(tmp, {}, {})
            image = np.zeros((50, 50, 3), np.uint8)
            result, record = overlay.draw(image, 1.0)
            np.testing.assert_array_equal(image, result)
            self.assertEqual(record["status"], "unavailable")

    def test_world_to_camera_translation(self):
        matrix = np.eye(4)
        matrix[0, 3] = -1
        k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
        pixels, valid = project(np.array([[1.0, 0, 2]]), matrix, k, (100, 100, 3))
        np.testing.assert_allclose(pixels, [[50, 50]])
        self.assertTrue(valid[0])
