import unittest
import numpy as np
from tracking_sync import resample_tracking


class TrackingSyncTests(unittest.TestCase):
    def test_rotation_crosses_180_using_short_arc(self):
        values = np.array([[0, 0, 0, 0, 0, 179], [2, 0, 0, 0, 0, -179]], dtype=float)
        result = resample_tracking(([0, 0.02], values), [0.01], pose=True)[1][0]
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(abs(result[-1]), 180)

    def test_no_extrapolation_gap_or_invalid_bridge(self):
        stream = ([0, 0.02, 0.1, 0.12], [[0], [2], [10], [float("nan")]])
        result = resample_tracking(stream, [-0.01, 0, 0.01, 0.06, 0.11, 0.13])[1][:, 0]
        np.testing.assert_allclose(result[1:3], [0, 1])
        self.assertTrue(np.isnan(result[[0, 3, 4, 5]]).all())

    def test_duplicate_timestamps_are_not_averaged_or_bridged(self):
        result = resample_tracking(
            ([0, 0.01, 0.01, 0.02], [[0], [1], [9], [2]]), [0.005, 0.01, 0.015]
        )[1]
        self.assertTrue(np.isnan(result).all())
