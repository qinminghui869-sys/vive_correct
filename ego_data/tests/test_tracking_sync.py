import unittest
import numpy as np
from tracking_sync import resample_tracking, normalize_tracking


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


class TimestampNormalizationTests(unittest.TestCase):
    def test_sort_identical_duplicates_and_invalid_time(self):
        stream, report = normalize_tracking(
            ([0.02, 0, 0.01, 0.01, np.nan], [[2], [0], [1], [1], [99]])
        )
        np.testing.assert_allclose(resample_tracking(stream, [0.005, 0.015])[1][:, 0], [0.5, 1.5])
        self.assertTrue(report["reordered"])
        self.assertEqual(report["identical_duplicates"], 1)
        self.assertEqual(report["invalid_timestamps"], 1)

    def test_singleton_and_empty_streams(self):
        result = resample_tracking(([1], [[4]]), [0, 1, 2])[1][:, 0]
        self.assertEqual(result[1], 4)
        self.assertTrue(np.isnan(result[[0, 2]]).all())
        result = resample_tracking(([], np.empty((0, 3))), [1])[1]
        self.assertTrue(np.isnan(result).all())

    def test_batched_rotation_matches_slerp(self):
        from scipy.spatial.transform import Rotation, Slerp

        angles = np.random.default_rng(4).uniform(-170, 170, (2, 3))
        values = np.column_stack([np.zeros((2, 3)), angles])
        times = np.linspace(0, 0.02, 9)
        got = resample_tracking(([0, 0.02], values), times, pose=True)[1][:, 3:]
        expected = Slerp([0, 0.02], Rotation.from_euler("xyz", angles, degrees=True))(times)
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", got, degrees=True).as_matrix(),
            expected.as_matrix(),
            atol=1e-12,
        )
