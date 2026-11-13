import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from wrist_alignment import hand_mount_summary


class HandMountTests(unittest.TestCase):
    def test_fixed_offset_recovered_across_rotated_poses(self):
        poses = np.array([[1, 2, 3, 0, 0, 0], [2, 3, 4, 10, 20, 30], [3, 2, 1, -20, 30, 40]], float)
        offset = np.array([0, -0.085, -0.058])
        roots = poses[:, :3] + Rotation.from_euler("xyz", poses[:, 3:], degrees=True).apply(offset)
        points = np.repeat(roots[:, None, :], 21, axis=1)
        summary = hand_mount_summary(points, poses)
        np.testing.assert_allclose(summary["wrist_landmark_in_tracker_m"], offset, atol=1e-12)
        self.assertLess(summary["offset_consistency_rms_m"], 1e-12)

    def test_missing_samples_are_reported(self):
        summary = hand_mount_summary(np.full((1, 21, 3), np.nan), np.zeros((1, 6)))
        self.assertEqual(summary["status"], "no_synchronized_samples")
