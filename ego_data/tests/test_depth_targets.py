import unittest
import numpy as np
from ego_data.depth_targets import plane_center


class DepthTargetTests(unittest.TestCase):
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
