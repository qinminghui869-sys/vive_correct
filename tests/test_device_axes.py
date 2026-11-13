"""Device references must never be rebuilt from image directions."""

import unittest
from unittest.mock import patch
import numpy as np

from vive_correct.batch.device_axes import bind_anchor
from vive_correct.batch.mount_calibration import fixed_pose
from vive_correct.estimate.geometry import Camera, LABEL_CORNERS
from vive_correct.estimate.find_stable_frame import fit_candidate


class DeviceAxesTests(unittest.TestCase):
    def setUp(self):
        self.source = {"baseline_source": "/report", "baseline_sha256": "hash"}
        transform = np.eye(4)
        transform[:3, 3] = [0.1, 0.2, 1.0]
        self.pose = {
            "T_camera_local": transform.tolist(),
            "orientation_status": "user_constraint_applied",
            "axis_review": {
                "reviewer_type": "user_direction_feedback",
                "input_direction_feedback_confirmed": True,
            },
        }

    def test_bound_user_reference_is_unchanged_and_idempotent(self):
        first = bind_anchor(self.pose, "left", 10, self.source)
        second = bind_anchor(first, "left", 10, self.source)
        self.assertTrue(first["axes_generated"])
        self.assertEqual(first["T_camera_local"], self.pose["T_camera_local"])
        self.assertEqual(first, second)

    def test_reference_cannot_cross_hand_or_frame(self):
        bound = bind_anchor(self.pose, "left", 10, self.source)
        for side, frame in [("right", 10), ("left", 11)]:
            result = bind_anchor(bound, side, frame, self.source)
            self.assertFalse(result["axes_generated"])
            self.assertEqual(result["axes_failure_reason"], "reference_scope_mismatch")

    def test_unreviewed_reference_keeps_origin_only(self):
        self.pose["orientation_status"] = "not_direction_reviewed"
        result = bind_anchor(self.pose, "left", 10, self.source)
        self.assertFalse(result["axes_generated"])
        self.assertIsNone(result["T_camera_local"])
        self.assertEqual(result["origin_camera_m"], [0.1, 0.2, 1.0])

    def test_display_conversion_cannot_become_device_reference(self):
        self.pose["local_axes_convention"] = "X_left_Y_up_Z_forward"
        result = bind_anchor(self.pose, "left", 10, self.source)
        self.assertFalse(result["axes_generated"])
        self.assertEqual(result["axes_failure_reason"], "use_original_device_reference")

    def test_fresh_detection_does_not_call_screen_axis_selector(self):
        camera = Camera(np.array([[800.0, 0, 960], [0, 800, 600], [0, 0, 1]]), (1920, 1200), "test")
        import cv2

        rotation = cv2.Rodrigues(np.array([0.3, 0.2, 1.2]))[0]
        corners = camera.project(LABEL_CORNERS, rotation, np.array([0.1, 0.1, 0.6]))
        with patch(
            "vive_correct.estimate.geometry.local_basis",
            side_effect=AssertionError("must not select axes"),
        ):
            result = fit_candidate(
                np.zeros((1200, 1920, 3), np.uint8), corners, camera, refine=False
            )
        self.assertFalse(result["axes_generated"])
        np.testing.assert_allclose(result["origin_camera_m"], [0.1, 0.1, 0.6], atol=1e-5)

    def test_tracking_accepts_rotated_device_without_screen_constraints(self):
        import cv2

        camera = Camera(np.array([[800.0, 0, 960], [0, 800, 600], [0, 0, 1]]), (1920, 1200), "test")
        rotation = cv2.Rodrigues(np.array([0.3, 0.1, 2.0]))[0]
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.1, 0.1, 0.6]
        corners = camera.project(LABEL_CORNERS, rotation, transform[:3, 3])
        with patch(
            "vive_correct.estimate.geometry.local_basis",
            side_effect=AssertionError("must not select axes"),
        ):
            tracked = fixed_pose(corners, LABEL_CORNERS, camera, transform)
        np.testing.assert_allclose(tracked, transform, atol=1e-5)
