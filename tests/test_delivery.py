"""The public report preserves poses and uncertainty without internal diagnostics."""

import unittest
import numpy as np

from vive_correct.batch.delivery import compact_report


class CompactReportTests(unittest.TestCase):
    def test_cv_only_identity_is_preserved_in_delivery(self):
        report = dict(
            identity="cv_image_half_unverified",
            reviewer_type="cv_only",
            observations=[
                dict(
                    frame_index=0,
                    stickers={
                        "left": dict(
                            origin_camera_m=[0, 0, 1],
                            axes_generated=False,
                            identity_review={"source": "cv_image_half_unverified"},
                        )
                    },
                )
            ],
        )
        result = compact_report(report, {})
        self.assertEqual(result["reviewer_type"], "cv_only")
        self.assertEqual(result["identity"], "cv_image_half_unverified")
        self.assertEqual(
            result["observations"][0]["stickers"]["left"]["identity_source"],
            "cv_image_half_unverified",
        )

    def test_compact_pose_preserves_transform_and_quality(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.1, 0.2, 1]
        pose = dict(
            T_camera_local=transform.tolist(),
            rms_px=0.2,
            image_corners_px=[[60, 60], [100, 60], [100, 90], [60, 90]],
            object_corners_local_m=[[0, 0, 0]] * 4,
            plane_ambiguous=True,
            orientation_180_unresolved=True,
            orientation_status="user_constraint_applied",
            physical_axes_verified=False,
            stability={"passed": False, "reason": "motion", "fb_max_px": 10},
            temporal_consistency={"passed": False, "reason": "pose_jump"},
            baseline_retention={"source_report": "/private/report"},
            axis_review={"old_from_new": np.eye(3).tolist()},
            identity_review={"visible_features": "debug"},
        )
        report = dict(
            observations=[
                dict(
                    frame_index=5,
                    timestamp_s=0.25,
                    stability_window_frames=11,
                    quality_status="conditional_motion",
                    image="/missing/source.png",
                    stickers={"left": pose},
                )
            ]
        )
        result = compact_report(
            report, {"accepted": False, "reason": "needs_validation", "fit": {"debug": 1}}
        )
        observation = result["observations"][0]
        self.assertEqual(result["schema_version"], 6)
        self.assertEqual(set(observation), {"frame_index", "timestamp_s", "stickers"})
        slim = observation["stickers"]["left"]
        np.testing.assert_array_equal(slim["T_camera_local"], transform)
        self.assertEqual(slim["quality_status"], "conditional_motion")
        self.assertTrue(slim["plane_ambiguous"])
        self.assertTrue(slim["orientation_180_unresolved"])
        self.assertTrue(slim["baseline_retained"])
        self.assertFalse(slim["stability_passed"])
        self.assertFalse(slim["temporal_consistency_passed"])
        self.assertFalse(slim["physical_axes_verified"])
        self.assertIsNone(slim["true_vive_origin_calibrated"])
        for field in (
            "image_corners_px",
            "object_corners_local_m",
            "axis_review",
            "identity_review",
            "baseline_retention",
            "stability",
            "temporal_consistency",
        ):
            self.assertNotIn(field, slim)
        self.assertEqual(
            result["mount_calibration"], {"accepted": False, "reason": "needs_validation"}
        )

    def test_missing_checks_remain_unknown(self):
        report = {
            "observations": [
                {"frame_index": 0, "stickers": {"right": {"T_camera_local": np.eye(4).tolist()}}}
            ]
        }
        pose = compact_report(report, {})["observations"][0]["stickers"]["right"]
        for field in (
            "stability_passed",
            "temporal_consistency_passed",
            "plane_ambiguous",
            "physical_axes_verified",
            "orientation_180_unresolved",
        ):
            self.assertIsNone(pose[field])
        self.assertEqual(pose["orientation_status"], "unknown")
        self.assertFalse(pose["baseline_retained"])

    def test_origin_only_has_no_transform_or_arrows(self):
        from unittest.mock import patch
        from vive_correct.estimate.geometry import Camera, estimate_pose
        from vive_correct.estimate.visualization import draw_poses
        from vive_correct.batch.selection_policy import temporal_check

        corners = np.array([[20.0, 20.0], [60.0, 20.0], [60.0, 50.0], [20.0, 50.0]])
        candidate = {
            "rms_px": 4.5,
            "R": np.eye(3),
            "t": np.array([0.1, 0.2, 1.0]),
            "corners": corners,
        }
        with (
            patch("vive_correct.estimate.geometry.pose_candidates", return_value=[candidate]),
            patch(
                "vive_correct.estimate.geometry.local_basis", side_effect=ValueError("axis gate")
            ),
        ):
            pose = estimate_pose(corners, None, allow_origin_only=True, device_fixed=False)
        self.assertFalse(pose["axes_generated"])
        self.assertIsNone(pose["T_camera_local"])
        self.assertEqual(pose["origin_camera_m"], [0.1, 0.2, 1.0])
        report = {"observations": [{"frame_index": 0, "stickers": {"left": pose}}]}
        slim = compact_report(report, {})["observations"][0]["stickers"]["left"]
        self.assertEqual(slim["origin_camera_m"], [0.1, 0.2, 1.0])
        self.assertEqual(slim["axes_failure_reason"], "direction_limits")
        self.assertIsNone(slim["T_camera_local"])
        self.assertIsNone(temporal_check([], pose, None)["passed"])
        camera = Camera(np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]]), (100, 100), "test")
        with patch("vive_correct.estimate.visualization.cv2.arrowedLine") as arrow:
            image = draw_poses(np.zeros((100, 100, 3), np.uint8), {"left": pose}, camera)
            arrow.assert_not_called()
            self.assertTrue(image.any())
