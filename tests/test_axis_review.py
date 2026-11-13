import unittest
import numpy as np

from vive_correct.batch.axis_review import reorient_pose
from vive_correct.estimate.geometry import Camera


class AxisReviewTests(unittest.TestCase):
    def test_review_preserves_origin_and_corner_projections(self):
        camera = Camera(np.array([[800.0, 0, 960], [0, 800, 600], [0, 0, 1]]), (1920, 1200), "test")
        transform = np.eye(4)
        transform[:3, 3] = [0.1, 0.2, 1.0]
        pose = dict(
            R_object_to_camera=np.eye(3).tolist(),
            t_camera_m=[0.1, 0.2, 1.0],
            T_camera_local=transform.tolist(),
            T_label_local=np.eye(4).tolist(),
            object_corners_local_m=[
                [-0.025, 0, -0.015],
                [0.025, 0, -0.015],
                [0.025, 0, 0.015],
                [-0.025, 0, 0.015],
            ],
        )
        before = camera.project(
            pose["object_corners_local_m"], np.eye(3), np.array(pose["t_camera_m"])
        )
        for basis in (np.diag([1.0, -1, -1]), np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])):
            result = reorient_pose(pose, basis, camera)
            after = camera.project(
                result["object_corners_local_m"],
                np.array(result["R_object_to_camera"]),
                np.array(result["t_camera_m"]),
            )
            np.testing.assert_allclose(before, after)
            np.testing.assert_allclose(result["t_camera_m"], pose["t_camera_m"])
            self.assertAlmostEqual(np.linalg.det(result["R_object_to_camera"]), 1)
        np.testing.assert_array_equal(pose["R_object_to_camera"], np.eye(3))
        with self.assertRaisesRegex(ValueError, "right-handed"):
            reorient_pose(pose, np.diag([1.0, -1, 1]), camera)

    def test_reference_conversion_is_idempotent(self):
        from vive_correct.batch.axis_review import normalize_reference_axes

        camera = Camera(np.array([[800.0, 0, 960], [0, 800, 600], [0, 0, 1]]), (1920, 1200), "test")
        transform = np.eye(4)
        transform[2, 3] = 1
        pose = dict(
            R_object_to_camera=np.eye(3).tolist(),
            t_camera_m=[0.0, 0.0, 1.0],
            T_camera_local=transform.tolist(),
            T_label_local=np.eye(4).tolist(),
            object_corners_local_m=[
                [-0.025, 0, -0.015],
                [0.025, 0, -0.015],
                [0.025, 0, 0.015],
                [-0.025, 0, 0.015],
            ],
        )
        converted = normalize_reference_axes(pose, camera, "X=Xref, Y=-Zref, Z=Yref")
        repeated = normalize_reference_axes(converted, camera, "X=Xref, Y=-Zref, Z=Yref")
        np.testing.assert_array_equal(converted["T_camera_local"], repeated["T_camera_local"])
        np.testing.assert_array_equal(np.array(converted["T_camera_local"])[:3, 3], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "unknown"):
            normalize_reference_axes(pose, camera, "")

    def test_front_facing_label_has_left_up_forward_axes(self):
        from vive_correct.estimate.geometry import local_basis, final_axis_angles

        camera = Camera(np.array([[800.0, 0, 960], [0, 800, 600], [0, 0, 1]]), (1920, 1200), "test")
        reference = np.diag([-1.0, -1.0, 1.0])
        translation = np.array([0.0, 0.0, 0.5])
        rotation = reference @ local_basis(reference, translation, camera)
        np.testing.assert_allclose(rotation, reference, atol=1e-6)
        np.testing.assert_allclose(final_axis_angles(rotation, translation, camera), [0, 0, 0])
