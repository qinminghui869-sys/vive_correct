import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from hand_calibration import fit_rigid, apply_mount, fit_calibration
from wrist_alignment import pose_matrix


class HandCalibrationTests(unittest.TestCase):
    def test_recovers_rotation_and_preserves_bones_under_moving_tracker(self):
        source = np.random.default_rng(2).normal(size=(21, 3)) * 0.05
        rotation = Rotation.from_euler("xyz", [20, 30, -15], degrees=True).as_matrix()
        target = source @ rotation.T + [0.02, -0.01, 0.03]
        matrix = fit_rigid(source, target)
        for yaw in (0, 90, 175):
            wrist = pose_matrix([1, 2, 3, 0, 0, yaw])
            world = source @ wrist[:3, :3].T + wrist[:3, 3]
            corrected = apply_mount(world, wrist, matrix)
            np.testing.assert_allclose(
                corrected, target @ wrist[:3, :3].T + wrist[:3, 3], atol=1e-12
            )
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(corrected, axis=0), axis=1),
                np.linalg.norm(np.diff(source, axis=0), axis=1),
                atol=1e-12,
            )

    def test_validation_frames_reject_bad_fit_and_leakage(self):
        rng = np.random.default_rng(5)
        rows = []
        for frame in range(5):
            for point in rng.normal(size=(6, 3)) * 0.05:
                rows.append(
                    dict(
                        frame=frame,
                        split="train" if frame < 3 else "validation",
                        source_tracker_m=point.tolist(),
                        target_tracker_m=(point + [0, 0, 0.1 if frame >= 3 else 0]).tolist(),
                    )
                )
        doc = dict(
            episode="episode",
            camera_calibration={},
            wrist_calibration_sha256="hash",
            reference_method="independent markers",
            hands={"left": rows},
        )
        self.assertFalse(fit_calibration(doc)["accepted"])
        rows[-1]["frame"] = 0
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_calibration(doc)

    def test_collinear_is_rejected(self):
        points = np.column_stack([np.arange(6), np.zeros((6, 2))])
        with self.assertRaisesRegex(ValueError, "Collinear"):
            fit_rigid(points, points)
