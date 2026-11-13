import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from wrist_alignment import fit_samples, correction_world, backproject, project_world


class AlignmentTests(unittest.TestCase):
    def test_rotating_wrist_local_offset(self):
        local = np.array([0.02, -0.01, 0.03])
        rotations = Rotation.from_euler(
            "xyz", [[0, 0, 0], [90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True
        ).as_matrix()
        samples = [
            {"delta_world_m": (r @ local).tolist(), "wrist_rotation": r.tolist()} for r in rotations
        ]
        offset, errors = fit_samples(samples, "wrist_local")
        np.testing.assert_allclose(offset, local, atol=1e-12)
        self.assertLess(max(errors), 1e-12)
        _, world_errors = fit_samples(samples, "world")
        self.assertGreater(max(world_errors), 0.01)
        config = {"wrists": {"left": {"model": "wrist_local", "offset_m": offset.tolist()}}}
        pose = np.array([0, 0, 0, 90, 0, 0])
        np.testing.assert_allclose(
            correction_world(config, "left", pose), rotations[1] @ local, atol=1e-12
        )
        np.testing.assert_allclose(correction_world(config, "right", pose), np.zeros(3))

    def test_world_offset_and_outlier(self):
        samples = [
            {"delta_world_m": [0.01, 0.02, -0.03], "wrist_rotation": np.eye(3).tolist()}
            for _ in range(4)
        ]
        samples.append({"delta_world_m": [1, 2, 3], "wrist_rotation": np.eye(3).tolist()})
        offset, errors = fit_samples(samples, "world")
        np.testing.assert_allclose(offset, [0.01, 0.02, -0.03])
        self.assertGreater(errors[-1], 1)

    def test_camera_projection_roundtrip(self):
        k = dict(fx=790, fy=795, cx=930, cy=599)
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", [30, 10, 15], degrees=True).as_matrix()
        camera[:3, 3] = [0.1, 0.3, -0.2]
        point = backproject(830, 620, 0.55, k)
        world = camera[:3, :3] @ point + camera[:3, 3]
        np.testing.assert_allclose(project_world(world, camera, k), [830, 620])
        self.assertIsNone(project_world(camera[:3, 3] - camera[:3, 2], camera, k))


if __name__ == "__main__":
    unittest.main()
