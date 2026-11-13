"""Exercise manual wrist offsets, independent sides and save/export interoperability."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from align_wrists import App
from wrist_alignment import correction_world


class ManualOffsetTests(unittest.TestCase):
    def app(self, path):
        app = App.__new__(App)
        app.args = SimpleNamespace(
            output=path, pose_euler_sequence="xyz", pose_radians=False, world_axes="LUF"
        )
        app.mount = {"T_head_rectified_camera": np.eye(4).tolist()}
        app.models = {"left": "wrist_local", "right": "world"}
        app.manual_offsets = {
            "left": np.array([0.01, -0.02, 0.03]),
            "right": np.array([-0.04, 0.05, 0.06]),
        }
        app.samples = []
        app.status = SimpleNamespace(set=lambda _: None)
        return app

    def test_manual_save_without_samples_both_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrists.json"
            app = self.app(path)
            app.save()
            saved = json.loads(path.read_text())
            self.assertEqual(saved["world_axes"], "LUF")
            self.assertEqual(len(saved["wrists"]), 2)
            for side in ("left", "right"):
                self.assertEqual(saved["wrists"][side]["method"], "manual")
                np.testing.assert_allclose(
                    saved["wrists"][side]["offset_m"], app.manual_offsets[side]
                )
            rotated_pose = np.array([0.1, 0.2, 0.3, 0, 0, 90])
            left = correction_world(saved, "left", rotated_pose)
            right = correction_world(saved, "right", rotated_pose)
            expected = (
                Rotation.from_euler("xyz", rotated_pose[3:], degrees=True).as_matrix()
                @ app.manual_offsets["left"]
            )
            np.testing.assert_allclose(left, expected, atol=1e-12)
            np.testing.assert_allclose(right, app.manual_offsets["right"])

    def test_manual_override_residual_and_fit_restore(self):
        app = self.app(Path("/tmp/unused_ego_manual.json"))
        rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        delta = rotation @ app.manual_offsets["left"]
        app.samples = [
            {"side": "left", "wrist_rotation": rotation.tolist(), "delta_world_m": delta.tolist()}
        ]
        offset, errors = app.fit("left")
        self.assertLess(errors[0], 1e-12)
        app.manual_offsets["left"] = np.zeros(3)
        self.assertGreater(app.fit("left")[1][0], 0.01)
        app.manual_offsets.pop("left")
        np.testing.assert_allclose(app.fit("left")[0], offset, atol=1e-12)
        # Editing left does not alter the independent right world-space correction.
        np.testing.assert_allclose(app.fit("right")[0], [-0.04, 0.05, 0.06])


if __name__ == "__main__":
    unittest.main()
