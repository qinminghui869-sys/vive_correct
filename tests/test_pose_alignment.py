import unittest
import tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from vive_correct.batch.pose_alignment import (
    PoseStream,
    camera_wrist,
    fit_mount,
    propagate,
    transform,
)


class PoseAlignmentTests(unittest.TestCase):
    def test_duplicate_pose_timestamps_are_excluded_and_not_bridged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.parquet"
            data = {key: [0.0, 0.0, 0.0, 0.0] for key in ("x", "y", "z", "roll", "pitch", "yaw")}
            data["timestamp"] = [0.0, 0.02, 0.02, 0.04]
            pq.write_table(pa.table(data), path)
            stream = PoseStream.load(path)
            np.testing.assert_array_equal(stream.invalid_times, [0.02])
            with self.assertRaisesRegex(ValueError, "tracking gap"):
                stream.at(0.01)

    def test_rotation_interpolation_crosses_wrap_without_flipping(self):
        matrices = np.repeat(np.eye(4)[None], 2, axis=0)
        matrices[:, :3, :3] = Rotation.from_euler("z", [179, -179], degrees=True).as_matrix()
        matrices[1, 0, 3] = 2
        stream = PoseStream(np.array([0.0, 0.02]), matrices)
        middle = stream.at(0.01)
        self.assertAlmostEqual(middle[0, 3], 1)
        self.assertAlmostEqual(
            abs(Rotation.from_matrix(middle[:3, :3]).as_euler("xyz", degrees=True)[2]), 180
        )
        with self.assertRaisesRegex(ValueError, "extrapolate"):
            stream.at(0.03)

    def test_tracking_gap_is_not_interpolated(self):
        stream = PoseStream(np.array([0.0, 1.0]), np.repeat(np.eye(4)[None], 2, axis=0))
        with self.assertRaisesRegex(ValueError, "tracking gap"):
            stream.at(0.5)

    def test_short_gap_with_invalid_tracking_sample_is_not_bridged(self):
        stream = PoseStream(
            np.array([0.0, 0.04]), np.repeat(np.eye(4)[None], 2, axis=0), np.array([0.02])
        )
        with self.assertRaisesRegex(ValueError, "tracking gap"):
            stream.at(0.01)
        np.testing.assert_array_equal(stream.at(0), np.eye(4))

    def fixture(self, moving=True):
        rng = np.random.default_rng(42)
        camera = transform([0.1, -0.2, 0.15, 0.05, 0.02, 0.07])
        mounts = {
            "left": transform([0.3, 0.1, -0.2, 0.01, 0.02, 0.03]),
            "right": transform([-0.2, 0.2, 0.1, -0.02, 0.01, 0.04]),
        }
        rows = []
        for side in mounts:
            for i in range(12):
                head = transform(rng.normal(0, 0.3, 6)) if moving else np.eye(4)
                wrist = transform(rng.normal(0, 0.4, 6)) if moving else np.eye(4)
                rows.append(
                    {
                        "side": side,
                        "world_head": head,
                        "world_wrist": wrist,
                        "camera_reference": camera_wrist(head, wrist, camera, mounts[side]),
                        "physical_axes_verified": True,
                        "split": "train" if i < 9 else "validation",
                    }
                )
        return rows, camera, mounts

    def test_joint_mounting_recovers_known_transforms_and_holds_out_frames(self):
        rows, camera, mounts = self.fixture()
        fit = fit_mount(rows, transform([0.12, -0.18, 0.13, 0.06, 0.01, 0.08]))
        self.assertTrue(fit["accepted"])
        np.testing.assert_allclose(fit["head_camera"], camera, atol=1e-6)
        for side in mounts:
            np.testing.assert_allclose(fit["wrist_reference"][side], mounts[side], atol=1e-6)
        self.assertEqual(len(fit["validation"]), 6)

    def test_static_observations_do_not_calibrate_camera_and_mounts(self):
        rows, camera, _ = self.fixture(moving=False)
        fit = fit_mount(rows, camera)
        self.assertFalse(fit["accepted"])
        self.assertLess(fit["jacobian_rank"], fit["parameter_count"])

    def test_unverified_axes_and_unvalidated_propagation_are_rejected(self):
        rows, camera, _ = self.fixture()
        rows[0]["physical_axes_verified"] = False
        with self.assertRaisesRegex(ValueError, "physical axes"):
            fit_mount(rows, camera)
        with self.assertRaisesRegex(ValueError, "Unvalidated"):
            propagate({}, [0], {"accepted": False})

    def test_validation_failure_prevents_propagation(self):
        rows, camera, _ = self.fixture()
        rows[-1]["camera_reference"] = rows[-1]["camera_reference"].copy()
        rows[-1]["camera_reference"][0, 3] += 0.2
        fit = fit_mount(rows, camera)
        self.assertFalse(fit["accepted"])

    def test_impossible_mounting_offset_fails_even_with_perfect_visual_fit(self):
        rows, camera, mounts = self.fixture()
        mounts["left"][0, 3] = 0.8
        for row in rows:
            row["camera_reference"] = camera_wrist(
                row["world_head"], row["world_wrist"], camera, mounts[row["side"]]
            )
        fit = fit_mount(rows, camera)
        self.assertFalse(fit["accepted"])
        self.assertFalse(fit["installation_plausible"])


if __name__ == "__main__":
    unittest.main()
