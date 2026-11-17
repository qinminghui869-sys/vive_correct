"""Axis projection, rigid binding and reviewed export regression tests."""

import unittest
import threading
import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from test_human_review import make_fixture

from vive_correct.axis_calibration import (
    axis_angles,
    local_rotation,
    preview,
    project_axes,
    corner_correction,
)
from vive_correct.human_review import ReviewStore
from vive_correct.parquet_correction import translate
from vive_correct.review_bundle import approve, load_bundle, stage_files
from vive_correct.review_io import atomic_json, read_json
from vive_correct.source_data import DATA
from vive_correct.review_server import ReviewServer
from vive_correct.estimate.geometry import Camera, LABEL_CORNERS, local_basis


class AxisTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode, self.output, self.mount = make_fixture(self.root)
        self.store = ReviewStore(self.output)

    def set_rotations(self, stream, angles):
        path = self.episode / DATA / f"observation.{stream}_pose.parquet"
        table = pq.read_table(path)
        for i, name in enumerate(("roll", "pitch", "yaw")):
            table = table.set_column(
                table.schema.get_field_index(name),
                name,
                pa.array(np.asarray(angles)[:, i], type=pa.float64()),
            )
        pq.write_table(table, path)

    def approve(self, angles=(0, 0, 0)):
        state = self.store.load(0)
        state["reviewer"] = "axis-test"
        for side, choice in state["selected"].items():
            choice.update(status="accepted", physical_side=side, axis_angles_deg=list(angles))
        return approve(self.store, 0, state, self.mount)

    def test_nonidentity_source_projection_matches_approval(self):
        self.set_rotations("head", [[10, 20, 30]] * 4)
        self.set_rotations("left_wrist", [[-20, 40, 50]] * 4)
        mount = read_json(self.mount)
        mount["T_head_rectified_camera"] = np.eye(4).tolist()
        rm = Rotation.from_euler("xyz", [5, -10, 15], degrees=True).as_matrix()
        for i in range(3):
            mount["T_head_rectified_camera"][i][:3] = rm[i].tolist()
        atomic_json(self.mount, mount)
        choice = dict(self.store.load(0)["selected"]["left"], axis_angles_deg=[30, -20, 90])
        before = list(self.output.rglob("human_review.json"))
        result = preview(self.episode, "left", choice, self.mount)
        self.assertEqual(before, list(self.output.rglob("human_review.json")))
        expected = (
            (Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix() @ rm).T
            @ Rotation.from_euler("xyz", [-20, 40, 50], degrees=True).as_matrix()
            @ local_rotation(choice)
        )
        np.testing.assert_allclose(
            np.asarray(result["T_camera_local"])[:3, :3], expected, atol=1e-12
        )
        origin = np.asarray(result["origin_camera_m"])
        expected_center = [600 * origin[0] / origin[2] + 320, 600 * origin[1] / origin[2] + 200]
        np.testing.assert_allclose(result["axis_projection"]["origin"], expected_center)
        bundle = self.approve([30, -20, 90])
        pose = read_json(bundle["report"])["observations"][0]["stickers"]["left"]
        np.testing.assert_allclose(pose["T_camera_local"], result["T_camera_local"])
        self.assertEqual(
            read_json(bundle["alignment"])["wrists"]["left"]["model"], "wrist_local_rigid"
        )

    def test_local_rotation_order_and_camera_plane(self):
        choice = dict(axis_angles_deg=[30, 40, 50])
        rx, ry, rz = [
            Rotation.from_euler(a, v, degrees=True).as_matrix() for a, v in zip("xyz", [30, 40, 50])
        ]
        np.testing.assert_allclose(local_rotation(choice), rx @ ry @ rz)
        for angles in ([0, 0, 0], [90, 0, 0], [-90, 0, 0]):
            r = local_rotation(dict(axis_angles_deg=angles))
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1)
        result = project_axes([0, 0, 0.01], np.diag([1, -1, -1]), np.eye(3))
        self.assertIsNone(result["endpoints"][2])
        for values in ([0, float("nan"), 0], [0, 1], [[1, 2, 3]], [0, float("inf"), 0]):
            with self.assertRaises(ValueError):
                axis_angles(dict(axis_angles_deg=values))

    def test_corner_correction_recovers_plane_with_nonidentity_source(self):
        self.set_rotations("head", [[10, 20, 30]] * 4)
        self.set_rotations("left_wrist", [[-20, 40, 50]] * 4)
        camera = Camera.load(self.episode / "calibrations/rectify_params.yaml")
        tilt = Rotation.from_euler("xyz", [12, -18, 8], degrees=True).as_matrix()
        label = tilt @ np.diag([1.0, -1, -1])
        target = tilt @ np.diag([-1.0, -1, 1])
        quad = camera.project(LABEL_CORNERS, label, np.array([-0.05, 0.02, 0.5]))
        choice = dict(self.store.load(0)["selected"]["left"], quad=quad.tolist())
        for corners in (quad, np.roll(quad, 2, axis=0), quad[::-1]):
            result = preview(self.episode, "left", dict(choice, quad=corners.tolist()), self.mount)
            correction = result["corner_correction"]
            self.assertIsNotNone(correction, result["corner_correction_error"])
            np.testing.assert_allclose(correction["rotation_camera_target"], target, atol=1e-6)
            corrected = preview(
                self.episode,
                "left",
                dict(choice, **{"axis_angles_deg": correction["axis_angles_deg"]}),
                self.mount,
            )
            np.testing.assert_allclose(
                np.array(corrected["T_camera_local"])[:3, :3], target, atol=1e-6
            )
            np.testing.assert_allclose(corrected["origin_camera_m"], result["origin_camera_m"])
        state = self.store.load(0)
        state["selected"]["left"].update(choice, axis_angles_deg=correction["axis_angles_deg"])
        self.store.save(0, state)
        loaded = self.store.load(0)["selected"]["left"]
        np.testing.assert_allclose(local_rotation(loaded), local_rotation(correction))

    def test_steep_corner_plane_and_gimbal_lock_keep_rigid_rotation(self):
        camera = Camera.load(self.episode / "calibrations/rectify_params.yaml")
        label = Rotation.from_euler("x", 65, degrees=True).as_matrix()
        position = np.array([0.0, 0, 0.5])
        with self.assertRaises(ValueError):
            local_basis(label, position, camera)
        target = label @ local_basis(label, position, camera, require_direction_limits=False)
        expected = local_rotation(dict(axis_angles_deg=[25, 90, 15]))
        result = corner_correction(
            dict(
                rotation_camera_label=label.tolist(),
                pnp_origin_camera_m=position.tolist(),
                calibration_sample=dict(rotation_camera_source=(target @ expected.T).tolist()),
            ),
            camera,
        )
        np.testing.assert_allclose(local_rotation(result), expected, atol=1e-12)
        np.testing.assert_allclose(target.T @ target, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(target), 1)

    def test_rigid_wrist_moves_with_source_rotation_and_preserves_nulls(self):
        self.set_rotations("left_wrist", [[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270]])
        bundle = self.approve([0, 90, 0])
        alignment = read_json(bundle["alignment"])
        raw = pq.read_table(self.episode / DATA / "observation.left_wrist_pose.parquet")
        actual, _ = translate(raw, "observation.left_wrist_pose.parquet", alignment)
        offset = np.asarray(alignment["wrists"]["left"]["offset_local_m"])
        rotation = Rotation.from_euler("z", [0, 90, 180, 270], degrees=True)
        np.testing.assert_allclose(
            np.column_stack([actual[c].to_numpy() - raw[c].to_numpy() for c in "xyz"]),
            rotation.apply(offset),
            atol=1e-10,
        )
        corrected = Rotation.from_euler(
            "xyz",
            np.column_stack([actual[c].to_numpy() for c in ("roll", "pitch", "yaw")]),
            degrees=True,
        ).as_matrix()
        np.testing.assert_allclose(
            corrected,
            rotation.as_matrix() @ local_rotation(dict(axis_angles_deg=[0, 90, 0])),
            atol=1e-10,
        )
        raw = raw.set_column(
            raw.schema.get_field_index("pitch"), "pitch", pa.array([0.0, None, float("nan"), 0.0])
        )
        actual, counts = translate(raw, "observation.left_wrist_pose.parquet", alignment)
        self.assertEqual(counts["x_skipped"], 2)
        for c in ("x", "y", "z", "roll", "pitch", "yaw"):
            np.testing.assert_equal(actual[c].to_numpy()[1:3], raw[c].to_numpy()[1:3])
        self.assertEqual(actual.schema, raw.schema)

    def test_hand_rotation_preserves_shape_and_skips_unsynchronized_rows(self):
        path = self.episode / DATA / "observation.hand_left_points.parquet"
        raw = pq.read_table(path)
        raw = raw.set_column(
            raw.schema.get_field_index("thumbTip_x"),
            "thumbTip_x",
            pa.array([0.3] * 5, type=pa.float32()),
        )
        pq.write_table(raw, path)
        bundle = self.approve([0, 0, 90])
        staged = stage_files(bundle)
        actual = pq.read_table(staged["files"][path.name]["path"])
        point = lambda t, joint: np.column_stack([t[f"{joint}_{c}"].to_numpy() for c in "xyz"])
        np.testing.assert_allclose(
            np.linalg.norm(point(actual, "thumbTip") - point(actual, "wrist"), axis=1),
            np.linalg.norm(point(raw, "thumbTip") - point(raw, "wrist"), axis=1),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            point(actual, "thumbTip")[:4] - point(actual, "wrist")[:4],
            [[0, 0.2, 0], [0, 0.175, 0], [0, 0.15, 0], [0, 0.125, 0]],
            atol=1e-7,
        )
        self.assertTrue(actual.slice(4).equals(raw.slice(4)))
        self.assertEqual(staged["files"][path.name]["corrected_groups"]["wrist_skipped"], 1)

    def test_draft_migration_and_axis_edits_invalidate_bundle(self):
        bundle = self.approve()
        state = self.store.load(0)
        state["selected"]["left"]["axis_angles_deg"] = [15, 25, 35]
        state["selected"]["left"]["status"] = "pending"
        self.store.save(0, state)
        self.assertEqual(
            ReviewStore(self.output).load(0)["selected"]["left"]["axis_angles_deg"], [15, 25, 35]
        )
        with self.assertRaises(ValueError):
            load_bundle(self.store, 0, self.mount)
        path = self.store.path(0).with_name("human_review.json")
        legacy = read_json(path)
        legacy["schema_version"] = 2
        for c in legacy["selected"].values():
            c.pop("axis_angles_deg")
            c.pop("axis_convention")
        atomic_json(path, legacy)
        migrated = self.store.load(0)
        self.assertTrue(migrated["needs_reconfirmation"])
        self.assertEqual(migrated["selected"]["left"]["axis_angles_deg"], [0, 0, 0])
        self.assertEqual(migrated["selected"]["left"]["quad"], legacy["selected"]["left"]["quad"])
        self.assertTrue(Path(bundle["report"]).exists())

    def test_missing_source_pose_and_depth_preview_agree(self):
        choice = self.store.load(0)["selected"]["left"]
        self.set_rotations("left_wrist", [[float("nan"), 0, 0]] * 4)
        with self.assertRaisesRegex(ValueError, "synchronized left wrist"):
            preview(self.episode, "left", choice, self.mount)
        self.set_rotations("left_wrist", [[0, 0, 0]] * 4)
        target = [-0.1, 0.05, 0.6]
        with patch(
            "ego_data.depth_targets.refine_targets",
            return_value={(0, "left"): dict(status="depth_plane_refined", origin_camera_m=target)},
        ):
            result = preview(self.episode, "left", choice, self.mount)
        np.testing.assert_allclose(result["origin_camera_m"], target)

    def test_preview_http_is_readonly_and_reuses_geometry_until_inputs_change(self):
        from vive_correct.review_server import preview as calculate

        server = ReviewServer(self.output, 0, self.mount)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        self.addCleanup(conn.close)
        choice = dict(self.store.load(0)["selected"]["left"], side="left")
        headers = {"Content-Type": "application/json", "X-Review-Token": server.token}

        def request(payload, auth=headers):
            conn.request("POST", "/api/axis-preview?id=0", json.dumps(payload), auth)
            response = conn.getresponse()
            return response.status, json.loads(response.read())

        self.assertEqual(request(choice, {"Content-Type": "application/json"})[0], 403)
        with patch("vive_correct.review_server.preview", wraps=calculate) as compute:
            status, source = request(choice)
            self.assertEqual(status, 200)
            status, rotated = request(dict(choice, axis_angles_deg=[0, 0, 90]))
            self.assertEqual(status, 200)
            self.assertEqual(compute.call_count, 1)
            np.testing.assert_allclose(rotated["origin_camera_m"], source["origin_camera_m"])
            np.testing.assert_allclose(
                np.asarray(rotated["T_camera_local"])[:3, :3],
                np.asarray(source["T_camera_local"])[:3, :3]
                @ Rotation.from_euler("z", 90, degrees=True).as_matrix(),
                atol=1e-12,
            )
            mount = read_json(self.mount)
            mount["T_head_rectified_camera"][0][3] += 0.01
            atomic_json(self.mount, mount)
            self.assertEqual(request(choice)[0], 200)
            self.assertEqual(compute.call_count, 2)
        self.assertEqual(request(dict(choice, quad=[]))[0], 400)
        self.assertFalse(self.store.path(0).with_name("human_review.json").exists())
        self.assertFalse(list(self.output.rglob("approval.json")))

    def test_rgb_rrd_axes_use_corrected_rotation(self):
        from ego_data.corrected_rgb import export_corrected_rgb
        import rerun as rr

        bundle = self.approve([20, 30, 40])
        staged = stage_files(bundle)
        vectors = []
        original = rr.Arrows3D

        def arrows(**kwargs):
            vectors.append(kwargs["vectors"])
            return original(**kwargs)

        with patch("ego_data.corrected_rgb.rr.Arrows3D", side_effect=arrows):
            export_corrected_rgb(
                self.episode,
                self.root / "axes.rrd",
                bundle["alignment"],
                selected=[0],
                reviewed_data=Path(staged["data_directory"]),
            )
        self.assertEqual(len(vectors), 2)
        expected = local_rotation(dict(axis_angles_deg=[20, 30, 40])).T * 0.04
        for actual in vectors:
            np.testing.assert_allclose(actual, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
