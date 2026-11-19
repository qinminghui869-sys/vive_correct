"""Eye-mode geometry, source preservation and approval/export integration."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from test_human_review import make_fixture
from vive_correct.camera_input import from_args, rectify, timestamp_suffix, video_suffix
from vive_correct.dataset import VIDEO_SUFFIX, TIMESTAMP_SUFFIX, discover
from vive_correct.eye_reference import (
    POINT_COLUMNS,
    POSE_COLUMNS,
    values,
    transform_poses,
    validate_hands,
    map_state_fields,
)
from vive_correct.hand_repair import world_points, OFFSET
from vive_correct.human_review import ReviewStore, checked_pose, decode_frame
from vive_correct.axis_calibration import preview
from vive_correct.parquet_correction import translate
from vive_correct.review_bundle import approve, stage_files, load_bundle
from vive_correct.review_actions import rebuild, export_approved
from vive_correct.review_io import atomic_json, read_json
from vive_correct.source_data import DATA, FILES, original_files
from vive_correct.batch.report_io import digest
from vive_correct.estimate.geometry import LABEL_CORNERS


def make_eye_fixture(root):
    episode, output, mount = make_fixture(root)
    calibration = root / "neon.json"
    atomic_json(
        calibration,
        dict(
            camera_matrix=[[600, 0, 320], [0, 600, 200], [0, 0, 1]],
            dist_coefs=[0, 0, 0, 0, 0],
            resolution=[640, 400],
        ),
    )
    context = from_args(
        SimpleNamespace(coordinate_mode="eye-reference", eye_calibration=calibration)
    )
    (episode / video_suffix(context)).parent.mkdir(parents=True)
    (episode / VIDEO_SUFFIX).rename(episode / video_suffix(context))
    (episode / TIMESTAMP_SUFFIX).rename(episode / timestamp_suffix(context))
    (episode / DATA / "observation.head_pose.parquet").unlink()
    times = np.arange(4) / 30
    skeleton = np.zeros((4, 50, 3))
    for block in (0, 25):
        for j in range(1, 25):
            skeleton[:, block + j] = [
                j * 0.002,
                (j % 5) * 0.003,
                (j % 3) * 0.002 * (1 if block else -1),
            ]
    fields = {"timestamp": times}
    fields.update(
        {f"node{i:02d}_{a}": skeleton[:, i, k] for i in range(50) for k, a in enumerate("xyz")}
    )
    pq.write_table(pa.table(fields), episode / DATA / "observation.hand_skeleton.parquet")
    state = {"timestamp": times, "untouched": [9] * 4}
    for side, block, device in (("left", 0, "device2"), ("right", 25, "device0")):
        path = episode / DATA / f"observation.{side}_wrist_pose.parquet"
        raw = pq.read_table(path)
        for axis, angles in zip(
            ("roll", "pitch", "yaw"), ([10, 15, 20, 25], [20, 25, 30, 35], [30, 40, 50, 60])
        ):
            raw = raw.set_column(
                raw.schema.get_field_index(axis), axis, pa.array(angles, type=pa.float64())
            )
        # Physically consistent camera/source fixture for a single shared extrinsic.
        from vive_correct.estimate.geometry import local_basis
        from ego_data.wrist_alignment import pose_matrix
        from vive_correct.axis_calibration import local_rotation, prepare_pose

        q = read_json(output / "features/sample/features.json")["features"][side]["quad"]
        observed, cam = checked_pose(episode, dict(frame_index=0, quad=q), context=context)
        prepare_pose(observed, cam, dict(frame_index=0, quad=q))
        label = np.asarray(observed["rotation_camera_label"])
        anchor = np.eye(4)
        anchor[:3, :3] = (
            label
            @ local_basis(
                label, np.asarray(observed["origin_camera_m"]), cam, require_direction_limits=False
            )
            @ local_rotation(dict(axis_angles_deg=[12, -18, 25]))
        )
        anchor[:3, 3] = observed["origin_camera_m"]
        extrinsic = pose_matrix([0.2, -0.4, 0.7, 20, -10, 35])
        initial = np.linalg.inv(extrinsic) @ anchor
        rows = np.tile(
            np.r_[
                initial[:3, 3], Rotation.from_matrix(initial[:3, :3]).as_euler("xyz", degrees=True)
            ],
            (4, 1),
        )
        rows[:, 0] += np.arange(4) * 0.01
        rows[:, 3:] += np.arange(4)[:, None] * [5, 3, 8]
        for i, col in enumerate(POSE_COLUMNS):
            raw = raw.set_column(raw.schema.get_field_index(col), col, pa.array(rows[:, i]))
        pq.write_table(raw, path)
        pose = values(raw, POSE_COLUMNS)
        points = world_points(skeleton, pose, block).reshape(4, 63)
        hand = pa.table(
            {
                "timestamp": times,
                **{
                    c: pa.array(points[:, i], type=pa.float32())
                    for i, c in enumerate(POINT_COLUMNS)
                },
            }
        )
        pq.write_table(hand, episode / DATA / f"observation.hand_{side}_points.parquet")
        state.update({f"{device}_{c}": raw[c] for c in POSE_COLUMNS})
    for name in ("action.parquet", "observation.state.parquet"):
        pq.write_table(
            pa.table(state).replace_schema_metadata({b"custom": b"keep"}), episode / DATA / name
        )
    atomic_json(
        episode / ".vive_rewrite_journal.json",
        dict(mapping={"L": "observation.device2_pose", "R": "observation.device0_pose"}),
    )
    report_path = output / "features/sample/features.json"
    report = read_json(report_path)
    report["camera_input"] = context
    for choice in report["features"].values():
        choice["frame_index"] = 0
    atomic_json(report_path, report)
    return episode, output, mount, context


class EyeReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.episode, self.output, self.mount, self.context = make_eye_fixture(self.root)
        self.store = ReviewStore(self.output)

    def accept(self):
        state = self.store.load(0)
        state["reviewer"] = "automated-test"
        for side, c in state["selected"].items():
            c.update(status="accepted", physical_side=side, axis_angles_deg=[12, -18, 25])
        return approve(self.store, 0, state, self.mount)

    def test_shared_fit_preserves_relative_poses_with_different_device_rotations(self):
        from copy import deepcopy
        from ego_data.wrist_alignment import pose_matrix
        from vive_correct.eye_reference import fit_shared_reference

        expected = pose_matrix([0.2, -0.4, 0.7, 20, -10, 35])
        originals = [
            pose_matrix([-0.1, 0.8, 0.3, 30, 40, 20]),
            pose_matrix([0.3, 0.9, 0.2, -20, 10, 60]),
        ]
        samples = []
        for side, original in zip(("left", "right"), originals):
            samples.append(
                dict(
                    side=side,
                    timestamp_s=1.0,
                    wrist_pose_original=np.r_[
                        original[:3, 3],
                        Rotation.from_matrix(original[:3, :3]).as_euler("xyz", degrees=True),
                    ].tolist(),
                    reference_pose=(expected @ original).tolist(),
                )
            )
        source = deepcopy(samples)
        actual, quality = fit_shared_reference(samples, require_bilateral=True)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertTrue(quality["consistent"])
        np.testing.assert_allclose(
            np.linalg.inv(samples[0]["reference_pose"]) @ samples[1]["reference_pose"],
            np.linalg.inv(originals[0]) @ originals[1],
            atol=1e-12,
        )
        source[1]["reference_pose"][0][3] += 0.3
        with self.assertRaisesRegex(ValueError, "共同外参不一致"):
            fit_shared_reference(source, require_bilateral=True)

    def test_shared_preview_changes_both_axes_and_cache_when_other_hand_changes(self):
        from copy import deepcopy
        from vive_correct.review_server import ReviewServer

        selected = {
            s: dict(c, axis_angles_deg=[12, -18, 25])
            for s, c in self.store.load(0)["selected"].items()
        }
        server = ReviewServer(self.output, 0, self.mount)
        try:
            first = server.axis_preview(0, dict(selected["left"], side="left", selected=selected))
            changed = deepcopy(selected)
            changed["right"]["axis_angles_deg"][0] += 10
            second = server.axis_preview(0, dict(changed["left"], side="left", selected=changed))
            right = server.axis_preview(0, dict(changed["right"], side="right", selected=changed))
            self.assertFalse(np.allclose(first["T_camera_local"], second["T_camera_local"]))
            np.testing.assert_allclose(
                second["calibration_sample"]["T_reference_source"],
                right["calibration_sample"]["T_reference_source"],
            )
            self.assertFalse(second["shared_fit"]["provisional"])
            self.assertEqual(second["local_axes_convention"], "source_vive_camera_projection")
            np.testing.assert_allclose(
                np.asarray(first["T_world_local"])[:3, :3],
                np.asarray(second["T_world_local"])[:3, :3],
            )
        finally:
            server.server_close()

    def test_inconsistent_two_hand_review_cannot_be_approved(self):
        state = self.store.load(0)
        state["reviewer"] = "automated-test"
        for side, c in state["selected"].items():
            c.update(status="accepted", physical_side=side, axis_angles_deg=[12, -18, 25])
        state["selected"]["right"]["axis_angles_deg"] = [120, 40, -100]
        with self.assertRaisesRegex(ValueError, "共同外参不一致"):
            approve(self.store, 0, state, self.mount)
        self.assertFalse(self.store.path(0).with_name("approval.json").exists())

    def test_eye_discovery_and_same_decode_as_scan(self):
        self.assertEqual(discover(self.episode, self.context), [self.episode])
        self.assertEqual(
            discover(self.episode), [self.episode]
        )  # legacy calibration remains discoverable
        capture = cv2.VideoCapture(str(self.episode / video_suffix(self.context)))
        ok, raw = capture.read()
        capture.release()
        self.assertTrue(ok)
        image, _ = decode_frame(self.episode, 0, self.context)
        np.testing.assert_array_equal(image, rectify(raw, self.context))
        with self.assertRaises(ValueError):
            rectify(raw[:100], self.context)

    def test_distorted_pixels_undistort_and_pnp_recovers_metric_pose(self):
        K = np.asarray(self.context["K"], float)
        d = np.array([-0.15, 0.05, 0.001, 0, 0])
        R = Rotation.from_euler("xyz", [165, 15, 10], degrees=True).as_matrix()
        t = np.array([-0.1, 0.05, 0.6])
        pixels = cv2.projectPoints(LABEL_CORNERS, cv2.Rodrigues(R)[0], t, K, d)[0]
        undistorted = cv2.undistortPoints(pixels, K, d, P=K).reshape(4, 2)
        choice = dict(frame_index=0, quad=undistorted.tolist())
        pose, _ = checked_pose(self.episode, choice, context=self.context)
        np.testing.assert_allclose(pose["origin_camera_m"], t, atol=1e-5)

    def test_preview_approval_anchor_and_propagation_agree(self):
        choice = dict(self.store.load(0)["selected"]["left"], axis_angles_deg=[12, -18, 25])
        selected = {
            side: dict(c, axis_angles_deg=[12, -18, 25])
            for side, c in self.store.load(0)["selected"].items()
        }
        result = preview(self.episode, "left", choice, self.mount, self.context, selected=selected)
        self.assertFalse(result["shared_fit"]["provisional"])
        self.assertEqual(len(result["hand_projection"]), 21)
        bundle = self.accept()
        alignment = read_json(bundle["alignment"])
        self.assertEqual(alignment["version"], 5)
        self.assertEqual(
            alignment["wrists"]["left"]["T_reference_source"],
            alignment["wrists"]["right"]["T_reference_source"],
        )
        sample = next(s for s in alignment["samples"] if s["side"] == "left")
        np.testing.assert_allclose(result["T_camera_local"], sample["camera_pose"], atol=1e-9)
        np.testing.assert_allclose(result["T_world_local"], sample["reference_pose"], atol=1e-9)
        camera_reference = np.asarray(alignment["T_camera_reference"])
        reference_source = np.asarray(alignment["T_reference_source"])
        np.testing.assert_allclose(
            camera_reference @ reference_source, alignment["T_camera_source"], atol=1e-12
        )
        np.testing.assert_array_equal(reference_source[:3, :3], np.eye(3))
        np.testing.assert_allclose(
            reference_source @ np.r_[alignment["camera_origin_source_m"], 1], [0, 0, 0, 1]
        )
        hand_name = "observation.hand_left_points.parquet"
        hand, _ = translate(pq.read_table(self.episode / DATA / hand_name), hand_name, alignment)
        world_points = values(hand, POINT_COLUMNS)[0].reshape(21, 3)
        camera_points = world_points @ camera_reference[:3, :3].T
        pixels = camera_points @ np.asarray(self.context["K"]).T
        np.testing.assert_allclose(
            result["hand_projection"], pixels[:, :2] / pixels[:, 2:], atol=1e-4
        )
        raw = pq.read_table(self.episode / DATA / FILES[0])
        corrected, _ = translate(raw, FILES[0], alignment)
        from ego_data.wrist_alignment import pose_matrix

        original = values(raw, POSE_COLUMNS)
        new = values(corrected, POSE_COLUMNS)
        np.testing.assert_array_equal(new[:, 3:], original[:, 3:])
        A = np.asarray(sample["reference_pose"])
        for i in range(4):
            expected = (
                A
                @ np.linalg.inv(pose_matrix(original[0], "xyz", True))
                @ pose_matrix(original[i], "xyz", True)
            )
            np.testing.assert_allclose(pose_matrix(new[i], "xyz", True), expected, atol=1e-9)

    def test_hand_shape_offset_and_field_preservation(self):
        bundle = self.accept()
        alignment = read_json(bundle["alignment"])
        delta = np.asarray(alignment["T_reference_source"])[:3, 3]
        staged = stage_files(bundle)
        for side in ("left", "right"):
            raw = pq.read_table(self.episode / DATA / f"observation.hand_{side}_points.parquet")
            hand = pq.read_table(
                Path(staged["data_directory"]) / f"observation.hand_{side}_points.parquet"
            )
            pose = pq.read_table(
                Path(staged["data_directory"]) / f"observation.{side}_wrist_pose.parquet"
            )
            q = values(hand, POINT_COLUMNS).reshape(-1, 21, 3)
            q0 = values(raw, POINT_COLUMNS).reshape(-1, 21, 3)
            np.testing.assert_allclose(q, q0 + delta, atol=1e-7)
            np.testing.assert_allclose(
                np.linalg.norm(q[:, 0] - values(pose, POSE_COLUMNS)[:, :3], axis=1),
                np.linalg.norm(OFFSET),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                np.linalg.norm(q[:, :, None] - q[:, None, :], axis=3),
                np.linalg.norm(q0[:, :, None] - q0[:, None, :], axis=3),
                atol=1e-7,
            )
            self.assertEqual(hand.schema.metadata[b"coordinate_frame"], b"eye_origin_standing_axes")
        state = pq.read_table(Path(staged["data_directory"]) / "observation.state.parquet")
        self.assertEqual(state["untouched"].to_pylist(), [9] * 4)
        self.assertEqual(state.schema.metadata[b"custom"], b"keep")
        raw_state = pq.read_table(self.episode / DATA / "observation.state.parquet")
        for device in ("device0", "device2"):
            for angle in ("roll", "pitch", "yaw"):
                self.assertEqual(state[f"{device}_{angle}"], raw_state[f"{device}_{angle}"])
        self.assertEqual(stage_files(bundle), staged)

    def test_origin_only_preserves_noncanonical_and_null_angles(self):
        bundle = self.accept()
        alignment = read_json(bundle["alignment"])
        name = "observation.left_wrist_pose.parquet"
        raw = pq.read_table(self.episode / DATA / name)
        for angle in ("roll", "pitch", "yaw"):
            raw = raw.set_column(
                raw.schema.get_field_index(angle),
                angle,
                pa.array([720.0, -450.0, None, 180.0], type=pa.float64()),
            )
        new, _ = translate(raw, name, alignment)
        for angle in ("roll", "pitch", "yaw"):
            self.assertEqual(new[angle], raw[angle])
        self.assertTrue(np.isnan(values(new, ("x", "y", "z"))[2]).all())
        with self.assertRaisesRegex(ValueError, "只允许平移"):
            transform_poses(
                values(raw, POSE_COLUMNS), alignment["T_camera_source"], preserve_rotation=True
            )

    def test_legacy_rotating_alignment_still_rotates_source(self):
        from ego_data.wrist_alignment import pose_matrix

        alignment = read_json(self.accept()["alignment"])
        alignment.update(
            version=4,
            extrinsic_model="shared_standing_to_eye",
            world_frame="eye_camera_at_reference_frame",
            world_axes="RDF",
            T_reference_source=alignment["T_camera_source"],
        )
        name = "observation.left_wrist_pose.parquet"
        raw = pq.read_table(self.episode / DATA / name)
        corrected, _ = translate(raw, name, alignment)
        for old, new in zip(values(raw, POSE_COLUMNS), values(corrected, POSE_COLUMNS)):
            np.testing.assert_allclose(
                pose_matrix(new),
                np.asarray(alignment["T_camera_source"]) @ pose_matrix(old),
                atol=1e-12,
            )

    def test_same_frame_required_and_draft_remains_editable(self):
        state = self.store.load(0)
        state["reviewer"] = "test"
        for side, c in state["selected"].items():
            c.update(status="accepted", physical_side=side)
        state["selected"]["right"]["frame_index"] = 1
        with self.assertRaisesRegex(ValueError, "同一参考帧"):
            approve(self.store, 0, state, self.mount)
        self.assertEqual(self.store.load(0)["status"], "pending")

    def test_mapping_conflict_and_mount_mismatch_rejected(self):
        validate_hands(self.episode)
        table = pq.read_table(self.episode / DATA / "action.parquet")
        self.assertEqual(
            map_state_fields(self.episode, table), {"left": "device2", "right": "device0"}
        )
        atomic_json(
            self.episode / ".vive_rewrite_journal.json",
            dict(mapping={"L": "observation.device0_pose"}),
        )
        with self.assertRaisesRegex(ValueError, "不一致"):
            map_state_fields(self.episode, table)
        path = self.episode / DATA / "observation.hand_left_points.parquet"
        hand = pq.read_table(path)
        hand = hand.set_column(
            hand.schema.get_field_index("wrist_x"),
            "wrist_x",
            pa.array(hand["wrist_x"].to_numpy() + 0.1),
        )
        pq.write_table(hand, path)
        with self.assertRaisesRegex(ValueError, "安装偏移"):
            validate_hands(self.episode)

    def test_calibration_and_source_changes_invalidate_approval(self):
        bundle = self.accept()
        self.assertEqual(load_bundle(self.store, 0, self.mount), bundle)
        cal = Path(self.context["calibration_path"])
        cal.write_text(cal.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "标定已变化"):
            load_bundle(self.store, 0, self.mount)

    def test_invalid_pose_and_joint_never_retain_source_coordinates(self):
        bundle = self.accept()
        alignment = read_json(bundle["alignment"])
        name = "observation.hand_left_points.parquet"
        raw = pq.read_table(self.episode / DATA / name)
        raw = pa.concat_tables([raw, raw.slice(0, 1)])
        raw = raw.set_column(0, "timestamp", pa.array([0.0, 1 / 30, 2 / 30, 0.1, 1.0]))
        z = raw["indexTip_z"].to_numpy().copy()
        z[1] = np.nan
        raw = raw.set_column(
            raw.schema.get_field_index("indexTip_z"), "indexTip_z", pa.array(z, type=pa.float32())
        )
        new, _ = translate(raw, name, alignment)
        self.assertTrue(np.isnan(values(new, POINT_COLUMNS)[-1]).all())
        self.assertTrue(np.isnan(values(new, ["indexTip_x", "indexTip_y", "indexTip_z"])[1]).all())
        poses = np.array([[1, 2, 3, 0, 0, np.nan]])
        self.assertTrue(np.isnan(transform_poses(poses, np.eye(4))).all())

    def test_rebuild_in_place_idempotent_and_backup_retained(self):
        originals = {n: digest(p) for n, p in original_files(self.episode).items()}
        self.accept()
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        first = {n: digest(self.episode / DATA / n) for n in FILES}
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        self.assertEqual(first, {n: digest(self.episode / DATA / n) for n in FILES})
        self.assertEqual(originals, {n: digest(p) for n, p in original_files(self.episode).items()})

    def test_eye_http_preview_does_not_depend_on_head_mount(self):
        from vive_correct.review_server import ReviewServer

        self.mount.unlink()
        server = ReviewServer(self.output, 0, self.mount)
        try:
            choice = self.store.load(0)["selected"]["left"]
            result = server.axis_preview(0, dict(choice, side="left", axis_angles_deg=[15, 5, 20]))
            expected = preview(
                self.episode,
                "left",
                dict(choice, axis_angles_deg=[15, 5, 20]),
                self.mount,
                self.context,
            )
            np.testing.assert_allclose(result["T_camera_local"], expected["T_camera_local"])
            np.testing.assert_allclose(result["hand_projection"], expected["hand_projection"])
            self.assertTrue(server.frame_png(0, 0).startswith(b"\x89PNG"))
        finally:
            server.server_close()

    def test_missing_hand_input_skips_catalog(self):
        from vive_correct.cv_features import scan_features
        from vive_correct.estimate.cv_config import CVConfig

        (self.episode / DATA / "observation.hand_skeleton.parquet").unlink()
        output = self.root / "skipped_output"
        result = scan_features(
            self.episode, output / "features" / "sample", CVConfig.load(), context=self.context
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("observation.hand_skeleton.parquet", result["skip_reason"])
        self.assertEqual(ReviewStore(output, allow_empty=True).paths, [])

    def test_eye_in_place_failure_restores_all_source_files(self):
        import os
        from vive_correct.source_data import apply_files

        bundle = self.accept()
        staged = stage_files(bundle)
        before = {n: digest(self.episode / DATA / n) for n in FILES}
        replace = os.replace
        failed = False

        def fail_once(source, target):
            nonlocal failed
            if not failed and Path(source).parent.name == "new" and Path(target).name == FILES[1]:
                failed = True
                raise OSError("injected replacement failure")
            return replace(source, target)

        with patch("vive_correct.source_data.os.replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                apply_files(
                    self.episode, staged["files"], bundle["baseline_hashes"], bundle["directory"]
                )
        self.assertTrue(failed)
        self.assertEqual(before, {n: digest(self.episode / DATA / n) for n in FILES})
        self.assertEqual(
            read_json(self.episode / ".vive_rebuild/transaction.json")["status"], "rolled_back"
        )

    def test_rrd_export_contains_all_frames_without_head_pose(self):
        self.accept()
        result = export_approved(self.store, 0, self.mount)
        self.assertEqual(result["status"], "complete", result.get("error"))
        stats = read_json(Path(result["recording"]).with_suffix(".json"))
        self.assertEqual(stats["frames"], 4)
        self.assertEqual(stats["valid_hand_frames"], {"left": 4, "right": 4})
        self.assertFalse(stats["per_frame_camera_registration"])
        self.assertEqual(stats["orientation_policy"], "preserve_source")
        self.assertEqual(stats["world_axes"], "RUB")
        self.assertTrue(export_approved(self.store, 0, self.mount)["reused"])


if __name__ == "__main__":
    unittest.main()
