"""Six-file reconstruction, immutable baselines and split review/export contracts."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from test_human_review import make_fixture

from vive_correct.batch.report_io import digest
from vive_correct.estimate.geometry import Camera
from vive_correct.review.human_review import ReviewStore, source_signature
from vive_correct.review.parquet_correction import position_groups, verify_table
from vive_correct.review.review_actions import export_approved, rebuild
from vive_correct.review.review_bundle import approve, review_state, stage_files
from vive_correct.review.review_io import atomic_json, read_json
from vive_correct.data.source_data import DATA, FILES, apply_files, original_files, recover


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode, self.output, self.mount = make_fixture(self.root)
        self.store = ReviewStore(self.output)

    def approve(self, store=None, index=0):
        store = store or self.store
        state = store.load(index)
        state["reviewer"] = "test-reviewer"
        for side, choice in state["selected"].items():
            choice.update(status="accepted", physical_side=side)
        return approve(store, index, state, self.mount)

    def hashes(self):
        return {n: digest(self.episode / DATA / n) for n in FILES}

    def test_approval_contains_geometry_images_and_does_not_touch_sources(self):
        before = self.hashes()
        with patch("vive_correct.export.rerun_export.export_recordings") as render:
            bundle = self.approve()
            render.assert_not_called()
        self.assertEqual(before, self.hashes())
        self.assertFalse(list(self.output.rglob("*.rrd")))
        report = read_json(bundle["report"])
        alignment = read_json(bundle["alignment"])
        self.assertEqual([r["frame_index"] for r in report["observations"]], [0, 1])
        for row in report["observations"]:
            for pose in row["stickers"].values():
                self.assertEqual(len(pose["label_center_px"]), 2)
                self.assertEqual(len(pose["origin_camera_m"]), 3)
        for sample in alignment["samples"]:
            self.assertEqual(len(sample["wrist_pose_original"]), 6)
            self.assertIn("depth_refinement", sample)
        self.assertTrue(
            all(
                Path(image[k]).is_file()
                for image in bundle["images"].values()
                for k in ("clean", "overlay")
            )
        )
        self.assertTrue(review_state(self.store, 0, self.mount)["approved"])

    def test_reprojection_failure_identifies_side_and_preserves_draft(self):
        # Regression: the desk label is visible, but its four corners do not fit
        # the calibrated rectangle within 6 px. Never approve by relaxing the gate.
        camera = Camera(
            np.array(
                [
                    [811.1893512894039, 0, 937.2783966064453],
                    [0, 811.1893512894039, 600.5800399780273],
                    [0, 0, 1],
                ]
            ),
            (1920, 1200),
            "regression",
        )
        quad = [[68.0, 742.0], [230.0, 720.0], [180.0, 798.0], [48.0, 804.0]]
        before = self.hashes()
        initial = self.store.load(0)
        for side, label in (("left", "左手"), ("right", "右手")):
            state = self.store.load(0)
            state["reviewer"] = "test-reviewer"
            state["selected"] = {
                s: dict(c, status="accepted", physical_side=s)
                for s, c in initial["selected"].items()
            }
            state["selected"][side].update(frame_index=2, quad=quad)
            with (
                self.subTest(side=side),
                patch("vive_correct.review.human_review.Camera.load", return_value=camera),
                patch(
                    "vive_correct.review.human_review.decode_frame",
                    return_value=(np.zeros((1200, 1920, 3), dtype=np.uint8), 4),
                ),
                patch("vive_correct.export.rerun_export.export_recordings") as render,
                self.assertRaisesRegex(ValueError, f"{label}第 2 帧：重投影误差 6.94 px.*6 px"),
            ):
                approve(self.store, 0, state, self.mount)
            render.assert_not_called()
            saved = self.store.load(0)
            self.assertEqual(saved["selected"][side]["quad"], quad)
            self.assertEqual(saved["selected"][side]["frame_index"], 2)
            self.assertEqual(saved["status"], "pending")
            self.assertFalse(self.store.path(0).with_name("approval.json").exists())
            self.assertEqual(before, self.hashes())

    def test_six_tables_preserve_nulls_partial_xyz_rotations_and_rows(self):
        path = self.episode / DATA / "observation.hand_left_points.parquet"
        raw = pq.read_table(path)
        raw = raw.set_column(
            raw.schema.get_field_index("wrist_x"),
            "wrist_x",
            pa.array([0.1, None, np.nan, 0.4, 0.5], type=pa.float32()),
        )
        raw = raw.replace_schema_metadata({b"original": b"preserve me"})
        pq.write_table(raw, path)
        bundle = self.approve()
        staged = stage_files(bundle)
        alignment = read_json(bundle["alignment"])
        self.assertEqual(set(staged["files"]), set(FILES))
        for name, row in staged["files"].items():
            original, result = pq.read_table(self.episode / DATA / name), pq.read_table(row["path"])
            self.assertEqual(original.schema, result.schema)
            self.assertEqual(len(original), len(result))
            changed = set()
            for side, columns in position_groups(name):
                x = np.column_stack([original[c].to_numpy() for c in columns])
                valid = np.isfinite(x).all(1)
                from scipy.spatial.transform import Rotation

                if name in ("action.parquet", "observation.state.parquet"):
                    angles = np.column_stack(
                        [original[f"{side}_wrist_{a}"].to_numpy() for a in ("roll", "pitch", "yaw")]
                    )
                    offsets = Rotation.from_euler("xyz", angles, degrees=True).apply(
                        alignment["wrists"][side]["offset_local_m"]
                    )
                else:
                    offsets = np.tile(alignment["wrists"][side]["offset_local_m"], (len(x), 1))
                    if "hand_" in name:
                        valid &= original["timestamp"].to_numpy() <= 0.1
                for i, c in enumerate(columns):
                    changed.add(c)
                    np.testing.assert_allclose(
                        result[c].to_numpy()[valid] - original[c].to_numpy()[valid],
                        offsets[valid, i],
                        atol=1e-7,
                    )
                    np.testing.assert_equal(
                        result[c].to_numpy()[~valid], original[c].to_numpy()[~valid]
                    )
                    self.assertTrue(result[c].is_null().equals(original[c].is_null()))
            for c in set(original.column_names) - changed:
                self.assertTrue(original[c].equals(result[c]), c)

    def test_verification_rejects_lost_schema_and_field_metadata(self):
        field = pa.field("x", pa.float64(), metadata={b"unit": b"metres"})
        expected = pa.Table.from_arrays(
            [pa.array([1.0, None])],
            schema=pa.schema([field], metadata={b"source": b"original"}),
        )
        lost_schema = expected.replace_schema_metadata(None)
        lost_field = expected.set_column(0, field.with_metadata(None), expected["x"])
        for actual in (lost_schema, lost_field):
            with self.subTest(schema=actual.schema), self.assertRaisesRegex(ValueError, "schema"):
                verify_table(actual, expected)

    def test_backup_repeat_and_changed_review_always_use_original_baseline(self):
        before, signature = self.hashes(), source_signature(self.episode)
        self.approve()
        first = rebuild(self.output, self.mount, in_place=True)
        self.assertEqual(first["status"], "complete", first)
        first_hashes = self.hashes()
        self.assertNotEqual(first_hashes, before)
        self.assertEqual(source_signature(self.episode), signature)
        self.assertTrue(review_state(self.store, 0, self.mount)["approved"])
        self.assertEqual({n: digest(p) for n, p in original_files(self.episode).items()}, before)
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        self.assertEqual(self.hashes(), first_hashes)
        state = self.store.load(0)
        state["selected"]["left"]["quad"][0][0] += 2
        self.store.save(0, state)
        bundle = self.approve()
        expected = stage_files(bundle)
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        self.assertEqual(self.hashes(), {n: r["sha256"] for n, r in expected["files"].items()})
        self.assertEqual({n: digest(p) for n, p in original_files(self.episode).items()}, before)

    def test_unknown_backup_damaged_backup_and_external_edits_are_rejected(self):
        bundle = self.approve()
        staged = stage_files(bundle)
        target = self.episode / DATA / FILES[0]
        backup = target.with_name(target.name + ".bak")
        backup.write_bytes(b"unregistered")
        with self.assertRaisesRegex(ValueError, "未登记"):
            apply_files(
                self.episode, staged["files"], bundle["baseline_hashes"], bundle["directory"]
            )
        self.assertEqual(backup.read_bytes(), b"unregistered")
        backup.unlink()
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        original_bytes = backup.read_bytes()
        backup.write_bytes(b"damaged")
        self.assertFalse(review_state(self.store, 0, self.mount)["approved"])
        backup.write_bytes(original_bytes)
        target.write_bytes(b"external edit")
        result = rebuild(self.output, self.mount, in_place=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("外部修改", result["entries"][0]["error"])
        self.assertEqual(target.read_bytes(), b"external edit")

    def test_backup_created_by_another_process_is_never_removed(self):
        bundle = self.approve()
        staged = stage_files(bundle)
        before = self.hashes()
        foreign = self.episode / DATA / (FILES[2] + ".bak")
        real_link = os.link

        def race(source, destination):
            if Path(destination) == foreign:
                foreign.write_bytes(b"external backup")
            return real_link(source, destination)

        with patch("vive_correct.data.source_data.os.link", side_effect=race):
            with self.assertRaises(FileExistsError):
                apply_files(
                    self.episode, staged["files"], bundle["baseline_hashes"], bundle["directory"]
                )
        self.assertEqual(before, self.hashes())
        self.assertEqual(foreign.read_bytes(), b"external backup")

    def test_render_read_guard_blocks_in_place_replacement(self):
        from vive_correct.data.source_data import baseline_read

        bundle = self.approve()
        staged = stage_files(bundle)
        before = self.hashes()
        with baseline_read(self.episode):
            with self.assertRaisesRegex(ValueError, "正在读取"):
                apply_files(
                    self.episode, staged["files"], bundle["baseline_hashes"], bundle["directory"]
                )
        self.assertEqual(before, self.hashes())

    def test_mid_write_failure_rolls_back_all_six_files(self):
        bundle = self.approve()
        staged = stage_files(bundle)
        before = self.hashes()
        real_replace = os.replace

        def fail(source, destination):
            if Path(source).parent.name == "new" and Path(destination).name == FILES[2]:
                raise OSError("simulated disk failure")
            return real_replace(source, destination)

        with patch("vive_correct.data.source_data.os.replace", side_effect=fail):
            with self.assertRaisesRegex(OSError, "simulated"):
                apply_files(
                    self.episode, staged["files"], bundle["baseline_hashes"], bundle["directory"]
                )
        self.assertEqual(before, self.hashes())
        self.assertFalse(any((self.episode / DATA / (n + ".bak")).exists() for n in FILES))
        self.assertEqual(
            read_json(self.episode / ".vive_rebuild/transaction.json")["status"], "rolled_back"
        )

    def test_process_interruption_restores_previous_success_not_initial_baseline(self):
        self.approve()
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        before = self.hashes()
        backup_hashes = {n: digest(p) for n, p in original_files(self.episode).items()}
        state = self.store.load(0)
        state["selected"]["left"]["quad"][0][0] += 2
        self.store.save(0, state)
        bundle = self.approve()
        staged = stage_files(bundle)

        class Crash(BaseException):
            pass

        real_replace = os.replace

        def crash(source, destination):
            if Path(source).parent.name == "new" and Path(destination).name == FILES[2]:
                raise Crash()
            return real_replace(source, destination)

        with patch("vive_correct.data.source_data.os.replace", side_effect=crash):
            with self.assertRaises(Crash):
                apply_files(
                    self.episode, staged["files"], bundle["baseline_hashes"], bundle["directory"]
                )
        self.assertEqual(
            read_json(self.episode / ".vive_rebuild/transaction.json")["status"], "applying"
        )
        recover(self.episode)
        self.assertEqual(self.hashes(), before)
        self.assertEqual(
            {n: digest(p) for n, p in original_files(self.episode).items()}, backup_hashes
        )
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")

    def test_missing_schema_or_sync_cannot_be_approved_and_mount_invalidates(self):
        bundle = self.approve()
        mount = read_json(self.mount)
        mount["T_head_rectified_camera"][0][3] += 0.01
        atomic_json(self.mount, mount)
        self.assertFalse(review_state(self.store, 0, self.mount)["approved"])
        self.assertEqual(rebuild(self.output, self.mount)["status"], "incomplete")
        self.approve()
        path = self.episode / DATA / "action.parquet"
        table = pq.read_table(path).drop(["left_wrist_x"])
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, "left_wrist_x"):
            self.approve()
        self.assertFalse(review_state(self.store, 0, self.mount)["approved"])
        self.assertTrue(Path(bundle["report"]).exists())

    def test_export_reads_staged_data_and_does_not_apply_offset_twice_after_overwrite(self):
        import rerun as rr

        from ego_data.corrected_rgb import export_corrected_rgb

        bundle = self.approve()
        staged = stage_files(bundle)
        expected = pq.read_table(staged["files"]["observation.left_wrist_pose.parquet"]["path"])
        expected_xyz = np.column_stack([expected[a].to_numpy() for a in "xyz"])
        originals = pq.read_table(self.episode / DATA / "observation.left_wrist_pose.parquet")
        self.assertEqual(rebuild(self.output, self.mount, in_place=True)["status"], "complete")
        logged = []
        real_points = rr.Points3D

        def points(values, **kwargs):
            logged.append(np.asarray(values).copy())
            return real_points(values, **kwargs)

        with patch("ego_data.corrected_rgb.rr.Points3D", side_effect=points):
            stats = export_corrected_rgb(
                self.episode,
                self.root / "actual.rrd",
                bundle["alignment"],
                reviewed_data=staged["data_directory"],
            )
        self.assertEqual(stats["frames"], 4)
        np.testing.assert_allclose(logged[0][0], [originals[a][0].as_py() for a in "xyz"])
        np.testing.assert_allclose(logged[1][0], expected_xyz[0])
        row = export_approved(self.store, 0, self.mount)
        self.assertEqual(row["status"], "complete", row)
        self.assertTrue(Path(row["recording"]).exists())
        with patch("vive_correct.review.review_actions.export_recordings") as render:
            cached = export_approved(self.store, 0, self.mount)
            self.assertTrue(cached["reused"])
            render.assert_not_called()

    def test_one_failed_episode_does_not_block_another_and_selector_is_exact(self):
        _, output, _ = make_fixture(self.root / "other")
        atomic_json(
            self.output / "features/second/features.json",
            read_json(output / "features/sample/features.json"),
        )
        store = ReviewStore(self.output)
        self.approve(store, 1)
        result = rebuild(self.output, self.mount)
        self.assertEqual([r["status"] for r in result["entries"]], ["failed", "prepared"])
        self.assertEqual(
            len(rebuild(self.output, self.mount, episode=store.load(1)["episode"])["entries"]), 1
        )
        with self.assertRaises(ValueError):
            rebuild(self.output, self.mount, episode=self.root / "missing")


if __name__ == "__main__":
    unittest.main()
