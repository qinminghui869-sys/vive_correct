"""Exercise manual review, immutable repairs and the local HTTP boundary."""

import http.client
import json
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ego_data.vive_origin_alignment import build_alignment
from vive_correct.dataset import TIMESTAMP_SUFFIX, VIDEO_SUFFIX
from vive_correct.estimate.geometry import VIDEO_KEY
from vive_correct.human_review import ReviewStore
from vive_correct.review_io import atomic_json, output_lock, read_json
from vive_correct.review_repair import (
    DATA,
    repair_dataset,
    write_corrected_wrists,
)
from vive_correct.review_repair import (
    main as repair_main,
)
from vive_correct.review_server import DEFAULT_PORT, ReviewServer
from vive_correct.review_server import main as server_main


def make_fixture(root):
    episode, output = root / "source", root / "output"
    (episode / VIDEO_SUFFIX).parent.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(episode / VIDEO_SUFFIX), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 400)
    )
    quads = [
        [[120.0, 262.0], [180.0, 262.0], [180.0, 298.0], [120.0, 298.0]],
        [[460.0, 262.0], [520.0, 262.0], [520.0, 298.0], [460.0, 298.0]],
    ]
    for frame in range(4):
        image = np.full((400, 640, 3), 180, dtype=np.uint8)
        for q in quads:
            cv2.fillConvexPoly(image, np.asarray(q, np.int32), (245, 245, 245))
        cv2.putText(
            image, f"TEST FRAME {frame}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        writer.write(image)
    writer.release()
    (episode / "calibrations").mkdir()
    (episode / "calibrations/rectify_params.yaml").write_text(
        yaml.safe_dump(
            {
                "cameras": {
                    VIDEO_KEY: {
                        "camera_model": "pinhole",
                        "distortion_model": "none",
                        "intrinsics": [600, 600, 320, 200],
                        "resolution": [640, 400],
                    }
                }
            }
        )
    )
    data = episode / DATA
    data.mkdir(parents=True)
    times = np.arange(4) / 30
    pq.write_table(pa.table({"timestamp": times}), episode / TIMESTAMP_SUFFIX)
    for side, x in (("head", 0.0), ("left_wrist", 1.0), ("right_wrist", 2.0)):
        fields = dict(
            timestamp=times,
            x=np.array([x, x + 0.01, x + 0.02, x + 0.03]),
            y=np.zeros(4),
            z=np.zeros(4),
            roll=np.zeros(4),
            pitch=np.zeros(4),
            yaw=np.zeros(4),
            episode_index=[0] * 4,
        )
        pq.write_table(
            pa.table(fields).replace_schema_metadata({b"example": b"preserve"}),
            data / f"observation.{side}_pose.parquet",
        )
    folder = output / "features" / "sample"
    atomic_json(
        folder / "features.json",
        dict(
            episode=str(episode),
            features={
                s: dict(frame_index=i, quad=quads[i], observation_id=f"auto-{s}")
                for i, s in enumerate(("left", "right"))
            },
            missing_hands=[],
        ),
    )
    atomic_json(folder / "candidate_pool.json", [dict(frame_index=2, quad=quads[0])])
    mount = root / "mount.json"
    atomic_json(
        mount,
        dict(
            T_head_rectified_camera=np.eye(4).tolist(),
            head_euler_sequence="xyz",
            head_euler_degrees=True,
        ),
    )
    from vive_correct.batch.hand_overlay import JOINTS

    hand_times = np.arange(5) / 30
    for side in ("left", "right"):
        fields = {"timestamp": hand_times, "episode_index": np.zeros(5, dtype=np.int64)}
        fields.update(
            {
                f"{joint}_{axis}": np.linspace(0.1, 0.2, 5).astype("float32")
                for joint in JOINTS
                for axis in "xyz"
            }
        )
        pq.write_table(pa.table(fields), data / f"observation.hand_{side}_points.parquet")
    fields = {
        "timestamp": hand_times,
        "episode_index": np.zeros(5, dtype=np.int64),
        "head_x": np.ones(5),
    }
    for side in ("left", "right"):
        fields.update(
            {
                f"{side}_wrist_{axis}": np.linspace(0.1, 0.2, 5).astype("float32")
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            }
        )
    for name in ("action.parquet", "observation.state.parquet"):
        pq.write_table(pa.table(fields), data / name)
    return episode, output, mount


def approve(store):
    state = store.load(0)
    state["reviewer"] = "synthetic-test"
    for side, item in state["selected"].items():
        item.update(status="accepted", physical_side=side)
    from vive_correct.review_bundle import approve as approve_bundle

    approve_bundle(store, 0, state, Path(store.load(0)["episode"]).parent / "mount.json")
    return store.load(0)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode, self.output, self.mount = make_fixture(self.root)
        self.store = ReviewStore(self.output)

    def test_legacy_reviews_require_confirmation_and_preserve_corrections(self):
        quad = [[121.0, 262.0], [181.0, 262.0], [181.0, 298.0], [121.0, 298.0]]
        atomic_json(
            self.store.path(0).with_name("human_review.json"),
            dict(
                schema_version=1,
                observations=[
                    dict(
                        image_side="left", frame_index=2, status="accepted", corrected_quad_px=quad
                    )
                ],
            ),
        )
        loaded = self.store.load(0)
        self.assertTrue(loaded["needs_reconfirmation"])
        self.assertEqual(loaded["selected"]["left"]["quad"], quad)
        self.assertEqual(loaded["status"], "pending")

    def test_pending_reason_distinguishes_draft_from_stale_and_empty_output(self):
        state = self.store.save(0, self.store.load(0))
        self.assertFalse(state["needs_reconfirmation"])
        self.assertIn("左手、右手尚未确认", state["pending_reason"])
        state = approve(self.store)
        self.assertEqual(state["pending_reason"], "")
        path = self.store.path(0)
        path.write_text(path.read_text() + "\n")
        self.assertIn("已变化", self.store.load(0)["pending_reason"])
        for main in (repair_main, server_main):
            with self.assertRaises(SystemExit) as error:
                main(["--out", ""])
            self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(ValueError, str(self.root / "wrong-output")):
            ReviewStore(self.root / "wrong-output")

    def test_approve_resume_separate_frames_and_manual_replacement(self):
        state = approve(self.store)
        self.assertEqual(ReviewStore(self.output).load(0)["status"], "ready")
        state["selected"]["left"].update(frame_index=2, candidate_id=None)
        result = self.store.save(0, state)
        self.assertEqual(result["selected"]["left"]["source"], "manual")
        report = self.store.observation_report(0)
        self.assertEqual([o["frame_index"] for o in report["observations"]], [1, 2])
        report_path = self.root / "observations.json"
        atomic_json(report_path, report)
        alignment = build_alignment(self.episode, report_path, self.mount)
        self.assertEqual({s["source_rgb_frame"] for s in alignment["samples"]}, {1, 2})
        self.assertEqual(alignment["alignment_timeline"], "rectified_rgb")

    def test_rejected_drafts_and_missing_candidates_can_be_saved(self):
        path = self.store.path(0)
        report = read_json(path)
        report["features"] = {}
        atomic_json(path, report)
        state = self.store.load(0)
        state["selected"]["left"].update(quad=[[1.0, 2.0]], status="rejected")
        saved = self.store.save(0, state)
        self.assertEqual(saved["selected"]["left"]["status"], "rejected")
        self.assertEqual(saved["status"], "pending")

    def test_invalid_corners_identity_and_duplicate_labels_cannot_pass(self):
        state = approve(self.store)
        cases = [
            dict(quad=[[0, 0]] * 4),
            dict(quad=[[-1, 200], [20, 200], [20, 240], [0, 240]]),
            dict(physical_side="unknown"),
            dict(frame_index=200),
        ]
        for values in cases:
            with self.subTest(values=values):
                changed = deepcopy(state)
                changed["selected"]["left"].update(values)
                with self.assertRaises(ValueError):
                    self.store.save(0, changed)
        state["selected"]["right"] = dict(state["selected"]["left"], physical_side="right")
        with self.assertRaisesRegex(ValueError, "同一个标签"):
            self.store.save(0, state)

    def test_changed_inputs_and_concurrent_writes_are_rejected(self):
        state = approve(self.store)
        changed = deepcopy(state)
        changed["selected"]["left"]["notes"] = "new decision"
        self.store.save(0, changed)
        with self.assertRaisesRegex(ValueError, "已被修改"):
            self.store.save(0, state)
        path = self.episode / "calibrations/rectify_params.yaml"
        path.write_text(path.read_text() + "\n")
        self.assertEqual(self.store.load(0)["status"], "pending")
        with self.assertRaises(ValueError):
            self.store.observation_report(0)

    def test_sync_failure_is_explicit(self):
        approve(self.store)
        report = self.store.observation_report(0)
        path = self.root / "observations.json"
        atomic_json(path, report)
        source = self.episode / DATA / "observation.left_wrist_pose.parquet"
        table = pq.read_table(source)
        table = table.set_column(
            table.schema.get_field_index("x"), "x", pa.array([None] * 4, type=pa.float64())
        )
        pq.write_table(table, source)
        with self.assertRaisesRegex(ValueError, "synchronized left wrist"):
            build_alignment(self.episode, path, self.mount)

    def test_edit_invalidates_current_delivery_and_export_lock_blocks_save(self):
        state = approve(self.store)
        for p in ("batch/results/manifest.json", "rerun/manifest.json", "repair_manifest.json"):
            atomic_json(
                self.output / p, {"entries": [{"episode": str(self.episode), "status": "complete"}]}
            )
        state["selected"]["left"]["notes"] = "changed"
        with output_lock(self.store.path(0).parent):
            with self.assertRaisesRegex(ValueError, "稍后"):
                self.store.save(0, state)
        self.store.save(0, state)
        self.assertEqual(read_json(self.output / "batch/results/manifest.json")["entries"], [])
        self.assertEqual(
            read_json(self.output / "rerun/manifest.json")["entries"][0]["status"], "stale"
        )

    def test_corrected_parquet_preserves_nulls_other_fields_and_schema(self):
        path = self.episode / DATA / "observation.left_wrist_pose.parquet"
        table = pq.read_table(path).set_column(1, "x", pa.array([1.0, None, np.nan, 4.0]))
        pq.write_table(table, path)
        alignment = {
            "wrists": {
                s: {"model": "world", "offset_m": [0.1, 0.2, 0.3]} for s in ("left", "right")
            }
        }
        destination = self.root / "corrected"
        write_corrected_wrists(self.episode, destination, alignment)
        result = pq.read_table(destination / DATA / path.name)
        self.assertEqual(result.schema, table.schema)
        self.assertEqual(result["x"].null_count, 1)
        self.assertTrue(np.isnan(result["x"][2].as_py()))
        np.testing.assert_allclose(result["x"].to_numpy()[[0, 3]], [1.1, 4.1])
        np.testing.assert_allclose(result["y"].to_numpy(), [0.2, 0, 0, 0.2])
        for name in ("timestamp", "roll", "pitch", "yaw", "episode_index"):
            self.assertTrue(result[name].equals(table[name]))
        write_corrected_wrists(self.episode, destination, alignment)
        self.assertTrue(pq.read_table(destination / DATA / path.name)["y"].equals(result["y"]))

    def fake_render(self, command, **kwargs):
        pending = Path(command[command.index("--output") + 1])
        alignment = read_json(Path(command[command.index("--wrist-calibration") + 1]))
        self.assertEqual(alignment["wrists"]["left"]["model"], "wrist_local_rigid")
        pending.write_bytes(b"test-recording")
        atomic_json(
            pending.with_suffix(".json"),
            {"frames": 4, "skipped_missing_head_frames": [], "mode": "rgb_corrected"},
        )

    def test_repair_reuses_offsets_for_rrd_and_repeated_runs(self):
        approve(self.store)
        original = {
            s: (self.episode / DATA / f"observation.{s}_wrist_pose.parquet").read_bytes()
            for s in ("left", "right")
        }
        with (
            patch("vive_correct.rerun_export.subprocess.run", side_effect=self.fake_render),
            patch(
                "ego_data.vive_origin_alignment.build_alignment",
                side_effect=AssertionError("RRD must reuse alignment"),
            ),
        ):
            first = repair_dataset(self.output, self.mount)
            second = repair_dataset(self.output, self.mount)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "complete")
        self.assertTrue(second["entries"][0]["reused"])
        alignment = read_json(first["entries"][0]["alignment"])
        for side in ("left", "right"):
            source = self.episode / DATA / f"observation.{side}_wrist_pose.parquet"
            self.assertEqual(source.read_bytes(), original[side])
            a = pq.read_table(
                first["entries"][0]["corrected_files"][f"observation.{side}_wrist_pose.parquet"][
                    "path"
                ]
            )
            b = pq.read_table(
                second["entries"][0]["corrected_files"][f"observation.{side}_wrist_pose.parquet"][
                    "path"
                ]
            )
            self.assertTrue(a.equals(b))
            raw = pq.read_table(source)
            for i, axis in enumerate("xyz"):
                np.testing.assert_allclose(
                    a[axis].to_numpy() - raw[axis].to_numpy(),
                    alignment["wrists"][side]["offset_m"][i],
                )

    def test_pending_not_exported_and_render_failure_not_published(self):
        with patch("vive_correct.review_repair.export_recordings") as export:
            result = repair_dataset(self.output, self.mount)
            export.assert_not_called()
            self.assertEqual(result["entries"][0]["status"], "pending")
        approve(self.store)
        with patch(
            "vive_correct.rerun_export.subprocess.run", side_effect=OSError("render failed")
        ):
            result = repair_dataset(self.output, self.mount)
        self.assertEqual(result["entries"][0]["status"], "failed")
        self.assertFalse(list(self.output.rglob("recording.rrd")))
        self.assertEqual(read_json(self.output / "batch/results/manifest.json")["entries"], [])

    def test_one_bad_episode_does_not_block_other_episodes(self):
        _, other_output, _ = make_fixture(self.root / "other")
        source = other_output / "features/sample/features.json"
        atomic_json(self.output / "features/second/features.json", read_json(source))
        store = ReviewStore(self.output)
        for index in (0, 1):
            state = store.load(index)
            state["reviewer"] = "synthetic-test"
            for side, item in state["selected"].items():
                item.update(status="accepted", physical_side=side)
            from vive_correct.review_bundle import approve as approve_bundle

            approve_bundle(store, index, state, self.mount)
        # Tracking failure affects one episode, without changing its reviewed video.
        (self.episode / DATA / "observation.left_wrist_pose.parquet").unlink()
        with patch("vive_correct.rerun_export.subprocess.run", side_effect=self.fake_render):
            result = repair_dataset(self.output, self.mount)
        self.assertEqual([r["status"] for r in result["entries"]], ["failed", "complete"])
        self.assertEqual(len(read_json(self.output / "rerun/manifest.json")["entries"]), 1)

    def test_changed_review_blocks_standalone_rrd_reexport(self):
        from vive_correct.rerun_export import export_recordings

        state = approve(self.store)
        with patch("vive_correct.rerun_export.subprocess.run", side_effect=self.fake_render):
            result = repair_dataset(self.output, self.mount)
        state["selected"]["left"]["status"] = "rejected"
        self.store.save(0, state)
        with patch("vive_correct.rerun_export.subprocess.run") as render:
            self.assertEqual(export_recordings(Path(result["entries"][0]["run"]), self.mount), 1)
            render.assert_not_called()

    def test_cached_export_skips_renderer_and_invalidates_changed_inputs(self):
        state = approve(self.store)
        with patch(
            "vive_correct.rerun_export.subprocess.run", side_effect=self.fake_render
        ) as render:
            first = repair_dataset(self.output, self.mount)
            with patch("vive_correct.human_review.checked_pose") as pose:
                unchanged = self.store.save(0, self.store.load(0))
                pose.assert_not_called()
            self.assertEqual(state["revision"], unchanged["revision"])
            second = repair_dataset(self.output, self.mount)
            self.assertTrue(second["entries"][0]["reused"])
            self.assertEqual(first["entries"][0]["recording"], second["entries"][0]["recording"])
            self.assertEqual(render.call_count, 1)
            from vive_correct.viewer import recordings_at

            self.assertEqual(
                recordings_at(second["entries"][0]["run"]), [Path(first["entries"][0]["recording"])]
            )
            # A changed mount, tracking stream or damaged artifact must rebuild.
            mount = read_json(self.mount)
            mount["T_head_rectified_camera"][0][3] = 0.01
            atomic_json(self.mount, mount)
            self.assertEqual(repair_dataset(self.output, self.mount)["status"], "incomplete")
            approve(self.store)
            repair_dataset(self.output, self.mount)
            self.assertEqual(render.call_count, 2)
            source = self.episode / DATA / "observation.left_wrist_pose.parquet"
            table = pq.read_table(source)
            table = table.set_column(
                table.schema.get_field_index("x"), "x", pa.array(table["x"].to_numpy() + 0.01)
            )
            pq.write_table(table, source)
            self.assertEqual(repair_dataset(self.output, self.mount)["status"], "incomplete")
            approve(self.store)
            result = repair_dataset(self.output, self.mount)
            self.assertEqual(render.call_count, 3)
            Path(result["entries"][0]["recording"]).write_bytes(b"damaged")
            repair_dataset(self.output, self.mount)
            self.assertEqual(render.call_count, 4)
            state = self.store.load(0)
            state["selected"]["left"]["quad"][0][0] += 1
            self.store.save(0, state)
            self.assertEqual(repair_dataset(self.output, self.mount)["status"], "incomplete")
            approve(self.store)
            repair_dataset(self.output, self.mount)
            self.assertEqual(render.call_count, 5)

    def test_export_one_preserves_other_completed_results_even_on_failure(self):
        _, other, _ = make_fixture(self.root / "other")
        atomic_json(
            self.output / "features/second/features.json",
            read_json(other / "features/sample/features.json"),
        )
        store = ReviewStore(self.output)
        for i in (0, 1):
            payload = store.load(i)
            payload["reviewer"] = "test"
            for side, choice in payload["selected"].items():
                choice.update(status="accepted", physical_side=side)
            from vive_correct.review_bundle import approve as approve_bundle

            approve_bundle(store, i, payload, self.mount)
        with patch("vive_correct.rerun_export.subprocess.run", side_effect=self.fake_render):
            repair_dataset(self.output, self.mount, indices=[1])
        original = read_json(self.output / "rerun/manifest.json")["entries"]
        with patch("vive_correct.rerun_export.subprocess.run", side_effect=OSError("test failure")):
            result = repair_dataset(self.output, self.mount, indices=[0])
        self.assertEqual(result["entries"][0]["status"], "failed")
        self.assertEqual(read_json(self.output / "rerun/manifest.json")["entries"], original)
        self.assertEqual(len(read_json(self.output / "repair_manifest.json")["entries"]), 2)


class ServerTests(unittest.TestCase):
    setUp = ReviewTests.setUp

    def test_split_approval_does_not_render_and_old_endpoint_is_gone(self):
        from vive_correct.review_bundle import review_state

        server = ReviewServer(self.output, 0, self.mount)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        self.addCleanup(connection.close)
        headers = {"Content-Type": "application/json", "X-Review-Token": server.token}
        connection.request("POST", "/api/confirm-export?id=0", "{}", headers)
        response = connection.getresponse()
        self.assertEqual(response.status, 410)
        response.read()
        payload = self.store.load(0)
        payload["reviewer"] = "test"
        for side, item in payload["selected"].items():
            item.update(status="accepted", physical_side=side)
        with patch("vive_correct.review_server.export_approved") as export:
            started = perf_counter()
            connection.request("POST", "/api/approve?id=0", json.dumps(payload), headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 202)
            response.read()
            self.assertLess(perf_counter() - started, 1)
            server.executor.shutdown(wait=True)
            export.assert_not_called()
        self.assertEqual(server.jobs[0]["status"], "approved")
        self.assertTrue(review_state(self.store, 0, self.mount)["approved"])
        self.assertFalse(list(self.output.rglob("*.rrd")))

    def test_approval_jobs_are_independent_and_bounded(self):
        from vive_correct.review_bundle import approve as approve_bundle

        for name in ("second", "third"):
            _, other, _ = make_fixture(self.root / name)
            atomic_json(
                self.output / f"features/{name}/features.json",
                read_json(other / "features/sample/features.json"),
            )
        store = ReviewStore(self.output)
        server = ReviewServer(self.output, 0, self.mount, approve_jobs=2)
        self.addCleanup(server.server_close)
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()

        def slow_approve(store, index, payload, calibration):
            if index < 2:
                entered[index].set()
                if not release.wait(5):
                    raise ValueError("test timeout")
            return approve_bundle(store, index, payload, calibration)

        def payload(index):
            state = store.load(index)
            state["reviewer"] = "test"
            for side, item in state["selected"].items():
                item.update(status="accepted", physical_side=side)
            return state

        with patch("vive_correct.review_server.approve", side_effect=slow_approve):
            try:
                server.start_job(0, "approve", payload(0))
                self.assertTrue(entered[0].wait(2))
                server.save_review(2, store.load(2))
                with self.assertRaises(ValueError):
                    server.save_review(0, store.load(0))
                server.start_job(1, "approve", payload(1))
                self.assertTrue(entered[1].wait(2))
                server.start_job(2, "approve", payload(2))
                self.assertEqual(server.jobs[2]["status"], "queued")
            finally:
                release.set()
                server.executor.shutdown(wait=True)
        self.assertEqual([server.jobs[i]["status"] for i in range(3)], ["approved"] * 3)

    def test_export_does_not_block_another_episode_review(self):
        from vive_correct.review_bundle import approve as approve_bundle

        _, other, _ = make_fixture(self.root / "second")
        atomic_json(
            self.output / "features/second/features.json",
            read_json(other / "features/sample/features.json"),
        )
        store = ReviewStore(self.output)
        state = store.load(0)
        state["reviewer"] = "test"
        for side, item in state["selected"].items():
            item.update(status="accepted", physical_side=side)
        approve_bundle(store, 0, state, self.mount)
        server = ReviewServer(self.output, 0, self.mount)
        self.addCleanup(server.server_close)
        entered, release = threading.Event(), threading.Event()
        approved = threading.Event()

        def export(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise ValueError("test timeout")
            return dict(status="complete")

        def second_approve(*args, **kwargs):
            result = approve_bundle(*args, **kwargs)
            approved.set()
            return result

        with (
            patch("vive_correct.review_server.export_approved", side_effect=export),
            patch("vive_correct.review_server.approve", side_effect=second_approve),
        ):
            try:
                server.start_job(0, "export")
                self.assertTrue(entered.wait(2))
                server.save_review(1, store.load(1))
                state = store.load(1)
                state["reviewer"] = "test"
                for side, item in state["selected"].items():
                    item.update(status="accepted", physical_side=side)
                server.start_job(1, "approve", state)
                self.assertTrue(approved.wait(3))
                self.assertTrue(server.jobs[0]["running"])
            finally:
                release.set()
                server.executor.shutdown(wait=True)
        self.assertEqual(server.jobs[1]["status"], "approved")

    def test_failed_approval_keeps_draft_without_acceptance(self):
        server = ReviewServer(self.output, 0, self.mount)
        self.addCleanup(server.server_close)
        payload = self.store.load(0)
        payload["reviewer"] = "test"
        for side, item in payload["selected"].items():
            item.update(status="accepted", physical_side=side)
        payload["selected"]["right"]["quad"] = [[0, 0]] * 4
        server.start_job(0, "approve", payload)
        server.executor.shutdown(wait=True)
        self.assertEqual(server.jobs[0]["status"], "failed")
        state = self.store.load(0)
        self.assertEqual(state["selected"]["right"]["quad"], [[0, 0]] * 4)
        self.assertEqual(state["status"], "pending")
        self.assertFalse(server.jobs[0]["review"]["approved"])

    def test_http_frame_review_and_port_conflict(self):
        self.assertEqual(DEFAULT_PORT, 18763)
        server = ReviewServer(self.output, 0, self.mount)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_port
        with self.assertRaises(OSError):
            ReviewServer(self.output, port, self.mount)
        conn = http.client.HTTPConnection("127.0.0.1", port)
        self.addCleanup(conn.close)
        conn.request("GET", "/api/episode?id=0")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        payload = json.loads(response.read())
        self.assertEqual(payload["frame_count"], 4)
        self.assertTrue(
            all(
                isinstance(v, str)
                for sig in payload["binding"]["input_signature"].values()
                if sig is not None
                for v in sig
            )
        )
        conn.request("GET", "/api/frame?id=0&frame=2")
        response = conn.getresponse()
        image = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (400, 640, 3))
        conn.request(
            "POST", "/api/review?id=0", json.dumps(payload), {"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        payload["reviewer"] = "synthetic-test"
        for side, item in payload["selected"].items():
            item.update(status="accepted", physical_side=side)
        conn.request(
            "POST",
            "/api/review?id=0",
            json.dumps(payload),
            {"Content-Type": "application/json", "X-Review-Token": server.token},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(response.read())["approved"])
        with patch(
            "vive_correct.review_server.ReviewServer", side_effect=OSError("Address in use")
        ):
            with self.assertRaises(SystemExit) as error:
                server_main(["--out", str(self.output), "--port", "18765"])
            self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
