"""Whole-episode rejection must persist and gate every supported downstream entry."""

import http.client
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_human_review import make_fixture

from vive_correct.batch.report_io import digest
from vive_correct.dataset import VIDEO_SUFFIX
from vive_correct.human_review import ReviewStore, verify_provenance
from vive_correct.review_actions import export_approved, rebuild
from vive_correct.review_bundle import approve, load_bundle, stage_files
from vive_correct.review_exclusions import FILENAME
from vive_correct.review_io import atomic_json, read_json
from vive_correct.review_repair import repair_dataset
from vive_correct.review_server import ReviewServer
from vive_correct.source_data import DATA, FILES
from vive_correct.task_locks import reserve


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode, self.output, self.mount = make_fixture(self.root)
        self.store = ReviewStore(self.output)

    def payload(self, index=0):
        state = self.store.load(index)
        return dict(state, reviewer="reviewer", reason="标签严重遮挡")

    def approve(self, index=0):
        payload = self.payload(index)
        for side, choice in payload["selected"].items():
            choice.update(status="accepted", physical_side=side)
        return approve(self.store, index, payload, self.mount)

    def test_missing_video_can_be_excluded_and_record_is_idempotent(self):
        (self.episode / VIDEO_SUFFIX).unlink()
        payload = self.payload()
        with patch("vive_correct.human_review.checked_pose") as geometry:
            state = self.store.reject_episode(0, payload)
        geometry.assert_not_called()
        self.assertEqual(state["status"], "excluded")
        path = self.output / FILENAME
        first = path.read_bytes()
        row = read_json(path)["entries"][0]
        self.assertEqual(row["episode"], str(self.episode))
        self.assertEqual(row["video"], str(self.episode / VIDEO_SUFFIX))
        self.assertEqual(row["features_file"], str(self.store.path(0)))
        self.assertEqual(row["reason"], "标签严重遮挡")
        self.assertEqual(row["reviewer"], "reviewer")
        self.assertTrue(row["rejected_at"])
        self.store.reject_episode(0, payload)
        self.assertEqual(path.read_bytes(), first)
        self.assertEqual(ReviewStore(self.output).load(0)["status"], "excluded")
        # Updating automatic candidates cannot silently restore an excluded episode.
        report = read_json(self.store.path(0))
        report["error"] = "new scan"
        atomic_json(self.store.path(0), report)
        self.assertEqual(self.store.load(0)["status"], "excluded")

    def test_validation_and_reserved_jobs_leave_no_rejection(self):
        payload = self.payload()
        for change in ({"reviewer": " "}, {"revision": "stale"}, {"binding": {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.store.reject_episode(0, dict(payload, **change))
        reservation = reserve(self.store.path(0).parent)
        try:
            with self.assertRaisesRegex(ValueError, "正在处理"):
                self.store.reject_episode(0, payload)
        finally:
            reservation.close()
        self.assertFalse((self.output / FILENAME).exists())

    def test_single_side_rejection_remains_a_draft(self):
        state = self.payload()
        state["selected"]["left"]["status"] = "rejected"
        result = self.store.save(0, state)
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["exclusion"])
        self.assertFalse((self.output / FILENAME).exists())

    def test_exclusion_survives_failure_to_update_old_display_manifest(self):
        with patch.object(self.store, "invalidate", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.store.reject_episode(0, self.payload())
        self.assertEqual(ReviewStore(self.output).load(0)["status"], "excluded")
        with self.assertRaisesRegex(ValueError, "不通过"):
            load_bundle(self.store, 0, self.mount)

    def test_previously_approved_episode_cannot_reuse_exports_or_change_source(self):
        bundle = self.approve()
        stage_files(bundle)  # Exercise the cached-Parquet path too.
        before = {name: digest(self.episode / DATA / name) for name in FILES}
        for relative in (
            "repair_manifest.json",
            "rerun/manifest.json",
            "batch/results/manifest.json",
        ):
            atomic_json(
                self.output / relative,
                {"entries": [{"episode": str(self.episode), "status": "complete"}]},
            )
        self.store.reject_episode(0, self.payload())
        for operation in (
            lambda: load_bundle(self.store, 0, self.mount),
            lambda: stage_files(bundle),
            lambda: export_approved(self.store, 0, self.mount),
            lambda: verify_provenance(read_json(bundle["report"])),
            lambda: self.store.save(0, self.payload()),
            lambda: self.approve(),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "不通过"):
                operation()
        with (
            patch("vive_correct.review_actions.stage_files") as prepare,
            patch("vive_correct.review_actions.export_recordings") as render,
        ):
            exported = repair_dataset(self.output, self.mount)
            for in_place in (False, True):
                result = rebuild(self.output, self.mount, in_place=in_place)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["entries"][0]["status"], "excluded")
        prepare.assert_not_called()
        render.assert_not_called()
        self.assertEqual(exported["entries"][0]["status"], "excluded")
        self.assertEqual(before, {name: digest(self.episode / DATA / name) for name in FILES})
        self.assertFalse(list(self.episode.rglob("*.bak")))
        self.assertEqual(
            read_json(self.output / "rerun/manifest.json")["entries"][0]["status"], "stale"
        )
        self.assertEqual(read_json(self.output / "batch/results/manifest.json")["entries"], [])

    def add_episode(self):
        episode, output, _ = make_fixture(self.root / "second")
        source = next((output / "features").rglob("features.json"))
        atomic_json(self.output / "features/second/features.json", read_json(source))
        self.store.refresh()
        return next(
            i for i in range(len(self.store.paths)) if self.store.load(i)["episode"] == str(episode)
        )

    def test_concurrent_rejections_do_not_lose_paths(self):
        other = self.add_episode()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda i: self.store.reject_episode(i, self.payload(i)), [0, other])
            )
        self.assertTrue(all(r["status"] == "excluded" for r in results))
        self.assertEqual(len(read_json(self.output / FILENAME)["entries"]), 2)

    def test_mixed_batch_skips_only_rejected_episode(self):
        other = self.add_episode()
        self.approve(other)
        self.store.reject_episode(0, self.payload())
        with patch(
            "vive_correct.review_actions.export_approved", return_value={"status": "complete"}
        ) as render:
            result = repair_dataset(self.output, self.mount)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(render.call_count, 1)
        self.assertEqual(render.call_args.args[1], other)
        result = rebuild(self.output, self.mount)
        self.assertEqual([row["status"] for row in result["entries"]], ["excluded", "prepared"])

    def test_http_rejection_auth_restart_and_cached_state(self):
        other = self.add_episode()
        self.approve(other)
        server = ReviewServer(self.output, 0, self.mount)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        self.addCleanup(conn.close)
        headers = {"Content-Type": "application/json", "X-Review-Token": server.token}
        payload = self.payload()
        server.state(0, full=True)
        self.assertTrue(server.state(other, full=True)["approved"])
        version = server.list_version()

        def request(path, body, auth=headers):
            conn.request("POST", path, json.dumps(body), auth)
            response = conn.getresponse()
            return response.status, json.loads(response.read())

        self.assertEqual(request("/api/reject?id=0", payload, {})[0], 403)
        code, state = request("/api/reject?id=0", payload)
        self.assertEqual(code, 200)
        self.assertEqual(state["status"], "excluded")
        self.assertFalse(state["approved"])
        self.assertNotEqual(version, server.list_version())
        self.assertEqual(server.state(0)["status"], "excluded")
        self.assertEqual(server.summary(0)["status"], "excluded")
        self.assertTrue(server.state(other)["approved"])
        for endpoint in ("review", "approve", "export"):
            self.assertEqual(request(f"/api/{endpoint}?id=0", payload)[0], 400)
        self.assertFalse(server.jobs)
        self.assertEqual(ReviewStore(self.output).load(0)["status"], "excluded")


if __name__ == "__main__":
    unittest.main()
