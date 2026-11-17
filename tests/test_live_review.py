"""Live publication, stable IDs, bounded scheduling and cache integrity."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_human_review import make_fixture
from vive_correct.cv_features import run_feature_dataset
from vive_correct.estimate.cv_config import CVConfig
from vive_correct.feature_progress import scan_progress
from vive_correct.human_review import ReviewStore, decode_frame
from vive_correct.review_io import atomic_json, output_lock, read_json
from vive_correct.review_server import ReviewServer
from vive_correct.review_repair import repair_dataset
from vive_correct.runtime_cache import CACHE, MemoryCache, cached_file
from vive_correct.task_locks import operation_lock


class LiveReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode, self.fixture_output, self.mount = make_fixture(self.root / "fixture")
        self.output = self.root / "live"
        self.report = read_json(self.fixture_output / "features/sample/features.json")

    def publish(self, name, report=None):
        path = self.output / "features" / name / "features.json"
        atomic_json(path, report or self.report)
        return path

    def server(self):
        server = ReviewServer(self.output, 0, self.mount)
        self.addCleanup(server.server_close)
        return server

    def test_cache_merges_misses_without_serializing_other_keys(self):
        cache = MemoryCache()
        entered, release = threading.Event(), threading.Event()
        calls = []

        def slow():
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(3))
            return [1]

        with ThreadPoolExecutor(4) as pool:
            first = pool.submit(cache.get, "same", slow)
            self.assertTrue(entered.wait(3))
            second = pool.submit(cache.get, "same", slow)
            self.assertEqual(pool.submit(cache.get, "other", lambda: [2]).result(3), [2])
            release.set()
            a, b = first.result(3), second.result(3)
        a.append(9)
        self.assertEqual(b, [1])
        self.assertEqual(calls, [1])
        with self.assertRaisesRegex(ValueError, "retry"):
            cache.get("failure", lambda: (_ for _ in ()).throw(ValueError("retry")))
        self.assertEqual(cache.get("failure", lambda: 42), 42)

    def test_frame_encoding_is_shared_and_invalidated(self):
        import cv2
        from vive_correct.dataset import VIDEO_SUFFIX

        self.publish("sample")
        server = self.server()
        with patch("vive_correct.review_server.cv2.imencode", wraps=cv2.imencode) as encode:
            with ThreadPoolExecutor(4) as pool:
                frames = list(pool.map(lambda _: server.frame_png(0, 0), range(4)))
            self.assertEqual(encode.call_count, 1)
            self.assertTrue(all(frame == frames[0] for frame in frames))
            (self.episode / VIDEO_SUFFIX).touch()
            self.assertEqual(server.frame_png(0, 0), frames[0])
            self.assertEqual(encode.call_count, 2)

    def test_preview_concurrency_limit_and_cache_hit_bypass(self):
        from vive_correct.review_server import preview as calculate

        self.publish("sample")
        server = ReviewServer(self.output, 0, self.mount, preview_jobs=2)
        self.addCleanup(server.server_close)
        choices = server.store.load(0)["selected"]
        left = dict(choices["left"], side="left")
        right = dict(choices["right"], side="right")
        cached = server.axis_preview(0, left)
        barrier, release = threading.Barrier(3), threading.Event()

        def blocked(*args):
            barrier.wait(timeout=5)
            self.assertTrue(release.wait(5))
            return calculate(*args)

        with patch("vive_correct.review_server.preview", side_effect=blocked):
            with ThreadPoolExecutor(3) as pool:
                first = pool.submit(server.axis_preview, 0, right)
                second = pool.submit(server.axis_preview, 0, dict(right, frame_index=2))
                try:
                    barrier.wait(timeout=5)
                    hot = pool.submit(
                        server.axis_preview, 0, dict(left, axis_angles_deg=[10, 0, 0])
                    ).result(3)
                    self.assertEqual(hot["origin_camera_m"], cached["origin_camera_m"])
                finally:
                    release.set()
                first.result(5)
                second.result(5)

    def refresh(self, server):
        server.last_refresh = 0
        server.refresh()

    def test_empty_start_out_of_order_publication_and_restart_keep_ids(self):
        server = self.server()
        self.assertEqual(server.store.paths, [])
        initial = server.list_version()
        z = self.publish("z")
        self.refresh(server)
        self.assertNotEqual(initial, server.list_version())
        draft = server.store.load(0)
        draft["reviewer"] = "keep"
        draft["selected"]["left"]["notes"] = "draft before arrival"
        server.save_review(0, draft)
        a = self.publish("a")
        self.refresh(server)
        self.assertEqual(server.store.paths, [z, a])
        self.assertEqual(server.store.load(0)["selected"]["left"]["notes"], "draft before arrival")
        reopened = ReviewStore(self.output)
        self.assertEqual(reopened.paths, [z, a])
        # A missing report must not let a later episode take its identifier.
        z.unlink()
        b = self.publish("b")
        reopened.refresh()
        self.assertEqual(reopened.paths, [z, a, b])
        self.assertTrue(server.summary(0)["unavailable"])

    def test_no_partial_publication_and_legacy_sorted_import(self):
        report = deepcopy(self.report)
        image = self.output / "features/z/left.jpg"
        report.update(publication_version=1, feature_images={"left": str(image)})
        z = self.publish("z", report)
        store = ReviewStore(self.output, allow_empty=True)
        self.assertEqual(store.paths, [])
        image.touch()
        store.refresh()
        self.assertEqual(store.paths, [])
        atomic_json(z.with_name("candidate_pool.json"), [])
        a = self.publish("a")
        store.refresh()
        self.assertEqual(store.paths, [a, z])

    def test_http_catalog_updates_without_deep_poll_validation(self):
        server = self.server()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        self.addCleanup(conn.close)

        def get(path):
            conn.request("GET", path)
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            return json.loads(response.read())

        self.assertEqual(get("/api/episodes"), [])
        version = get("/api/status")["list_version"]
        published = time.monotonic()
        self.publish("new")
        deadline = published + 2
        while time.monotonic() < deadline:
            if get("/api/status")["list_version"] != version:
                break
            time.sleep(0.05)
        else:
            self.fail("publication did not appear within 2s")
        with patch(
            "vive_correct.review_server.review_state", side_effect=AssertionError("heavy poll")
        ):
            rows = get("/api/episodes")
            self.assertEqual(len(rows), 1)
        state = get("/api/episode?id=0")
        self.assertEqual(state["frame_count"], 4)
        with patch(
            "vive_correct.review_server.review_state", side_effect=AssertionError("heavy poll")
        ):
            self.assertEqual(get("/api/review-state?id=0")["revision"], state["revision"])

    def test_two_exports_do_not_occupy_approval_queue_and_reserve_queued_episode(self):
        for name in ("a", "b", "c"):
            self.publish(name)
        server = self.server()
        entered, release, approved = threading.Event(), threading.Event(), threading.Event()

        def export(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(10))
            return dict(status="complete")

        def approve(*args, **kwargs):
            approved.set()

        with (
            patch("vive_correct.review_server.export_approved", side_effect=export),
            patch("vive_correct.review_server.approve", side_effect=approve),
        ):
            try:
                server.start_job(0, "export")
                self.assertTrue(entered.wait(2))
                server.start_job(1, "export")
                self.assertEqual(server.jobs[1]["status"], "queued")
                with self.assertRaisesRegex(ValueError, "正在处理"):
                    server.save_review(1, server.store.load(1))
                # A separate process cannot mutate a queued browser episode.
                code = "from vive_correct.task_locks import operation_lock\nimport sys\nwith operation_lock(sys.argv[1]): pass"
                result = subprocess.run(
                    [sys.executable, "-c", code, str(server.store.path(1).parent)],
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                server.save_review(2, server.store.load(2))
                server.start_job(2, "approve", server.store.load(2))
                self.assertTrue(approved.wait(2))
                self.assertTrue(server.jobs[0]["running"])
            finally:
                release.set()
                server.executor.shutdown(wait=True)
                server.export_executor.shutdown(wait=True)
        self.assertIn("queue_s", server.jobs[1])
        self.assertIn("elapsed_s", server.jobs[2])

    def test_cli_export_is_bounded_and_one_failure_does_not_abort_batch(self):
        for name in ("a", "b", "c"):
            self.publish(name)
        store = ReviewStore(self.output)
        for index in range(3):
            payload = store.load(index)
            payload["reviewer"] = "parallel-export-test"
            for side, choice in payload["selected"].items():
                choice.update(status="accepted", physical_side=side)
            store.save(index, payload)
        gate = threading.Barrier(2)
        guard = threading.Lock()
        active = peak = 0

        def export(store, index, *args):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            try:
                if index < 2:
                    gate.wait(timeout=5)
                if index == 1:
                    raise ValueError("render failed")
                return dict(index=index, status="complete")
            finally:
                with guard:
                    active -= 1

        with patch("vive_correct.review_actions.export_approved", side_effect=export):
            result = repair_dataset(self.output, self.mount, export_jobs=2)
        self.assertEqual(peak, 2)
        self.assertEqual(
            [r["status"] for r in result["entries"]], ["complete", "failed", "complete"]
        )
        store.path(1).unlink()
        with patch(
            "vive_correct.review_actions.export_approved", return_value=dict(status="complete")
        ):
            result = repair_dataset(self.output, self.mount, export_jobs=2)
        self.assertEqual(
            [r["status"] for r in result["entries"]], ["complete", "failed", "complete"]
        )

    def test_restart_marks_unfinished_jobs_interrupted_without_resubmission(self):
        path = self.publish("a")
        atomic_json(
            path.with_name("review_job.json"),
            dict(index=0, kind="export", status="rendering", running=True),
        )
        with patch("vive_correct.review_server.export_approved") as export:
            server = self.server()
            self.assertEqual(server.jobs[0]["status"], "interrupted")
            self.assertFalse(server.jobs[0]["running"])
            export.assert_not_called()
        server.save_review(0, server.store.load(0))
        self.assertFalse(path.with_name("review_job.json").exists())

    def test_interrupted_scanner_and_single_writer(self):
        self.output.mkdir()
        atomic_json(
            self.output / "feature_progress.json",
            dict(
                run_id="one",
                status="running",
                entries=[dict(episode=str(self.episode), folder="features/a", status="queued")],
            ),
        )
        atomic_json(
            self.output / "features/a/scan_status.json", dict(run_id="one", status="scanning")
        )
        with output_lock(self.output, name=".feature_scan.lock"):
            self.assertEqual(scan_progress(self.output)["entries"][0]["status"], "scanning")
            with self.assertRaises(ValueError):
                with output_lock(self.output, name=".feature_scan.lock"):
                    pass
        self.assertEqual(scan_progress(self.output)["entries"][0]["status"], "interrupted")

    def test_resume_requires_candidate_pool_and_preserves_reviewed_results(self):
        args = SimpleNamespace(
            episodes_file=None,
            coarse_stride=15,
            feature_search="fast",
            max_frames=None,
            dry_run=False,
            jobs=1,
        )
        folder = self.output / "features" / self.episode.relative_to(self.root)
        report = dict(
            self.report,
            schema_version=2,
            publication_version=1,
            features={},
            search_mode="fast",
            mode="same_frame",
            stop_reason="test",
            seconds=0.1,
            feature_images={},
            error=None,
        )
        atomic_json(folder / "features.json", report)

        def worker(*args):
            atomic_json(folder / "candidate_pool.json", {})
            return report

        with (
            patch("vive_correct.cv_features.select_episodes", return_value=[self.episode]),
            patch("vive_correct.cv_features.ProcessPoolExecutor", ThreadPoolExecutor),
            patch("vive_correct.cv_features.render_features"),
            patch("vive_correct.cv_features._scan_published", side_effect=worker) as scan,
        ):
            self.assertEqual(run_feature_dataset(args, self.root, self.output, CVConfig.load()), 0)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(run_feature_dataset(args, self.root, self.output, CVConfig.load()), 0)
            self.assertEqual(scan.call_count, 1)
            atomic_json(folder / "human_review.json", dict(schema_version=3))
            (folder / "candidate_pool.json").unlink()
            self.assertEqual(run_feature_dataset(args, self.root, self.output, CVConfig.load()), 1)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(read_json(folder / "features.json"), report)

    def test_worker_failure_does_not_abort_other_episodes(self):
        second, _, _ = make_fixture(self.root / "second")
        args = SimpleNamespace(
            episodes_file=None,
            coarse_stride=15,
            feature_search="fast",
            max_frames=None,
            dry_run=False,
            jobs=2,
        )
        original_report = dict(
            self.report,
            mode="same_frame",
            stop_reason="test",
            seconds=0.1,
            feature_images={},
            error=None,
            video_space="rectified",
            status="features_ready",
        )

        def worker(ep, folder, *args):
            if ep == self.episode:
                raise RuntimeError("worker crashed")
            report = dict(original_report, episode=str(ep))
            atomic_json(folder / "features.json", report)
            return report

        with (
            patch("vive_correct.cv_features.select_episodes", return_value=[self.episode, second]),
            patch("vive_correct.cv_features.ProcessPoolExecutor", ThreadPoolExecutor),
            patch("vive_correct.cv_features._scan_published", side_effect=worker),
            patch("vive_correct.cv_features.render_features"),
        ):
            self.assertEqual(run_feature_dataset(args, self.root, self.output, CVConfig.load()), 1)
        progress = scan_progress(self.output)
        self.assertEqual([r["status"] for r in progress["entries"]], ["failed", "complete"])
        self.assertEqual(len(ReviewStore(self.output).paths), 1)


class CacheTests(unittest.TestCase):
    def test_lru_budget_copy_isolation_and_file_replacement_invalidation(self):
        cache = MemoryCache(limit=3000)
        array = np.zeros(100)
        first = cache.get("a", lambda: array)
        first[0] = 5
        self.assertEqual(cache.get("a", lambda: None)[0], 0)
        cache.get("b", lambda: np.ones(100))
        self.assertLessEqual(cache.size, cache.limit)
        self.assertNotIn("a", cache.values)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_text("one")
            first_stat = path.stat()
            self.assertEqual(cached_file("test", path, path.read_text), "one")
            replacement = path.with_suffix(".new")
            replacement.write_text("two")
            os.utime(replacement, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
            replacement.replace(path)
            self.assertEqual(cached_file("test", path, path.read_text), "two")

    def test_frame_cache_matches_cold_decode(self):
        with TemporaryDirectory() as directory:
            episode, _, _ = make_fixture(Path(directory))
            CACHE.clear()
            cold, count = decode_frame(episode, 1)
            with patch(
                "vive_correct.human_review._decode_frame",
                side_effect=AssertionError("decoded twice"),
            ):
                warm, warm_count = decode_frame(episode, 1)
            np.testing.assert_array_equal(cold, warm)
            self.assertEqual(count, warm_count)

    def test_operation_lock_is_reentrant_but_not_shared_between_threads(self):
        with TemporaryDirectory() as directory:
            with operation_lock(directory):
                with operation_lock(directory):
                    pass
                with ThreadPoolExecutor(1) as pool:

                    def attempt():
                        with operation_lock(directory):
                            pass

                    with self.assertRaises(ValueError):
                        pool.submit(attempt).result()
