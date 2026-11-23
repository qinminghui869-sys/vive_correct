"""Final-result AI suggestions stay bounded, geometric, and separate from human approval."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from test_human_review import make_fixture
from vive_correct import ai_final_review as ai
from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.openai_vision import OpenAIConfig
from vive_correct.human_review import ReviewStore
from vive_correct.review_io import atomic_json, read_json
from vive_correct.review_server import ReviewServer


def decision(box="ok", axis="keep", steps=None):
    return dict(box=box, region=None, axis=axis, rotations=steps or [], reason="检查依据")


def pose(angles=None, ambiguous=False):
    return dict(
        corner_correction=dict(axis_angles_deg=angles or [10, 20, 30], plane_ambiguous=ambiguous)
    )


class FinalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode, self.output, self.mount = make_fixture(self.root)
        self.store = ReviewStore(self.output)
        self.path = self.store.path(0)
        self.original = self.store.load(0)["selected"]
        self.config = OpenAIConfig("fake-secret", "https://unused.invalid", "test-vision", "test")
        self.cv = CVConfig.load()

    def run_worker(self, reply=None, **kwargs):
        reply = reply or {"decisions": [dict(side=s, **decision()) for s in ("left", "right")]}
        transport = Mock(return_value=json.dumps(reply))
        ai.run_review(self.path, self.mount, self.cv, self.config, transport=transport, **kwargs)
        return transport, read_json(self.path.with_name("ai_review.json"))

    def test_valid_no_change_is_one_call_and_never_human_approval(self):
        before = self.path.read_bytes()
        transport, report = self.run_worker()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["calls"], 1)
        self.assertFalse(report["adjusted"])
        self.assertFalse(report["human_verified"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.path.with_name("human_review.json").exists())
        self.assertFalse(self.path.with_name("approval.json").exists())
        self.assertEqual(self.store.load(0)["status"], "pending")
        content = transport.call_args.args[1]["messages"][0]["content"]
        self.assertEqual(sum(c["type"] == "image_url" for c in content), 1)
        metadata = json.loads(content[1]["text"].split("：", 1)[1])
        self.assertEqual(metadata["image_size_px"], [1600, 1026])
        self.assertEqual(len(metadata["rows"]), 2)
        self.assertNotIn("fake-secret", json.dumps(report))

    def test_cache_input_and_model_binding(self):
        self.run_worker()
        transport, report = self.run_worker()
        transport.assert_not_called()
        self.assertTrue(ai.load_advice(self.path, self.mount)["valid"])
        mount = read_json(self.mount)
        mount["head_euler_sequence"] = "zyx"
        atomic_json(self.mount, mount)
        self.assertFalse(ai.load_advice(self.path, self.mount)["valid"])
        transport, report = self.run_worker()
        self.assertEqual(transport.call_count, 1)
        self.config = OpenAIConfig("fake-secret", "https://unused.invalid", "other-model", "test")
        transport, report = self.run_worker()
        self.assertEqual(transport.call_count, 1)

    def test_failure_has_no_retry_and_does_not_block_manual_review(self):
        transport = Mock(side_effect=TimeoutError("fake-secret timeout"))
        ai.run_review(self.path, self.mount, self.cv, self.config, transport=transport)
        report = read_json(self.path.with_name("ai_review.json"))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["calls"], 1)
        self.assertEqual(transport.call_count, 1)
        self.assertNotIn("fake-secret", report["error"])
        state = self.store.load(0)
        self.store.save(0, dict(state, reviewer="human"))
        self.assertTrue(self.path.with_name("human_review.json").exists())

    def test_saved_human_draft_is_not_changed_and_stale_ai_reference_is_rejected(self):
        state = self.store.load(0)
        state["selected"]["left"]["axis_angles_deg"] = [25, 90, 10]
        self.store.save(0, dict(state, reviewer="human"))
        human_path = self.path.with_name("human_review.json")
        before = human_path.read_bytes()
        self.run_worker()
        self.assertEqual(before, human_path.read_bytes())
        self.assertTrue(self.store.load(0)["has_human_review"])
        advice = ai.load_advice(self.path, self.mount)
        payload = dict(self.store.load(0), ai_reference=advice["revision"])
        payload["selected"]["left"]["notes"] = "adopted suggestion"
        self.store.save(0, payload)
        self.assertEqual(self.store.load(0)["ai_reference"], advice["revision"])
        payload = self.store.load(0)
        report = read_json(self.path)
        report["features"]["left"]["frame_index"] = 3
        atomic_json(self.path, report)
        payload.update(binding=self.store.binding(0))
        with self.assertRaisesRegex(ValueError, "AI 建议"):
            self.store.save(0, payload)

    def test_box_change_uses_new_pnp_and_followup_never_redetects(self):
        revised = deepcopy(self.original["left"])
        revised["quad"][0][0] += 2
        ask = Mock(
            side_effect=[
                {
                    "left": decision("adjust", "rotate", [{"axis": "X", "degrees": 90}]),
                    "right": decision(),
                },
                {"left": decision("adjust"), "right": decision()},
            ]
        )
        with (
            patch.object(ai, "geometry", return_value=(pose(), None)),
            patch.object(ai, "redetect", return_value=revised) as redetect,
        ):
            selected, rounds, issues = ai.review_choices(
                self.episode, self.original, self.mount, self.cv, ask
            )
        self.assertEqual(ask.call_count, 2)
        redetect.assert_called_once()
        self.assertEqual(selected["left"]["quad"], revised["quad"])
        # First-round rotation was based on the old corners, so only the new PnP baseline is used.
        np.testing.assert_allclose(selected["left"]["axis_angles_deg"], [10, 20, 30])
        self.assertIn("人工", issues["left"])
        self.assertEqual(len(rounds), 2)

    def test_redetection_failure_keeps_original(self):
        ask = Mock(return_value={"left": decision("adjust"), "right": decision()})
        with (
            patch.object(ai, "geometry", return_value=(pose(), None)),
            patch.object(ai, "redetect", side_effect=ValueError("no valid label")),
        ):
            selected, _, issues = ai.review_choices(
                self.episode, self.original, self.mount, self.cv, ask
            )
        self.assertEqual(selected, self.original)
        self.assertEqual(ask.call_count, 1)
        self.assertIn("left", issues)

    def test_axis_change_is_two_calls_at_most_and_keeps_corners(self):
        ask = Mock(
            side_effect=[
                {
                    "left": decision(axis="rotate", steps=[{"axis": "Y", "degrees": 90}]),
                    "right": decision(),
                },
                {
                    "left": decision(axis="rotate", steps=[{"axis": "Z", "degrees": 180}]),
                    "right": decision(),
                },
            ]
        )
        with patch.object(ai, "geometry", return_value=(pose(), None)):
            selected, _, _ = ai.review_choices(
                self.episode, self.original, self.mount, self.cv, ask
            )
        self.assertEqual(ask.call_count, 2)
        for s in ("left", "right"):
            self.assertEqual(selected[s]["quad"], self.original[s]["quad"])
            self.assertEqual(selected[s]["frame_index"], self.original[s]["frame_index"])

    def test_missing_or_ambiguous_pnp_never_rotates(self):
        for p in (None, pose(ambiguous=True)):
            ask = Mock(return_value={"left": decision(axis="pnp"), "right": decision()})
            with patch.object(ai, "geometry", return_value=(p, "missing" if p is None else None)):
                selected, _, issues = ai.review_choices(
                    self.episode, self.original, self.mount, self.cv, ask
                )
            self.assertEqual(selected, self.original)
            self.assertEqual(ask.call_count, 1)
            self.assertIn("left", issues)

    def test_local_redetection_requires_pixel_filter_and_real_pnp(self):
        candidate = dict(quad=np.asarray(self.original["left"]["quad"]), score=1)
        with (
            patch(
                "vive_correct.estimate.label_detection.detect_label_candidates",
                return_value={"candidates": {"left": [candidate]}},
            ) as detector,
            patch("vive_correct.cv_features.feature_rejection_reason", return_value=None),
        ):
            revised = ai.redetect(
                self.episode, "left", self.original["left"], [100, 500, 400, 900], self.cv
            )
            self.assertEqual(revised["quad"], candidate["quad"].tolist())
            self.assertIsNone(revised["candidate_id"])
            self.assertEqual(detector.call_args.args[1], {"left": [64, 200, 256, 360]})
            candidate["quad"] = np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
            with self.assertRaisesRegex(ValueError, "PnP"):
                ai.redetect(self.episode, "left", self.original["left"], None, self.cv)

    def test_async_queue_does_not_hold_human_save_lock(self):
        entered, release = threading.Event(), threading.Event()
        args = SimpleNamespace(
            ai_final_review=True,
            initial_mount=self.mount,
            model_timeout=1,
            transport="api",
            model=None,
            model_config=None,
            model_jobs=1,
        )

        def delayed(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))

        with (
            patch.object(ai, "load_config", return_value=self.config),
            patch.object(ai, "run_review", side_effect=delayed),
        ):
            with ai.FinalReviewQueue(args, self.output, self.cv) as queue:
                try:
                    queue.submit(self.path.parent)
                    self.assertTrue(entered.wait(3))
                    state = self.store.load(0)
                    self.store.save(0, dict(state, reviewer="human"))
                    self.assertTrue(self.store.load(0)["has_human_review"])
                finally:
                    release.set()

    def test_server_exposes_advice_and_rejects_different_mount(self):
        self.run_worker()
        with ReviewServer(self.output, 0, self.mount) as server:
            state = server.state(0, full=True)
            self.assertTrue(state["ai_review"]["valid"])
            self.assertFalse(state["has_human_review"])
            self.assertFalse(state["approved"])
            other = self.root / "other_mount.json"
            config = read_json(self.mount)
            config["head_euler_sequence"] = "zyx"
            atomic_json(other, config)
            server.calibration = other
            self.assertFalse(server.state(0, full=True)["ai_review"]["valid"])


class ResponseAndRotationTests(unittest.TestCase):
    def test_schema_rejects_unbounded_or_arbitrary_operations(self):
        row = dict(side="left", **decision())
        for changes in (
            {"axis": "matrix"},
            {"box": "replace"},
            {"region": [0, 0, 1001, 500]},
            {"region": [0, 0, float("nan"), 500]},
            {"rotations": [{"axis": "X", "degrees": 90}]},
            {"axis": "rotate", "rotations": [{"axis": "X", "degrees": 45}]},
            {"axis": "rotate", "rotations": [{"axis": "X", "degrees": 90}] * 4},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ai.validate_response({"decisions": [dict(row, **changes)]}, ["left"])
        with self.assertRaises(ValueError):
            ai.validate_response({"decisions": [row, row]}, ["left", "right"])
        row.update(axis="rotate", rotations=[{"axis": "X", "degrees": 90}] * 2)
        with self.assertRaises(ValueError):
            ai.validate_response({"decisions": [row]}, ["left"], followup=True)
        row.update(axis="keep", rotations=[], matrix=[[float("nan")]])
        parsed = ai.validate_response({"decisions": [row]}, ["left"])
        self.assertNotIn("matrix", parsed["left"])

    def test_all_quarter_turns_match_browser_including_gimbal_lock(self):
        cases = []
        for angles in ([0, 0, 0], [25, 90, 15], [25, -90, 15], [30, 40, 50]):
            for axis in "XYZ":
                for degree in (-90, 90, 180):
                    cases.append(dict(angles=angles, steps=[{"axis": axis, "degrees": degree}]))
            cases.append(dict(angles=angles, steps=[{"axis": a, "degrees": 90} for a in "XYZ"]))
        script = Path(__file__).resolve().parents[1] / "vive_correct/review_ui/axis_rotation.js"
        result = subprocess.run(
            [
                "node",
                "-e",
                "const fs=require('fs'),vm=require('vm');vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));"
                "console.log(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(c=>"
                "c.steps.reduce((a,s)=>rotateCurrentAxes(a,'XYZ'.indexOf(s.axis),s.degrees),c.angles))));",
                str(script),
            ],
            input=json.dumps(cases),
            text=True,
            capture_output=True,
            check=True,
        )
        for case, browser in zip(cases, json.loads(result.stdout)):
            actual = ai.rotate_local(case["angles"], case["steps"])
            np.testing.assert_allclose(
                Rotation.from_euler("XYZ", actual, degrees=True).as_matrix(),
                Rotation.from_euler("XYZ", browser, degrees=True).as_matrix(),
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
