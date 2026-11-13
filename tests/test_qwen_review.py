"""Model responses must preserve identity, uncertainty and proper rotations."""

from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from vive_correct.batch.qwen_review import validate_decisions, review_candidates
from vive_correct.estimate.openai_vision import OpenAIError


class QwenReviewTests(unittest.TestCase):
    def decision(self):
        return dict(
            candidate_id="a",
            is_label=True,
            clear=True,
            physical_side="right",
            confidence=0.9,
            axis_status="unknown",
            old_from_new=None,
            axis_reason="Ambiguous rectangle",
            mounting_reason="Offset unmeasured",
        )

    def test_unknown_axes_and_physical_side_are_preserved(self):
        decision = self.decision()
        self.assertEqual(validate_decisions({"decisions": [decision]}, ["a"]), [decision])

    def test_malformed_or_improper_proposals_are_rejected(self):
        decision = self.decision()
        cases = [[], {"decisions": [None]}, {"decisions": []}, {"decisions": [decision, decision]}]
        for change in (dict(is_label="true"), dict(confidence=float("nan"))):
            value = deepcopy(decision)
            value.update(change)
            cases.append({"decisions": [value]})
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_decisions(data, ["a"])

    def test_unsolicited_rotation_cannot_discard_identity_or_rotate_pose(self):
        decision = self.decision()
        decision.update(axis_status="inferred", old_from_new=[[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        result = validate_decisions({"decisions": [decision]}, ["a"])[0]
        self.assertTrue(result["is_label"])
        self.assertEqual(result["axis_status"], "unknown")
        self.assertIsNone(result["old_from_new"])

    def test_content_rejection_is_not_retried(self):
        config = SimpleNamespace(model="qwen-test", api_key="secret-value")
        with patch("vive_correct.batch.qwen_review.time.sleep") as sleep:
            calls = []

            def rejected(*args):
                calls.append(1)
                raise OpenAIError("HTTP 400 data_inspection_failed")

            with self.assertRaises(OpenAIError):
                review_candidates([], None, config, retries=3, transport=rejected)
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_retry_is_bounded_and_secret_is_redacted(self):
        config = SimpleNamespace(model="qwen-test", api_key="secret-value")
        calls = []

        def failed(*args):
            calls.append(1)
            raise OpenAIError("secret-value timeout")

        with patch("vive_correct.batch.qwen_review.time.sleep"):
            with self.assertRaises(OpenAIError) as raised:
                review_candidates([], None, config, retries=2, transport=failed)
        self.assertEqual(len(calls), 2)
        self.assertNotIn(config.api_key, str(raised.exception))

    def test_schema_failure_retries_then_returns_valid_response(self):
        config = SimpleNamespace(model="qwen-test", api_key="secret-value")
        responses = iter(['{"decisions":[null]}', '{"decisions":[]}'])
        with patch("vive_correct.batch.qwen_review.time.sleep"):
            self.assertEqual(
                review_candidates(
                    [], None, config, retries=2, transport=lambda *args: next(responses)
                ),
                [],
            )


class AutomaticDeliveryTests(unittest.TestCase):
    def test_empty_observation_still_exports_and_failure_returns_nonzero(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from vive_correct.batch.automatic import run_automatic, write_json
        from vive_correct.batch.report_io import digest

        config = SimpleNamespace(model="qwen-test", api_key="secret-value")
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            report = work / "input.json"
            write_json(report, dict(episode="/episode", observations=[]))
            entry = dict(index=1, episode="/episode", report=str(report))
            write_json(work / "scan_manifest.json", dict(entries=[entry]))

            def calibration(report, initial, output):
                value = dict(
                    episode="/episode",
                    accepted=False,
                    reason="no_anchor",
                    source_report_sha256=digest(report),
                )
                write_json(output / "calibration.json", value)
                return value

            with (
                patch(
                    "vive_correct.batch.automatic.review_episode",
                    side_effect=ValueError("API failed"),
                ),
                patch("vive_correct.batch.automatic.calibrate", side_effect=calibration),
            ):
                self.assertEqual(run_automatic(work, config, "/initial", jobs=12), 1)
            manifest = json.loads((work / "batch/delivery/gallery_manifest.json").read_text())
            self.assertEqual(manifest["no_observation_count"], 1)
            self.assertEqual(manifest["entries"][0]["status"], "review_error")
            self.assertTrue((work / "batch/delivery/images/01_global.jpg").is_file())
            self.assertFalse((work / "batch/delivery/evidence").exists())


class CalibrationTrustTests(unittest.TestCase):
    def test_tracking_does_not_promote_unverified_axes(self):
        import numpy as np
        from vive_correct.batch.mount_calibration import track_reference

        pose = dict(
            object_corners_local_m=np.zeros((4, 3)).tolist(),
            image_corners_px=[[10, 10], [20, 10], [20, 20], [10, 20]],
            T_camera_local=np.eye(4).tolist(),
            physical_axes_verified=False,
        )
        episode = SimpleNamespace(video="/video", timestamps=[0.0])
        with patch("vive_correct.batch.mount_calibration.cv2.VideoCapture") as capture:
            capture.return_value.read.return_value = (True, np.zeros((30, 30, 3), dtype=np.uint8))
            track = track_reference(
                episode, dict(frame_index=0, stickers={"left": pose}), "left", radius=0
            )
        self.assertFalse(track["observations"][0]["physical_axes_verified"])
