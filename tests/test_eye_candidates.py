"""Label-only common-frame proposals and the fixed eye sampling schedule."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from test_cv_features import fixture
from test_eye_reference import make_eye_fixture
from vive_correct.camera_input import from_args, rectify, video_suffix
from vive_correct.cv_features import scan_features
from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.label_detection import label_metrics
from vive_correct.eye_candidates import joint_candidate, rank_frames, sample_indices
from vive_correct.human_review import ReviewStore


def long_video(episode, context):
    image, quads = fixture()
    writer = cv2.VideoWriter(
        str(episode / video_suffix(context)), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 400)
    )
    for _ in range(316):
        writer.write(image)
    writer.release()
    return image, quads


class JointFrameTests(unittest.TestCase):
    def test_lights_without_labels_are_not_candidates(self):
        image = np.full((600, 800, 3), 210, np.uint8)
        for x in (200, 600):
            cv2.rectangle(image, (x - 35, 330), (x + 35, 390), (30, 30, 30), -1)
            cv2.circle(image, (x, 440), 5, (230, 230, 40), -1)
        self.assertIsNone(joint_candidate(image, [], 11))

    def test_two_halves_and_single_label_without_invented_other_hand(self):
        image = np.zeros((600, 800, 3), np.uint8)

        def label(x, score=0.8):
            return dict(
                quad=np.array([[x, 300], [x + 40, 300], [x + 40, 325], [x, 325]]),
                selection_score=score,
                status="appearance_candidate",
            )

        same_half = joint_candidate(image, [label(80), label(220)], 4)
        self.assertEqual(set(same_half["hands"]), {"left"})
        self.assertEqual(same_half["label_count"], 1)
        self.assertFalse(same_half["other_hand_visibility_verified"])
        self.assertEqual(joint_candidate(image, [label(80), label(580)], 4)["label_count"], 2)
        self.assertEqual(joint_candidate(image, [label(80), label(80)], 4)["label_count"], 1)
        self.assertIsNone(joint_candidate(image, [label(380)], 4))
        self.assertIsNone(joint_candidate(image, [dict(label(80), status="review_candidate")], 4))

    def test_frame_ranking_prefers_two_labels_and_retains_individual_frames(self):
        frames = [
            dict(frame_index=f, score=s, label_count=n)
            for f, s, n in [(300, 10, 1), (305, 0.8, 2), (310, 0.9, 2)]
        ]
        self.assertEqual([c["frame_index"] for c in rank_frames(frames)], [310, 305, 300])

    def test_sampling_starts_at_ten_seconds_stride_five_maximum_one_hundred(self):
        self.assertEqual(list(sample_indices(10000, 30)), list(range(300, 800, 5)))
        self.assertEqual(list(sample_indices(316, 30)), [300, 305, 310, 315])
        self.assertEqual(list(sample_indices(300, 30)), [])
        self.assertEqual(sample_indices(10000, 29.97).start, 300)

    def test_short_video_does_not_fall_back_to_early_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _, _, context = make_eye_fixture(root)
            result = scan_features(episode, root / "scan", CVConfig.load(), context=context)
            self.assertEqual(result["sampled_frame_indices"], [])
            self.assertEqual(result["stop_reason"], "video_ends_before_10s")
            self.assertEqual(result["reference_candidates"], [])

    def test_single_labels_stay_on_their_own_frame_and_both_choices_load_same_frame(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _, _, context = make_eye_fixture(root)
            image, quads = long_video(episode, context)
            labels = [dict(label_metrics(image, q), quad=q) for q in quads]
            config = replace(
                CVConfig.load(), feature_fast_max_samples=2, feature_fast_timeout_s=0.0001
            )
            for search in ("fast", "full"):
                output = root / search
                detections = [{"candidates": {"left": [labels[i % 2]]}} for i in range(4)]
                with patch(
                    "vive_correct.cv_features.detect_label_candidates", side_effect=detections
                ):
                    result = scan_features(
                        episode, output / "features/sample", config, context=context, search=search
                    )
                self.assertEqual(result["sampled_frame_indices"], [300, 305, 310, 315])
                self.assertEqual(result["examined_frames"], 4)
                self.assertTrue(all(c["label_count"] == 1 for c in result["reference_candidates"]))
                for c in result["reference_candidates"]:
                    self.assertTrue(
                        all(v["frame_index"] == c["frame_index"] for v in c["hands"].values())
                    )
                store = ReviewStore(output)
                choices = store.load(0)["selected"]
                self.assertEqual(
                    {c["frame_index"] for c in choices.values()}, {result["reference_frame"]}
                )
                self.assertEqual(sum(bool(c["quad"]) for c in choices.values()), 1)
                self.assertEqual(len(store.reference_candidates(0)), 4)
                self.assertTrue(Path(result["reference_frame_image"]).is_file())

    def test_common_frame_asset_is_undistorted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _, _, context = make_eye_fixture(root)
            long_video(episode, context)
            calibration = Path(context["calibration_path"])
            doc = json.loads(calibration.read_text())
            doc["dist_coefs"] = [-0.5, 0.1, 0, 0, 0]
            calibration.write_text(json.dumps(doc))
            context = from_args(
                SimpleNamespace(coordinate_mode="eye-reference", eye_calibration=calibration)
            )
            _, quads = fixture()

            def proposal(im, labels, f):
                return dict(
                    frame_index=f,
                    hands={
                        "left": dict(
                            frame_index=f,
                            quad=quads[0],
                            status="appearance_candidate",
                            selection_score=0.8,
                        )
                    },
                    label_count=1,
                    evidence="single_label_pending_other_hand",
                    score=0.8,
                )

            with patch("vive_correct.eye_candidates.joint_candidate", side_effect=proposal):
                result = scan_features(episode, root / "scan", CVConfig.load(), context=context)
            cap = cv2.VideoCapture(str(episode / video_suffix(context)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, result["reference_frame"])
            _, raw = cap.read()
            cap.release()
            expected = cv2.imdecode(
                cv2.imencode(".jpg", rectify(raw, context))[1], cv2.IMREAD_COLOR
            )
            actual = cv2.imread(result["reference_frame_image"])
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.array_equal(actual, raw))
