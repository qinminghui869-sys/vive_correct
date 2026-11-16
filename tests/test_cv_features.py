import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from vive_correct.cv_features import choose_features, scan_features, feature_video, RAW_LEFT_SUFFIX
from vive_correct.dataset import VIDEO_SUFFIX
from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.label_detection import black_context_metrics, label_metrics


def fixture():
    image = np.full((400, 640, 3), 180, np.uint8)
    quads = []
    for cx in (150, 490):
        cv2.ellipse(image, (cx, 300), (70, 55), 0, 0, 360, (25, 25, 25), -1)
        quad = np.array([[cx - 30, 282], [cx + 30, 282], [cx + 30, 318], [cx - 30, 318]], float)
        cv2.fillConvexPoly(image, quad.astype(int), (230, 230, 230))
        quads.append(quad)
    return image, quads


class ContextTests(unittest.TestCase):
    def test_real_shell_encloses_white_label_without_counting_label_as_black(self):
        image, quads = fixture()
        result = black_context_metrics(image, quads[0])
        self.assertGreater(result["shell_black_fraction"], 0.8)
        self.assertGreater(result["enclosed_fraction"], 0.95)
        self.assertEqual(result["label_black_fraction"], 0)
        self.assertLess(result["surround_black_fraction"], result["shell_black_fraction"])

    def test_repeated_keyboard_keys_are_not_a_tracker_body(self):
        from vive_correct.estimate.label_detection import tracker_body_metrics

        image = np.full((400, 640, 3), 25, np.uint8)
        quad = np.array([[280, 280], [320, 280], [320, 305], [280, 305]], float)
        for x in (220, 280, 340):
            for y in (235, 280, 325):
                cv2.rectangle(image, (x, y), (x + 40, y + 25), (220, 220, 220), -1)
        result = tracker_body_metrics(image, quad)
        self.assertGreaterEqual(result["nearby_white_components"], 4)
        self.assertFalse(result["compact_body"])

    def test_one_sided_black_strip_does_not_enclose_label(self):
        image, quads = fixture()
        image[:] = 180
        cv2.fillConvexPoly(image, quads[0].astype(int), (230, 230, 230))
        cv2.rectangle(image, (110, 270), (190, 280), (20, 20, 20), -1)
        self.assertLess(black_context_metrics(image, quads[0])["enclosed_fraction"], 0.1)
        self.assertFalse(label_metrics(image, quads[0])["accepted"])

    def test_surround_threshold_changes_only_context_classification(self):
        from dataclasses import replace

        image, quads = fixture()
        image[np.all(image == 180, axis=2)] = 110
        default = CVConfig.load()
        a = black_context_metrics(image, quads[0], default)
        b = black_context_metrics(image, quads[0], replace(default, surround_dark_threshold=120))
        self.assertEqual(a["shell_black_fraction"], b["shell_black_fraction"])
        self.assertGreater(b["surround_black_fraction"], a["surround_black_fraction"])

    def test_enclosure_cannot_bypass_missing_black_material_on_one_side(self):
        image, quads = fixture()
        cv2.rectangle(image, (118, 275), (182, 280), (150, 150, 150), -1)
        self.assertFalse(label_metrics(image, quads[0])["accepted"])
        cv2.rectangle(image, (110, 280), (122, 322), (180, 180, 180), -1)
        cv2.rectangle(image, (118, 275), (182, 285), (180, 180, 180), -1)
        self.assertFalse(label_metrics(image, quads[0])["accepted"])

    def test_neutral_black_box_accepts_label_but_colored_shell_does_not(self):
        from vive_correct.estimate.label_detection import box_metrics

        image, quads = fixture()
        self.assertTrue(box_metrics(image, quads[0])["box_supported"])
        image[np.all(image == 25, axis=2)] = (115, 40, 15)
        self.assertFalse(box_metrics(image, quads[0])["box_supported"])

    def test_real_background_distractors_fail_box_or_body_gate(self):
        root = Path(__file__).parent / "fixtures/cv_box_negatives"
        for item in json.loads((root / "cases.json").read_text()):
            with self.subTest(name=item["name"]):
                result = label_metrics(cv2.imread(str(root / item["image"])), item["quad"])
                from vive_correct.cv_features import feature_candidate_allowed

                self.assertFalse(
                    feature_candidate_allowed(dict(result, quad=item["quad"]), CVConfig.load())
                )

    def test_feature_gate_rejects_acute_reflection_without_requiring_fixed_aspect(self):
        from vive_correct.cv_features import feature_candidate_allowed

        image, quads = fixture()
        c = dict(label_metrics(image, quads[0]), quad=quads[0])
        self.assertTrue(feature_candidate_allowed(c, CVConfig.load()))
        c["quad"] = [[0, 0], [120, 35], [70, 40], [-10, 30]]
        self.assertFalse(feature_candidate_allowed(c, CVConfig.load()))


class FeatureTests(unittest.TestCase):
    def test_near_best_pair_preferred_and_weak_pair_not_forced(self):
        best = {
            s: {"selection_score": 1, "frame_index": i} for i, s in enumerate(("left", "right"))
        }
        pair = {s: {"selection_score": 0.8, "frame_index": 10} for s in best}
        selected, mode = choose_features(best, [pair], 0.7)
        self.assertEqual(mode, "same_frame")
        self.assertEqual(selected["left"]["frame_index"], 10)
        pair["right"]["selection_score"] = 0.3
        self.assertEqual(choose_features(best, [pair], 0.7)[1], "separate_frames")

    def test_raw_left_is_never_used_when_rectified_is_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / RAW_LEFT_SUFFIX
            raw.parent.mkdir(parents=True)
            raw.touch()
            self.assertEqual(feature_video(root), (root / VIDEO_SUFFIX, "rectified"))
            with self.assertLogs("vive_correct.cv_features", level="ERROR") as logs:
                result = scan_features(root, root / "out", CVConfig.load())
            self.assertEqual(result["features"], {})
            self.assertEqual(result["error"], "missing_video")
            self.assertIn("raw fallback disabled", logs.output[0])
            rectified = root / VIDEO_SUFFIX
            rectified.parent.mkdir(parents=True)
            rectified.touch()
            self.assertEqual(feature_video(root), (rectified, "rectified"))

    def test_missing_video_creates_explicit_diagnostics_not_fake_detections(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = scan_features(root / "missing", root / "out", CVConfig.load())
            self.assertEqual(result["error"], "missing_video")
            self.assertEqual(result["features"], {})
            self.assertEqual(set(result["missing_hands"]), {"left", "right"})
            self.assertTrue(all(Path(p).is_file() for p in result["feature_images"].values()))

    def test_real_video_pair_outputs_two_features_and_one_frame_without_calibration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "episode" / VIDEO_SUFFIX
            video.parent.mkdir(parents=True)
            image, _ = fixture()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 400))
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(image)
            writer.release()
            result = scan_features(root / "episode", root / "out", CVConfig.load(), stride=3)
            self.assertEqual(result["mode"], "same_frame")
            self.assertEqual(set(result["features"]), {"left", "right"})
            self.assertFalse(result["human_verified"])
            self.assertEqual(len({c["frame_image"] for c in result["features"].values()}), 1)
            self.assertTrue(
                all(cv2.imread(p).shape == (420, 940, 3) for p in result["feature_images"].values())
            )
            self.assertEqual(
                json.loads((root / "out/features.json").read_text())["missing_hands"], []
            )

            for item in result["features"].values():
                self.assertEqual(len(item["quad"]), 4)
                self.assertEqual(len(item["label_center_px"]), 2)
                self.assertTrue(Path(item["clean_frame_image"]).is_file())
                self.assertEqual(item["image_size_px"], [640, 400])
                np.testing.assert_allclose(item["label_center_px"][1], 300, atol=2)
            review_path = root / "out/human_review.json"
            review = json.loads(review_path.read_text())
            self.assertEqual(len(review["observations"]), 2)
            self.assertTrue(all(x["status"] == "pending" for x in review["observations"]))
            review_path.write_text('{"reviewer": "preserve-me"}')
            scan_features(root / "episode", root / "out", CVConfig.load(), stride=3)
            self.assertEqual(json.loads(review_path.read_text())["reviewer"], "preserve-me")

            from dataclasses import replace
            from vive_correct.cv_features import refilter_features

            refined = refilter_features(
                root / "out",
                root / "refined",
                replace(CVConfig.load(), max_label_black_fraction=0.1),
            )
            self.assertEqual(refined["mode"], "same_frame")
            self.assertEqual(refined["cache_action"], "tightened_feature_gates")
            self.assertTrue(refined["refilter_cache_key"])
            with self.assertRaises(ValueError):
                refilter_features(
                    root / "out", root / "bad", replace(CVConfig.load(), box_min_black=0.9)
                )

    def test_corrupt_rectified_video_does_not_fall_back(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (VIDEO_SUFFIX, RAW_LEFT_SUFFIX):
                path = root / suffix
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"invalid video")
            with self.assertLogs("vive_correct.cv_features", level="ERROR"):
                result = scan_features(root, root / "out", CVConfig.load())
            self.assertEqual(result["error"], "cannot_decode_video")
            self.assertEqual(result["video_space"], "rectified")
            self.assertEqual(result["features"], {})

    def test_raw_candidate_cache_is_rejected(self):
        from vive_correct.cv_features import refilter_features

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features.json").write_text(
                json.dumps(
                    dict(
                        episode=str(root),
                        video=str(root / RAW_LEFT_SUFFIX),
                        video_space="raw_left_unrectified",
                    )
                )
            )
            with self.assertLogs("vive_correct.cv_features", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "rectified"):
                    refilter_features(root, root / "out", CVConfig.load())
