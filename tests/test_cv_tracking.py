"""Conservative CV gates, identity continuity and offline CLI contract."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import cv2
import numpy as np

from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.cv_tracking import SideTracker, select_tracked_frames
from vive_correct.estimate.find_stable_frame import SelectionConfig
from vive_correct.estimate.label_detection import detect_labels, label_metrics, refine_corners


def scene(left=(150, 310), right=(470, 310), color=(230, 230, 230)):
    image = np.full((480, 640, 3), 160, np.uint8)
    quads = []
    for cx, cy in (left, right):
        cv2.ellipse(image, (cx, cy), (65, 55), 0, 0, 360, (20, 20, 20), -1)
        q = np.array(
            [[cx - 30, cy - 18], [cx + 30, cy - 18], [cx + 30, cy + 18], [cx - 30, cy + 18]], float
        )
        cv2.fillConvexPoly(image, q.astype(int), color)
        quads.append(q)
    return image, quads


class DetectorTests(unittest.TestCase):
    def test_white_shadow_and_color_gates(self):
        for color, accepted in (
            ((230, 230, 230), True),
            ((80, 80, 80), True),
            ((130, 210, 230), False),
            ((255, 255, 255), True),
        ):
            image, q = scene(color=color)
            self.assertEqual(label_metrics(image, q[0])["accepted"], accepted)
            self.assertEqual(bool(detect_labels(image)["left"]), accepted)

    def test_text_allowed_and_bright_background_rejected(self):
        image, q = scene()
        cv2.putText(image, "V", (140, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        self.assertTrue(label_metrics(image, q[0])["accepted"])
        background = np.full_like(image, 180)
        cv2.fillConvexPoly(background, q[0].astype(int), (240, 240, 240))
        self.assertFalse(detect_labels(background, {"left": [0, 0, 640, 480]})["left"])

    def test_crop_small_label_and_partial_edge_rejected(self):
        image, q = scene()
        self.assertFalse(detect_labels(image, {"left": [125, 295, 180, 328]})["left"])
        small = np.full_like(image, 160)
        cv2.rectangle(small, (50, 300), (100, 350), (20, 20, 20), -1)
        cv2.rectangle(small, (65, 315), (77, 325), (230, 230, 230), -1)
        self.assertFalse(detect_labels(small)["left"])
        cv2.rectangle(image, (115, 286), (190, 296), (160, 160, 160), -1)
        # A second unsupported edge must still be rejected; a single weak edge
        # can now be rescued by independently measured black-shell enclosure.
        cv2.rectangle(image, (114, 286), (124, 335), (160, 160, 160), -1)
        self.assertFalse(label_metrics(image, q[0])["accepted"])

    def test_perspective_and_refinement(self):
        image = np.full((400, 640, 3), 20, np.uint8)
        quad = np.array([[60, 80], [190, 95], [150, 130], [70, 120]], float)
        cv2.fillConvexPoly(image, quad.astype(int), (230, 230, 230))
        self.assertTrue(label_metrics(image, quad)["accepted"])
        shifted = quad + [1, 1]
        refined = refine_corners(image, shifted)
        self.assertLess(np.linalg.norm(refined - quad, axis=1).mean(), 1.5)
        self.assertEqual(len(detect_labels(image, {"left": [0, 0, 640, 400]})["left"]), 1)

    def test_scaled_proposals_return_native_pixel_corners(self):
        image, quads = scene()
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        result = detect_labels(image)
        self.assertEqual(set(side for side, values in result.items() if values), {"left", "right"})
        for side, expected in zip(("left", "right"), quads):
            from vive_correct.estimate.cv_tracking import align_quad

            # Nearest-neighbor expansion makes each source pixel a 3x3 block.
            native_edges = expected * 3 + [[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]]
            found = align_quad(native_edges, result[side][0])
            self.assertLess(np.linalg.norm(found - native_edges, axis=1).max(), 1.5)

    def test_configuration_is_validated_and_hashed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cv.yaml"
            path.write_text("max_saturation: 0.15\n")
            changed = CVConfig.load(path)
            self.assertNotEqual(changed.report()["sha256"], CVConfig.load().report()["sha256"])
            for text in (
                "unknown: 1",
                "feature_fast_max_samples: 1",
                "feature_fast_max_samples: 2.5",
                "feature_fast_timeout_s: 0",
                "min_white_fraction: 2",
                "min_contrast: .nan",
                "thresholds: [true]",
                "association_margin: 0.8",
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    CVConfig.load(path)


class TrackingTests(unittest.TestCase):
    def test_default_initialization_requires_five_consecutive_frames(self):
        tracker = SideTracker()
        self.assertEqual(tracker.window, 5)
        image, quads = scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for index in range(4):
            tracker.update(index, gray, [{"quad": q} for q in quads])
            self.assertFalse(tracker.initialized)
        tracker.update(4, gray, [{"quad": q} for q in quads])
        self.assertTrue(tracker.initialized)

    def test_init_requires_window_and_both_sides(self):
        image, quads = scene()
        selection = SelectionConfig(window=3)
        short = select_tracked_frames(enumerate([image] * 2), selection=selection)
        self.assertFalse(short["best"])
        found = select_tracked_frames(enumerate([image] * 4), selection=selection)
        self.assertEqual(set(found["best"]), {"left", "right"})
        self.assertFalse(found["identity_verified"])

    def test_current_frame_is_selected_without_stability_window(self):
        image, quads = scene()
        candidates = [dict(quad=q, score=1.0, compact_body=True, box_supported=True) for q in quads]
        detections = [dict(candidates={"left": candidates}, rejected={}) for _ in range(5)]
        # Immediately after initialization, only the left hand is re-detected.
        detections.append(dict(candidates={"left": [dict(candidates[0], score=2.0)]}, rejected={}))
        with (
            patch(
                "vive_correct.estimate.cv_tracking.detect_label_candidates", side_effect=detections
            ),
            patch(
                "vive_correct.estimate.find_stable_frame.measure_stability",
                side_effect=AssertionError("stability gate must not run"),
            ),
            patch("vive_correct.estimate.find_stable_frame.fit_candidate", return_value={}) as fit,
        ):
            result = select_tracked_frames(
                enumerate([image] * 6), selection=SelectionConfig(window=5), camera=object()
            )
        self.assertEqual(result["best"]["left"]["frame_index"], 5)
        self.assertEqual(result["best"]["left"]["window"], [5, 5])
        self.assertEqual(result["best"]["right"]["frame_index"], 4)
        self.assertFalse(result["stability_gate_enabled"])
        self.assertEqual(fit.call_count, 3)
        self.assertNotIn("passed", result["best"]["left"]["pose"]["stability"])

    def test_geometry_gate_still_rejects_invalid_pose(self):
        image, quads = scene()
        candidates = [dict(quad=q, score=1.0, compact_body=True, box_supported=True) for q in quads]
        with (
            patch(
                "vive_correct.estimate.cv_tracking.detect_label_candidates",
                return_value=dict(candidates={"left": candidates}, rejected={}),
            ),
            patch(
                "vive_correct.estimate.find_stable_frame.fit_candidate",
                side_effect=ValueError("bad geometry"),
            ),
        ):
            result = select_tracked_frames(
                enumerate([image] * 5), selection=SelectionConfig(window=5), camera=object()
            )
        self.assertEqual(result["best"], {})
        self.assertEqual(result["rejected"]["geometry_gate"], 2)

    def test_crossing_keeps_identity_with_real_lk(self):
        tracker = SideTracker(window=3)
        for index in range(71):
            # Different vertical positions keep both labels visible through a crossing.
            image, quads = scene((150 + 5 * index, 260), (490 - 5 * index, 390))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            candidates = [{"quad": q, "score": 1.0} for q in quads]
            matched = tracker.update(index, gray, candidates[::-1])
        self.assertTrue(tracker.initialized)
        self.assertEqual(set(matched), {"left", "right"})
        self.assertGreater(
            matched["left"]["quad"][:, 0].mean(), matched["right"]["quad"][:, 0].mean()
        )

    def test_loss_and_frame_gap_never_reinitialize_by_half(self):
        tracker = SideTracker(window=3)
        image, quads = scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = [{"quad": q} for q in quads]
        for index in range(3):
            tracker.update(index, gray, candidates)
        self.assertTrue(tracker.initialized)
        self.assertFalse(tracker.update(10, gray, candidates))
        for index in range(11, 16):
            self.assertFalse(tracker.update(index, gray, candidates))

    def test_ambiguous_association_is_unknown(self):
        tracker = SideTracker(window=3)
        image, quads = scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for index in range(3):
            tracker.update(index, gray, [{"quad": q} for q in quads])
        matched = tracker.update(
            3, gray, [{"quad": quads[0]}, {"quad": quads[0] + 1}, {"quad": quads[1]}]
        )
        self.assertNotIn("left", matched)
        self.assertIn("right", matched)


class OfflinePipelineTests(unittest.TestCase):
    def test_real_video_pipeline_never_calls_model_and_resumes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import yaml
        from vive_correct.data.dataset import VIDEO_SUFFIX, TIMESTAMP_SUFFIX
        from vive_correct.estimate.geometry import VIDEO_KEY
        from vive_correct.run_dataset import main

        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "dataset/desk/example"
            video = episode / VIDEO_SUFFIX
            video.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
            self.assertTrue(writer.isOpened())
            image, _ = scene()
            for _ in range(15):
                writer.write(image)
            writer.release()
            timestamps = episode / TIMESTAMP_SUFFIX
            timestamps.parent.mkdir(parents=True)
            pq.write_table(pa.table({"timestamp": np.arange(15) / 30}), timestamps)
            calibration = episode / "calibrations/rectify_params.yaml"
            calibration.parent.mkdir()
            calibration.write_text(
                yaml.safe_dump(
                    {
                        "cameras": {
                            VIDEO_KEY: {
                                "camera_model": "pinhole",
                                "distortion_model": "none",
                                "intrinsics": [500, 500, 320, 240],
                                "resolution": [640, 480],
                            }
                        }
                    }
                )
            )
            for flags, folder in (
                (["--profile", "cv", "--model-review", "on", "--stage", "all"], "cv"),
                (["--model-review", "off"], "off"),
            ):
                args = [
                    "--dataset",
                    str(root / "dataset"),
                    "--out",
                    str(root / folder),
                    "--jobs",
                    "1",
                    *flags,
                ]
                with (
                    patch(
                        "vive_correct.estimate.openai_vision.load_config",
                        side_effect=AssertionError("model config"),
                    ),
                    patch(
                        "vive_correct.estimate.openai_vision._post_chat",
                        side_effect=AssertionError("network"),
                    ),
                    patch(
                        "vive_correct.batch.automatic.run_automatic",
                        side_effect=AssertionError("model pipeline"),
                    ),
                    patch("subprocess.run", side_effect=AssertionError("external model process")),
                ):
                    self.assertEqual(main(args), 0)
                    plan = json.loads((root / folder / "run_plan.json").read_text())
                    self.assertEqual(plan["selection"]["window"], 5)
                    summary = json.loads(
                        (root / folder / "batch/delivery/summary.json").read_text()
                    )
                    self.assertIsNone(summary["model"])
                    self.assertEqual(set(summary["episodes"][0]["hands"]), {"left", "right"})
                    self.assertFalse(summary["human_verified"])
                    delivery = root / folder / "batch/delivery"
                    self.assertTrue((delivery / "tracked.html").is_file())
                    self.assertIn('href="tracked.html"', (delivery / "index.html").read_text())
                    features = json.loads((delivery / "features_summary.json").read_text())
                    self.assertEqual(features["counts"]["episodes"], 1)
                    report = features["episodes"][0]
                    self.assertEqual(report["video_space"], "rectified")
                    self.assertTrue(Path(report["human_review_file"]).is_file())
                    self.assertTrue(
                        all("label_center_px" in c for c in report["features"].values())
                    )
                    self.assertEqual(main(args), 0)
                    self.assertIn('href="tracked.html"', (delivery / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
