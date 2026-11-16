"""Geometry regressions and selection policy, independent of private videos."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pyarrow as pa
import yaml

from vive_correct.estimate.find_stable_frame import (
    Episode,
    SelectionConfig,
    measure_stability,
    select_frames,
)
from vive_correct.estimate.geometry import (
    LABEL_CORNERS,
    Camera,
    VIDEO_KEY,
    estimate_pose,
    order_quad,
)
from vive_correct.estimate.label_detection import (
    BORDER_MIN_DARK,
    black_border_score,
    detect_labels,
    label_likeness_score,
    select_best_label_frames,
)
from vive_correct.estimate.plot_three_views_new import render
from vive_correct.estimate.sticker_pipeline import read_observations, run

cv2.setNumThreads(1)
FIXTURE = Path(__file__).parent / "fixtures/wrist_label_poses.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


class GeometryTests(unittest.TestCase):
    def test_origin_only_pose_is_json_serializable(self):
        case = CASES[0]
        camera = Camera(np.array(case["K"]), tuple(case["resolution"]), "fixture")
        pose = estimate_pose(case["corners"], camera)
        decoded = json.loads(json.dumps(pose))
        self.assertFalse(decoded["axes_generated"])
        self.assertIsInstance(decoded["low_resolution"], bool)
        self.assertIsInstance(decoded["plane_ambiguous"], bool)

    def test_reprojection_threshold_is_six_pixels_inclusive(self):
        from vive_correct.estimate.geometry import MAX_RMS_PX

        self.assertEqual(MAX_RMS_PX, 6.0)
        candidate = {"rms_px": 6.0, "R": np.eye(3), "t": np.array([0.0, 0.0, 1.0])}
        with (
            patch("vive_correct.estimate.geometry.pose_candidates", return_value=[candidate]),
            patch(
                "vive_correct.estimate.geometry.local_basis", side_effect=ValueError("axis gate")
            ) as basis,
        ):
            with self.assertRaisesRegex(ValueError, "direction limits"):
                estimate_pose(None, None, device_fixed=False)
            basis.assert_called_once()
            basis.reset_mock()
            candidate["rms_px"] = 6.001
            with self.assertRaisesRegex(ValueError, "RMS exceeds 6 px"):
                estimate_pose(None, None, device_fixed=False)
            basis.assert_not_called()

    def test_all_reviewed_origins_and_final_axes_are_preserved(self):
        # These reviewed quads face the camera too obliquely for the axis limits.
        rejected = {
            ("07_desk_f900", "right"),
            ("08_desk_f460", "right"),
            ("09_desk_f40", "right"),
            ("10_workbench_f420", "left"),
        }
        for case in CASES:
            with self.subTest(sample=case["sample"], side=case["side"]):
                camera = Camera(np.array(case["K"]), tuple(case["resolution"]), "fixture")
                if (case["sample"], case["side"]) in rejected:
                    with self.assertRaisesRegex(ValueError, "direction limits"):
                        estimate_pose(case["corners"], camera, device_fixed=False)
                    continue
                pose = estimate_pose(case["corners"], camera, device_fixed=False)
                self.assertLessEqual(pose["axis_direction_deg"]["x_from_left"], 40)
                self.assertLessEqual(pose["axis_direction_deg"]["y_from_up"], 40)
                self.assertLessEqual(pose["axis_direction_deg"]["z_from_forward"], 50)
                T = np.array(pose["T_camera_local"])
                expected = np.array(case["T_camera_local"])
                expected[:3, :3] = expected[:3, :3] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                np.testing.assert_allclose(T, expected, atol=1e-5)
                self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1)
                np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8)
                projected = camera.project(pose["object_corners_local_m"], T[:3, :3], T[:3, 3])
                rms = np.sqrt(np.mean(np.sum((projected - pose["image_corners_px"]) ** 2, axis=1)))
                self.assertAlmostEqual(rms, pose["rms_px"], places=7)
                self.assertIsNone(pose["T_label_vive"])

    def test_corner_order_does_not_change_the_origin_or_final_axes(self):
        case = CASES[0]
        camera = Camera(np.array(case["K"]), tuple(case["resolution"]), "fixture")
        points = np.array(case["corners"])
        for q in (points[::-1], points[[2, 0, 3, 1]], np.roll(points, 1, axis=0)):
            pose = estimate_pose(q, camera, device_fixed=False)
            expected = np.array(case["T_camera_local"])
            expected[:3, :3] = expected[:3, :3] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            np.testing.assert_allclose(pose["T_camera_local"], expected, atol=1e-5)

    def test_bad_or_tiny_quads_are_rejected(self):
        for q in (
            [[0, 0]] * 4,
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[0, 0], [40, 0], [40, 40], [float("nan"), 40]],
        ):
            with self.assertRaises(ValueError):
                order_quad(q)

    def test_small_label_requires_explicit_threshold_and_quality_flag(self):
        camera = Camera(
            np.array([[800.0, 0, 960], [0, 800, 600], [0, 0, 1]]), (1920, 1200), "fixture"
        )
        # Frontal 50×30 mm label at 1.8 m: the short edge is 13.3 px.
        rotation = np.diag([-1.0, -1.0, 1.0])
        quad = camera.project(LABEL_CORNERS, rotation, np.array([0.0, 0.0, 1.8]))
        with self.assertRaises(ValueError):
            estimate_pose(quad, camera, device_fixed=False)
        pose = estimate_pose(quad, camera, min_edge_px=8, device_fixed=False)
        self.assertTrue(pose["low_resolution"])
        self.assertAlmostEqual(pose["projected_min_edge_px"], 40 / 3, places=4)
        self.assertLess(pose["rms_px"], 2)
        self.assertLessEqual(pose["axis_direction_deg"]["z_from_forward"], 50)

    def test_camera_rejects_distortion_and_other_camera_features(self):
        camera = {
            "camera_model": "pinhole",
            "distortion_model": "none",
            "intrinsics": [800, 800, 960, 600],
            "resolution": [1920, 1184],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.yaml"
            for override in (
                {"distortion_model": "radtan"},
                {"distortion_coeffs": [0.1]},
                {"feature_key": "neon"},
            ):
                path.write_text(
                    yaml.safe_dump({"cameras": {"headband_left_rgb_rectified": camera | override}})
                )
                with self.assertRaises(ValueError):
                    Camera.load(path)
            path.write_text(yaml.safe_dump({"cameras": {"headband_left_rgb_rectified": camera}}))
            self.assertEqual(Camera.load(path).resolution, (1920, 1184))


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.config = SelectionConfig()
        self.gray = np.random.default_rng(42).integers(0, 255, (240, 320), dtype=np.uint8)
        self.quad = np.array([[220, 160], [340, 160], [340, 240], [220, 240]], float)

    def test_stationary_texture_passes(self):
        result = measure_stability(
            [self.gray] * self.config.window, self.quad, (640, 480), self.config
        )
        self.assertTrue(result["passed"])
        self.assertLess(result["median_motion_px_per_frame"], 0.01)

    def test_moving_texture_and_incomplete_window_fail(self):
        frames = [
            cv2.warpAffine(
                self.gray,
                np.float32([[1, 0, 2 * (i - self.config.window // 2)], [0, 1, 0]]),
                (320, 240),
            )
            for i in range(self.config.window)
        ]
        self.assertFalse(measure_stability(frames, self.quad, (640, 480), self.config)["passed"])
        self.assertFalse(
            measure_stability(frames[:9], self.quad, (640, 480), self.config)["passed"]
        )

    def test_invalid_selection_options_fail(self):
        for options in (
            {"window": 20},
            {"stride": 0},
            {"max_frames": 10},
            {"max_fb_px": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SelectionConfig(**options)


class EpisodeTests(unittest.TestCase):
    def load_episode(self, video_size, timestamps):
        camera = Camera(np.eye(3), (1920, 1184), "fixture")
        capture = Mock()
        capture.isOpened.return_value = True
        properties = {
            cv2.CAP_PROP_FRAME_WIDTH: video_size[0],
            cv2.CAP_PROP_FRAME_HEIGHT: video_size[1],
            cv2.CAP_PROP_FRAME_COUNT: 21,
        }
        capture.get.side_effect = properties.__getitem__
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (
                root / f"videos/chunk-000/{VIDEO_KEY}/episode_000000.mp4",
                root / f"data/chunk-000/episode_000000/{VIDEO_KEY}.parquet",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with (
                patch("vive_correct.estimate.find_stable_frame.Camera.load", return_value=camera),
                patch(
                    "vive_correct.estimate.find_stable_frame.cv2.VideoCapture", return_value=capture
                ),
                patch(
                    "vive_correct.estimate.find_stable_frame.pq.read_table",
                    return_value=pa.table({"timestamp": timestamps}),
                ),
            ):
                return Episode.load(root)

    def test_native_cropped_resolution_is_supported(self):
        episode = self.load_episode((1920, 1184), np.arange(21) / 30)
        self.assertEqual(episode.camera.resolution, (1920, 1184))

    def test_video_calibration_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.load_episode((1920, 1200), np.arange(21) / 30)

    def test_timestamp_count_or_order_mismatch_is_rejected(self):
        for times in (np.arange(20), np.zeros(21), np.arange(21)[::-1]):
            with self.assertRaisesRegex(ValueError, "timestamps"):
                self.load_episode((1920, 1184), times)


class SelectionTests(unittest.TestCase):
    def selection(self, sides_by_frame):
        camera = Camera(np.eye(3), (100, 100), "fixture")
        episode = Episode(Path("/episode"), Path("/video"), camera, np.arange(100) / 30)
        quad = [[20, 20], [50, 20], [50, 50], [20, 50]]
        frames = [(index, np.full((1, 1, 3), index, np.uint8), []) for index in sides_by_frame]
        provider = lambda im: {side: [quad] for side in sides_by_frame[int(im[0, 0, 0])]}
        pose = {"rms_px": 0.1, "image_corners_px": quad}
        with (
            patch(
                "vive_correct.estimate.find_stable_frame.frame_windows", return_value=iter(frames)
            ),
            patch(
                "vive_correct.estimate.find_stable_frame.estimate_pose",
                side_effect=lambda *args, **kwargs: dict(pose),
            ),
            patch(
                "vive_correct.estimate.find_stable_frame.measure_stability",
                return_value={"passed": True},
            ),
        ):
            return select_frames(episode, provider, refine=False)

    def test_later_pair_wins_over_earlier_single(self):
        result = self.selection({10: ["left"], 20: ["right"], 30: ["left", "right"]})
        self.assertEqual(result["mode"], "both_hands_same_frame")
        self.assertEqual([o["frame_index"] for o in result["observations"]], [30])

    def test_separate_frames_remain_separate(self):
        result = self.selection({10: ["left"], 20: ["right"]})
        self.assertEqual(result["mode"], "separate_frames")
        self.assertEqual([list(o["labels"]) for o in result["observations"]], [["left"], ["right"]])

    def test_missing_hand_is_not_synthesized(self):
        result = self.selection({10: ["left"]})
        self.assertEqual(result["mode"], "single_hand")
        self.assertEqual(set(result["observations"][0]["labels"]), {"left"})
        self.assertEqual(self.selection({10: []})["mode"], "no_usable_label")


class InterfaceTests(unittest.TestCase):
    def test_dark_shell_alone_never_becomes_a_label(self):
        image = np.full((480, 640, 3), 180, np.uint8)
        cv2.ellipse(image, (180, 350), (75, 55), 0, 0, 360, (15, 15, 15), -1)
        self.assertTrue(all(not values for values in detect_labels(image).values()))

    def test_white_label_on_black_shell_is_detected(self):
        image = np.full((480, 640, 3), 180, np.uint8)
        cv2.ellipse(image, (160, 360), (90, 70), 0, 0, 360, (20, 20, 20), -1)
        cv2.rectangle(image, (130, 340), (190, 376), (230, 230, 230), -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quad = np.array([[130, 340], [190, 340], [190, 376], [130, 376]], float)
        self.assertGreaterEqual(black_border_score(gray, quad), BORDER_MIN_DARK)
        found = detect_labels(image)
        self.assertTrue(found["left"])
        self.assertFalse(found["right"])

    def test_white_label_without_black_border_is_rejected(self):
        image = np.full((480, 640, 3), 200, np.uint8)
        cv2.rectangle(image, (130, 340), (190, 376), (240, 240, 240), -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quad = np.array([[130, 340], [190, 340], [190, 376], [130, 376]], float)
        self.assertLess(black_border_score(gray, quad), BORDER_MIN_DARK)
        found = detect_labels(image, {"left": [0, 0, 320, 480]})
        self.assertFalse(found["left"])

    def test_select_best_label_frames_keeps_one_side_each(self):
        weak = np.full((480, 640, 3), 180, np.uint8)
        cv2.ellipse(weak, (160, 360), (90, 70), 0, 0, 360, (40, 40, 40), -1)
        cv2.rectangle(weak, (130, 340), (190, 376), (230, 230, 230), -1)
        strong = np.full((480, 640, 3), 180, np.uint8)
        cv2.ellipse(strong, (160, 360), (100, 80), 0, 0, 360, (15, 15, 15), -1)
        cv2.rectangle(strong, (125, 338), (195, 380), (235, 235, 235), -1)
        gray = cv2.cvtColor(strong, cv2.COLOR_BGR2GRAY)
        quad = np.array([[125, 338], [195, 338], [195, 380], [125, 380]], float)
        self.assertGreater(label_likeness_score(gray, quad), 0)
        best = select_best_label_frames([(3, weak), (9, strong), (12, weak)])
        self.assertIn("left", best)
        self.assertEqual(best["left"]["frame_index"], 9)
        self.assertNotIn("right", best)

    def test_review_file_cannot_silently_change_episode(self):
        episode = Episode(Path("/correct"), Path("/video"), None, np.arange(100))
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "review.json"
            p.write_text(
                json.dumps(
                    {
                        "episode": "/wrong",
                        "frame_index": 30,
                        "labels": {"left": CASES[0]["corners"]},
                    }
                )
            )
            with self.assertRaises(ValueError):
                read_observations(p, episode)

    def test_cv_default_does_not_load_or_call_a_model(self):
        camera = Camera(np.eye(3), (100, 100), "fixture")
        episode = Episode(Path("/episode"), Path("/video"), camera, np.arange(100) / 30)
        empty = {
            "mode": "no_usable_label",
            "observations": [],
            "examined_windows": 1,
            "rejected": {},
        }
        with (
            TemporaryDirectory() as tmp,
            patch("vive_correct.estimate.sticker_pipeline.Episode.load", return_value=episode),
            patch("vive_correct.estimate.sticker_pipeline.select_frames", return_value=empty),
            patch(
                "vive_correct.estimate.openai_vision.load_config",
                side_effect=AssertionError("Unexpected AI setup"),
            ),
        ):
            stale = Path(tmp) / "sticker_pose_rgb.png"
            stale.write_bytes(b"old")
            report = run("/episode", tmp)
            self.assertEqual(report["mode"], "no_usable_label")
            self.assertFalse(stale.exists())
            self.assertTrue((Path(tmp) / "sticker_pose_report_new.json").exists())

    def test_ai_requires_explicit_frame(self):
        episode = Episode(
            Path("/episode"),
            Path("/video"),
            Camera(np.eye(3), (100, 100), "fixture"),
            np.arange(100),
        )
        with patch("vive_correct.estimate.sticker_pipeline.Episode.load", return_value=episode):
            with self.assertRaisesRegex(ValueError, "requires --frame"):
                run("/episode", detector="ai")

    def test_three_view_handles_single_side_with_transformed_corners(self):
        case = CASES[0]
        camera = Camera(np.array(case["K"]), tuple(case["resolution"]), "fixture")
        pose = estimate_pose(case["corners"], camera, device_fixed=False)
        report = {
            "axis_length_m": 0.04,
            "observations": [{"frame_index": 30, "stickers": {"left": pose}}],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            out = Path(tmp) / "views.png"
            path.write_text(json.dumps(report))
            render(path, out)
            self.assertGreater(out.stat().st_size, 1000)
            with self.assertRaises(ValueError):
                render(path, out, 1)


if __name__ == "__main__":
    unittest.main()
