"""Selection must preserve constraints and avoid model-confidence driven rotations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from vive_correct.batch.selection_policy import choose_pair, load_constraints, temporal_check
from vive_correct.batch.report_io import digest
from vive_correct.estimate.geometry import Camera


def item(frame, *, stable=True, constrained=False, edge=20):
    return {
        "frame_index": frame,
        "confidence": 1.0,
        "pose": {
            "orientation_status": "user_constraint_applied" if constrained else "model_uncertain",
            "stability": {"passed": stable},
            "visual_quality": {"min_edge_px": edge, "sharpness": 10.0},
            "rms_px": 0.1,
        },
    }


class SelectionTests(unittest.TestCase):
    def test_same_frame_beats_separate_equal_quality_observations(self):
        selected = choose_pair(
            {"left": [item(1, edge=40), item(3)], "right": [item(2, edge=40), item(3)]}
        )
        self.assertEqual({v["frame_index"] for v in selected.values()}, {3})

    def test_same_frame_cannot_replace_explicit_direction_constraint(self):
        selected = choose_pair({"left": [item(1, constrained=True), item(3)], "right": [item(3)]})
        self.assertEqual(selected["left"]["frame_index"], 1)

    def test_same_frame_motion_cannot_replace_stable_observation(self):
        selected = choose_pair({"left": [item(1), item(3, stable=False)], "right": [item(3)]})
        self.assertEqual(selected["left"]["frame_index"], 1)

    def test_constraint_is_bound_to_episode_frame_and_verified_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            manifest = root / "manifest.json"
            pose = {
                "orientation_status": "user_constraint_applied",
                "axis_review": {
                    "reviewer_type": "user_direction_feedback",
                    "input_direction_feedback_confirmed": True,
                },
            }
            report.write_text(
                json.dumps(
                    {
                        "episode": "/one",
                        "observations": [{"frame_index": 12, "stickers": {"left": pose}}],
                    }
                )
            )
            manifest.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "episode": "/one",
                                "report": str(report),
                                "report_sha256": digest(report),
                            }
                        ]
                    }
                )
            )
            result = load_constraints(manifest)
            self.assertNotIn("/other", result)
            self.assertEqual(result["/one"][0]["frame_index"], 12)
            self.assertEqual(result["/one"][0]["physical_side"], "left")
            report.write_text("{}")
            with self.assertRaises(ValueError):
                load_constraints(manifest)

    def test_temporal_track_rejects_pose_flip(self):
        camera = Camera(np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]]), (100, 100), "test")
        pose = {
            "image_corners_px": [[30, 30], [70, 30], [70, 60], [30, 60]],
            "object_corners_local_m": np.zeros((4, 3)).tolist(),
            "T_camera_local": np.eye(4).tolist(),
        }
        frames = [np.zeros((50, 50), np.uint8)] * 3

        def flow(a, b, p, *args, **kwargs):
            return p.copy(), np.ones((4, 1), np.uint8), None

        flipped = np.eye(4)
        flipped[0, 0] = flipped[1, 1] = -1
        with (
            patch("vive_correct.batch.selection_policy.cv2.calcOpticalFlowPyrLK", side_effect=flow),
            patch("vive_correct.batch.selection_policy.fixed_pose", return_value=flipped),
        ):
            result = temporal_check(frames, pose, camera)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "pose_jump")


class ReviewIntegrationTests(unittest.TestCase):
    def test_model_rotation_is_ignored_and_same_frame_pair_is_exported(self):
        from types import SimpleNamespace
        from vive_correct.batch.automatic import review_episode, write_json

        camera = Camera(np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]]), (100, 100), "test")
        image = np.full((100, 100, 3), 120, np.uint8)
        transform = np.eye(4)
        transform[2, 3] = 1
        pose = {
            "T_camera_local": transform.tolist(),
            "image_corners_px": [[30, 30], [70, 30], [70, 60], [30, 60]],
            "object_corners_local_m": np.zeros((4, 3)).tolist(),
            "rms_px": 0.1,
        }
        episode = SimpleNamespace(
            path=Path("/episode"), camera=camera, timestamps=np.arange(30) / 30
        )
        config = SimpleNamespace(model="qwen-test", api_key="secret-value", base_url="https://test")

        def transport(config, body):
            text = " ".join(c.get("text", "") for c in body["messages"][0]["content"])
            if "bbox" in text:
                return json.dumps(
                    {
                        s: {"bbox": [250, 250, 750, 650], "evidence": "fixture"}
                        for s in ("left", "right")
                    }
                )
            side = "left" if "candidate_id=left" in text else "right"
            return json.dumps(
                {
                    "decisions": [
                        {
                            "candidate_id": side,
                            "physical_side": side,
                            "is_label": True,
                            "clear": True,
                            "confidence": 0.99,
                            "visible_features": "label",
                            "axis_status": "inferred",
                            "old_from_new": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                        }
                    ]
                }
            )

        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            folder = work / "search/01_episode"
            report = work / "original.json"
            # Different centres avoid duplicate-patch rejection.
            other = json.loads(json.dumps(pose))
            other["image_corners_px"] = [[5, 5], [25, 5], [25, 25], [5, 25]]
            write_json(
                folder / "candidates.json",
                {
                    "episode": "/episode",
                    "candidates": [
                        {"id": s, "side_hint": s, "frame_index": 10, "quality": 1, "pose": p}
                        for s, p in [("left", pose), ("right", other)]
                    ],
                },
            )
            write_json(
                folder / "scan_complete.json",
                {"candidate_pool_sha256": digest(folder / "candidates.json")},
            )
            write_json(report, {"episode": "/episode", "observations": []})
            with (
                patch("vive_correct.batch.automatic.Episode.load", return_value=episode),
                patch(
                    "vive_correct.batch.automatic.frame_windows",
                    side_effect=lambda *args: iter([(10, image, [image[:, :, 0]] * 11)]),
                ),
                patch("vive_correct.batch.automatic.fit_candidate", return_value=pose),
                patch(
                    "vive_correct.batch.automatic.temporal_check",
                    return_value={"passed": False, "reason": "pose_jump"},
                ),
                patch(
                    "vive_correct.batch.automatic.measure_stability", return_value={"passed": True}
                ),
                patch("vive_correct.batch.automatic.read_frame", return_value=image),
                patch("vive_correct.batch.automatic.draw_poses", return_value=image),
            ):
                result = review_episode(
                    {"index": 1, "episode": "/episode", "report": str(report)},
                    work,
                    config,
                    limit=2,
                    first_pass=2,
                    transport=transport,
                )
            output = json.loads(Path(result["report"]).read_text())
            self.assertEqual(result["mode"], "both_hands_same_frame")
            self.assertEqual(output["model_review"]["requested_batches"], 2)
            for p in output["observations"][0]["stickers"].values():
                self.assertIsNone(p["T_camera_local"])
                np.testing.assert_allclose(p["origin_camera_m"], transform[:3, 3])
                self.assertEqual(p["orientation_status"], "axes_not_generated")
                self.assertEqual(p["quality_status"], "verified")
                self.assertFalse(p["temporal_consistency"]["passed"])


class IdentityGuardTests(unittest.TestCase):
    def test_candidate_outside_independently_localized_tracker_is_rejected(self):
        from vive_correct.batch.qwen_review import matches_tracker

        pose = {"image_corners_px": [[10, 80], [20, 80], [20, 90], [10, 90]]}
        locations = {"left": {"bbox": [0, 0, 300, 300]}, "right": None}
        self.assertFalse(matches_tracker(pose, "left", locations, (100, 100)))
        self.assertFalse(matches_tracker(pose, "right", locations, (100, 100)))

    def test_rejected_selection_falls_back_to_checked_candidate(self):
        from types import SimpleNamespace
        from vive_correct.batch.identity_guard import verify_choices

        a, b = item(1, edge=40), item(2)
        for i in (a, b):
            i["pose"]["image_corners_px"] = [[30, 30], [70, 30], [70, 60], [30, 60]]
        episode = SimpleNamespace(camera=SimpleNamespace(resolution=(100, 100)))
        config = SimpleNamespace(model="qwen-test", base_url="https://test", api_key="secret")
        boxes = iter(
            [{"left": None, "right": None}, {"left": {"bbox": [250, 250, 750, 650]}, "right": None}]
        )
        with (
            TemporaryDirectory() as tmp,
            patch(
                "vive_correct.batch.identity_guard.locate_wrist_trackers",
                side_effect=lambda *a, **k: next(boxes),
            ),
        ):
            selected, records, errors, count = verify_choices(
                {"left": [a, b], "right": []},
                episode,
                config,
                Path(tmp),
                lambda e, f: np.full((100, 100, 3), f, np.uint8),
            )
        self.assertEqual(selected["left"]["frame_index"], 2)
        self.assertEqual([r["passed"] for r in records], [False, True])
        self.assertEqual(count, 2)
        self.assertFalse(errors)


class BaselineRetentionTests(unittest.TestCase):
    def test_all_observations_loaded_without_upgrading_review_status(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, manifest = root / "report.json", root / "manifest.json"
            report.write_text(
                json.dumps(
                    {
                        "episode": "/one",
                        "observations": [
                            {
                                "frame_index": 340,
                                "quality_status": "verified",
                                "stickers": {
                                    "right": {"orientation_status": "not_direction_reviewed"}
                                },
                            }
                        ],
                    }
                )
            )
            manifest.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "episode": "/one",
                                "report": str(report),
                                "report_sha256": digest(report),
                            }
                        ]
                    }
                )
            )
            self.assertEqual(load_constraints(manifest), {})
            loaded = load_constraints(manifest, retain_observations=True)["/one"][0]
            self.assertEqual(loaded["pose"]["orientation_status"], "not_direction_reviewed")
            self.assertEqual(loaded["pose"]["quality_status"], "verified")
            self.assertNotIn("constraint_source", loaded)

    def test_motion_baseline_survives_new_candidate_and_same_frame_preference(self):
        baseline = item(550, stable=False, constrained=True)
        baseline["pose"]["baseline_retention"] = {"source_report_sha256": "fixture"}
        selected = choose_pair({"left": [baseline, item(360)], "right": [item(360)]})
        self.assertIs(selected["left"], baseline)

    def test_failed_localization_falls_back_to_baseline(self):
        from types import SimpleNamespace
        from vive_correct.batch.identity_guard import verify_choices

        baseline, candidate = item(1), item(2, edge=40)
        baseline["pose"]["baseline_retention"] = {"source_report_sha256": "fixture"}
        candidate["pose"]["visual_quality"]["sharpness"] = 20
        for value in (baseline, candidate):
            value["pose"]["image_corners_px"] = [[30, 30], [70, 30], [70, 60], [30, 60]]
        with (
            TemporaryDirectory() as tmp,
            patch(
                "vive_correct.batch.identity_guard.locate_wrist_trackers",
                return_value={"left": None, "right": None},
            ),
        ):
            choices, _, _, requests = verify_choices(
                {"left": [baseline, candidate], "right": []},
                SimpleNamespace(camera=SimpleNamespace(resolution=(100, 100))),
                SimpleNamespace(model="test", base_url="https://test", api_key="secret"),
                Path(tmp),
                lambda *args: np.zeros((100, 100, 3), np.uint8),
            )
        self.assertIs(choices["left"], baseline)
        self.assertEqual(requests, 1)
