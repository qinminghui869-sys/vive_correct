import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from vive_correct.cv_evaluation import audit_best, audit_tracks, main, score_frames


class EvaluationTests(unittest.TestCase):
    def test_unreviewed_frames_do_not_become_ground_truth(self):
        with patch("vive_correct.cv_evaluation.cv2.VideoCapture", side_effect=AssertionError):
            result = score_frames(Path("/data"), [{"human_reviewed": False}], None)
        self.assertEqual(result["frames"], 0)
        self.assertIsNone(result["precision"])
        self.assertEqual(result["acceptance"], "pending")

    def test_video_disjoint_split_is_enforced(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"episode": "desk/a", "split": split} for split in ("tuning", "holdout")
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "leakage"):
                main(
                    [
                        "score",
                        "--dataset",
                        str(root),
                        "--out",
                        str(root / "out"),
                        "--annotations",
                        str(path),
                    ]
                )

    def test_best_and_tracking_audits_expose_wrong_sides(self):
        q = [[10, 10], [50, 10], [50, 40], [10, 40]]
        annotations = [
            dict(
                episode="desk/a",
                frame_index=i,
                human_reviewed=True,
                labels=[dict(side=side, quad=q)],
            )
            for i, side in enumerate(("left", "right"))
        ]
        report = dict(
            episode="/data/desk/a",
            track_observations=[dict(frame_index=i, labels={"left": q}) for i in range(2)],
        )
        tracking = audit_tracks(report, annotations)
        self.assertEqual(tracking["identity_switches"], 1)
        self.assertEqual(tracking["wrong_side"], 1)
        summary = {
            "episodes": [
                dict(
                    episode="/data/desk/a",
                    hands={"left": dict(frame_index=1, pose={"image_corners_px": q})},
                )
            ]
        }
        self.assertEqual(audit_best(summary, annotations)["wrong_side"], 1)

    def test_one_to_one_matches_and_corner_winding(self):
        image = np.zeros((80, 80, 3), np.uint8)
        quad = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], float)
        annotations = [
            dict(
                episode="desk/a",
                frame_index=0,
                human_reviewed=True,
                labels=[dict(side="left", quad=quad[::-1].tolist())],
            )
        ]
        with patch("vive_correct.cv_evaluation.cv2.VideoCapture") as capture:
            capture.return_value.read.return_value = (True, image)
            result = score_frames(
                Path("/data"), annotations, lambda *args, **kwargs: {"left": [quad, quad]}
            )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["median_corner_error_px"], 0)
