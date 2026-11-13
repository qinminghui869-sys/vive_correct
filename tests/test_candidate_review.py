"""Identity review must not silently accept unstable or stale candidates."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from vive_correct.batch.candidate_pool import scan
from vive_correct.estimate.find_stable_frame import Episode, fit_candidate
from vive_correct.estimate.geometry import Camera
from vive_correct.estimate.label_detection import detect_labels
from vive_correct.batch.reviewed_reconstruction import load_choices


class CandidateReviewTests(unittest.TestCase):
    def test_roi_boundary_cannot_complete_a_clipped_label(self):
        image = np.zeros((200, 200, 3), np.uint8)
        image[80:126, 50:121] = 230
        complete = detect_labels(image, {"left": [0, 0, 200, 200]})
        clipped = detect_labels(image, {"left": [80, 40, 180, 180]})
        self.assertTrue(complete["left"])
        self.assertFalse(clipped["left"])

    def test_clipped_label_is_rejected_before_geometric_fit(self):
        camera = Camera(np.eye(3), (100, 100), "fixture")
        image = np.zeros((100, 100, 3), np.uint8)
        for quad in (
            [[20, 80], [60, 80], [60, 99], [20, 99]],
            [[0, 20], [40, 20], [40, 50], [0, 50]],
        ):
            with self.assertRaisesRegex(ValueError, "boundary"):
                fit_candidate(image, quad, camera)

    def test_scan_continues_after_first_pair_and_preserves_alternatives(self):
        camera = Camera(np.eye(3), (100, 100), "fixture")
        episode = Episode(Path("/episode"), Path("/video"), camera, np.arange(100) / 30)
        image = np.full((100, 100, 3), 120, np.uint8)
        quad = [[20, 20], [60, 20], [60, 50], [20, 50]]
        pose = {"image_corners_px": quad, "rms_px": 0.1}
        windows = [(10, image, []), (30, image, [])]
        with (
            TemporaryDirectory() as tmp,
            patch("vive_correct.batch.candidate_pool.Episode.load", return_value=episode),
            patch("vive_correct.batch.candidate_pool.frame_windows", return_value=iter(windows)),
            patch(
                "vive_correct.batch.candidate_pool.detect_labels",
                return_value={"left": [quad], "right": [quad]},
            ),
            patch("vive_correct.batch.candidate_pool.fit_candidate", return_value=pose),
            patch(
                "vive_correct.batch.candidate_pool.measure_stability", return_value={"passed": True}
            ),
        ):
            report = scan("/episode", tmp)
        self.assertEqual(report["examined_windows"], 2)
        self.assertEqual(len(report["candidates"]), 4)
        self.assertEqual({c["frame_index"] for c in report["candidates"]}, {10, 30})
        self.assertEqual(report["identity"], "unreviewed")

    def make_review(self, root, *, passed=True):
        pool = root / "candidates.json"
        pool.write_text(
            json.dumps(
                {
                    "episode": "/episode",
                    "candidates": [
                        {
                            "id": "a",
                            "frame_index": 10,
                            "side_hint": "right",
                            "stability": {"passed": passed},
                        }
                    ],
                }
            )
        )
        review = root / "review.json"
        review.write_text(
            json.dumps(
                {
                    "candidate_pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
                    "accepted": [{"candidate_id": "a", "side": "left"}],
                }
            )
        )
        return pool, review

    def test_review_can_correct_image_side_hint(self):
        with TemporaryDirectory() as tmp:
            pool, review = self.make_review(Path(tmp))
            _, frames = load_choices(pool, review)
        self.assertEqual(set(frames[10]), {"left"})

    def test_modified_pool_invalidates_old_review(self):
        with TemporaryDirectory() as tmp:
            pool, review = self.make_review(Path(tmp))
            pool.write_text(pool.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_choices(pool, review)

    def test_identity_review_cannot_override_stability_failure(self):
        with TemporaryDirectory() as tmp:
            pool, review = self.make_review(Path(tmp), passed=False)
            with self.assertRaisesRegex(ValueError, "unstable"):
                load_choices(pool, review)


if __name__ == "__main__":
    unittest.main()
