import unittest
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from vive_correct.batch.guided_search import guided_pool, proposals_in_trackers, search_frames
from vive_correct.estimate.geometry import Camera


class GuidedSearchTests(unittest.TestCase):
    def test_bad_guided_frame_does_not_discard_other_candidates_and_cache_resumes(self):
        with TemporaryDirectory() as temporary:
            cache = Path(temporary)
            episode = SimpleNamespace(timestamps=np.arange(100), camera=None)
            config = SimpleNamespace(model="gpt-6-astra", base_url="test", api_key="secret")

            def read_frame(ep, frame):
                if frame == 5:
                    raise ValueError("decode failed secret")
                return np.full((10, 10, 3), frame, np.uint8)

            with (
                patch(
                    "vive_correct.batch.guided_search.locate_wrist_trackers",
                    return_value={"left": None, "right": None},
                ) as locate,
                patch(
                    "vive_correct.batch.guided_search.proposals_in_trackers",
                    side_effect=lambda im, loc, cam, frame: [{"frame_index": frame}],
                ),
            ):
                result = guided_pool(episode, cache, config, read_frame, count=3)
                self.assertEqual(len(result["candidates"]), 2)
                self.assertEqual(result["requests"], 2)
                self.assertEqual(result["errors"][0]["frame_index"], 5)
                self.assertNotIn("secret", result["errors"][0]["reason"])
                locate.reset_mock()
                resumed = guided_pool(episode, cache, config, read_frame, count=3)
                self.assertEqual(resumed["candidates"], result["candidates"])
                self.assertEqual(resumed["requests"], 0)
                locate.assert_not_called()
                self.assertEqual(list(cache.glob("*.tmp")), [])

    def test_recovery_pair_is_reviewed_and_stops_before_background_budget(self):
        from vive_correct.batch.automatic import review_episode, write_json
        from vive_correct.batch.report_io import digest
        from vive_correct.batch.selection_policy import choose_pair

        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            ep = root / "episode"
            pose = dict(
                image_corners_px=[[10, 10], [60, 10], [60, 40], [10, 40]],
                origin_camera_m=[0, 0, 0.5],
                axes_generated=False,
                T_camera_local=None,
                rms_px=0.1,
                low_resolution=False,
            )

            def candidate(name, frame, side):
                return dict(
                    id=name, frame_index=frame, side_hint=side, quality=1, pose=deepcopy(pose)
                )

            coarse = [candidate("bad1", 1, "left"), candidate("bad2", 2, "right")]
            guided = [candidate("good_left", 10, "left"), candidate("good_right", 11, "right")]
            fine = [candidate("background", 20, "left")]
            source = root / "input.json"
            write_json(source, dict(episode=str(ep), observations=[]))
            folder = root / "search/01_episode"
            write_json(folder / "candidates.json", dict(episode=str(ep), candidates=coarse))
            write_json(
                folder / "scan_complete.json",
                dict(candidate_pool_sha256=digest(folder / "candidates.json")),
            )
            episode = SimpleNamespace(
                path=ep, camera=SimpleNamespace(resolution=(100, 100)), timestamps=np.arange(30)
            )
            image = np.zeros((100, 100, 3), np.uint8)
            module = "vive_correct.batch.automatic."
            for name, value in [
                ("Episode.load", episode),
                ("measure_stability", {"passed": True}),
                ("temporal_check", {"passed": True}),
                ("visual_quality", dict(min_edge_px=30, sharpness=100)),
                ("read_frame", image),
                ("draw_poses", image),
                ("save_image", None),
                ("refinement_pool", {"candidates": fine}),
            ]:
                stack.enter_context(patch(module + name, return_value=value))
            stack.enter_context(patch(module + "diverse_candidates", side_effect=lambda c, n: c))
            stack.enter_context(
                patch(module + "frame_windows", side_effect=lambda *a: iter([(0, image, [])]))
            )
            stack.enter_context(
                patch(module + "fit_candidate", side_effect=lambda *a: deepcopy(pose))
            )

            def review(items, *args, **kwargs):
                return [
                    dict(
                        candidate_id=i["id"],
                        is_label=i["id"].startswith("good"),
                        clear=True,
                        physical_side=i["candidate"]["side_hint"],
                        confidence=0.99,
                        visible_features="test",
                    )
                    for i in items
                ]

            reviewer = stack.enter_context(patch(module + "review_candidates", side_effect=review))
            stack.enter_context(
                patch(
                    module + "verify_choices",
                    side_effect=lambda a, *args, **kw: (choose_pair(a), [], [], 0),
                )
            )
            recovery = stack.enter_context(
                patch(
                    "vive_correct.batch.guided_search.guided_pool",
                    return_value=dict(candidates=guided, requests=2, frames=[], errors=[]),
                )
            )
            result = review_episode(
                dict(index=1, episode=str(ep), report=str(source)),
                root,
                SimpleNamespace(model="gpt-6-astra", base_url="test", api_key="unused"),
                first_pass=2,
                batch_size=2,
                limit=10,
                guided_frames=6,
            )
            self.assertEqual(result["hands"], ["left", "right"])
            self.assertEqual(reviewer.call_count, 2)
            recovery.assert_called_once()
            report = json.loads(Path(result["report"]).read_text())
            self.assertTrue(report["model_review"]["refined_search"])
            self.assertEqual(report["model_review"]["evaluated_candidates"], 4)
            self.assertEqual(report["model_review"]["stop_reason"], "bilateral_stable")
            self.assertEqual(report["model_review"]["missing_hands"], [])

    def test_uniform_frames_are_bounded_unique_and_include_late_video(self):
        self.assertEqual(search_frames(12, 6), [5, 6])
        self.assertEqual(search_frames(10, 6), [])
        frames = search_frames(1000, 6)
        self.assertEqual(len(frames), 6)
        self.assertEqual((frames[0], frames[-1]), (5, 994))

    def test_boxes_are_only_search_regions_and_never_synthetic_labels(self):
        image = np.full((400, 600, 3), 30, np.uint8)
        camera = Camera(np.array([[800.0, 0, 300], [0, 800, 200], [0, 0, 1]]), (600, 400), "test")
        locations = {"left": {"bbox": [200, 550, 400, 750]}, "right": None}
        self.assertEqual(proposals_in_trackers(image, locations, camera, 10), [])
        cv2.rectangle(image, (140, 240), (180, 264), (230, 230, 230), -1)
        # Another white rectangle outside the localized tracker must not be proposed.
        cv2.rectangle(image, (440, 240), (480, 264), (230, 230, 230), -1)
        found = proposals_in_trackers(image, locations, camera, 10)
        self.assertEqual(len(found), 1)
        center = np.asarray(found[0]["pose"]["image_corners_px"]).mean(0)
        np.testing.assert_allclose(center, [160, 252], atol=1)
        self.assertGreater(np.linalg.norm(center - [180, 260]), 15)
        self.assertNotIn("physical_side", found[0])
        self.assertEqual(found[0]["proposal_source"], "tracker_guided_pixel_detection")

    def test_partial_tracker_box_can_contain_complete_label_but_not_clipped_label(self):
        camera = Camera(np.array([[800.0, 0, 300], [0, 800, 200], [0, 0, 1]]), (600, 400), "test")
        locations = {"left": {"bbox": [200, 850, 400, 1000]}, "right": None}
        complete = np.full((400, 600, 3), 30, np.uint8)
        cv2.rectangle(complete, (140, 365), (180, 389), (230, 230, 230), -1)
        self.assertEqual(len(proposals_in_trackers(complete, locations, camera, 10)), 1)
        clipped = np.full_like(complete, 30)
        cv2.rectangle(clipped, (140, 380), (180, 410), (230, 230, 230), -1)
        self.assertEqual(proposals_in_trackers(clipped, locations, camera, 10), [])
