"""The CV arm is offline, explicitly unverified, and uses the same fixed budget."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vive_correct.batch.automatic import review_episode, write_json
from vive_correct.batch.report_io import digest
from vive_correct.run_dataset import main


class AblationTests(unittest.TestCase):
    def test_summary_keeps_missing_samples_and_does_not_compare_different_frame_positions(self):
        from vive_correct.ablation import summarize

        with TemporaryDirectory() as temporary:
            work = Path(temporary)
            write_json(
                work / "inventory.json",
                dict(
                    entries=[
                        dict(index=1, episode="/data/valid", missing_inputs=[]),
                        dict(index=2, episode="/data/missing", missing_inputs=["video"]),
                    ]
                ),
            )
            for mode, frame in (("off", 10), ("on", 20)):
                report = work / mode / "report.json"
                write_json(
                    report,
                    dict(
                        observations=[
                            dict(
                                frame_index=frame,
                                timestamp_s=frame / 30,
                                overlay="image.png",
                                stickers={
                                    "left": dict(
                                        origin_camera_m=[0, 0, 1],
                                        rms_px=0.2,
                                        source_candidate_id=str(frame),
                                        stability={"passed": True},
                                    )
                                },
                            )
                        ],
                        model_review={"evaluated_candidates": 24},
                    ),
                )
                write_json(
                    work / mode / "batch/results/manifest.json",
                    dict(
                        entries=[
                            dict(
                                index=1,
                                episode="/data/valid",
                                report=str(report),
                                status="complete",
                            )
                        ]
                    ),
                )
            with patch("vive_correct.ablation_report.render"):
                result = summarize(work)
            self.assertEqual(result["summary"]["off"]["total_samples"], 2)
            self.assertEqual(result["summary"]["on"]["missing_input"], 1)
            self.assertEqual(result["summary"]["on"]["processed"], 1)
            self.assertTrue(result["entries"][0]["equal_candidate_count"])
            self.assertIsNone(
                result["entries"][0]["comparison"]["left"]["same_frame_origin_delta_m"]
            )
            report = work / "on/report.json"
            rejected = json.loads(report.read_text())
            rejected["model_review"]["warnings"] = [
                dict(kind="content_rejected", candidate_ids=["a", "b", "c", "d"])
            ]
            write_json(report, rejected)
            with patch("vive_correct.ablation_report.render"):
                result = summarize(work)
            self.assertEqual(result["summary"]["on"]["service_rejected_candidates"], 4)
            self.assertEqual(result["summary"]["on"]["incomplete_review_samples"], 1)
            self.assertEqual(result["complete_review_subset"]["sample_count"], 0)

    def test_off_cli_does_not_load_credentials_or_probe(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "dataset/sample/calibrations/rectify_params.yaml"
            calibration.parent.mkdir(parents=True)
            calibration.write_text("fixture")
            with (
                patch("vive_correct.estimate.openai_vision.load_config") as credentials,
                patch("vive_correct.estimate.openai_vision._post_chat") as network,
                patch("vive_correct.run_dataset.estimate_dataset", return_value=[]),
                patch("vive_correct.run_dataset.scan_manifest"),
                patch("vive_correct.batch.automatic.run_automatic", return_value=0) as run,
            ):
                self.assertEqual(
                    main(
                        [
                            "--dataset",
                            str(root / "dataset"),
                            "--out",
                            str(root / "out"),
                            "--qwen-review",
                            "off",
                            "--fixed-candidates",
                        ]
                    ),
                    0,
                )
                credentials.assert_not_called()
                network.assert_not_called()
                self.assertFalse(run.call_args.kwargs["qwen_review"])
                self.assertTrue(run.call_args.kwargs["fixed_candidates"])
                self.assertEqual(run.call_args.kwargs["guided_frames"], 0)

    def test_off_skips_both_model_stages_and_does_not_early_stop(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            work = Path(temporary)
            ep = work / "episode"
            report = work / "input.json"
            write_json(report, dict(episode=str(ep), observations=[]))
            pose = dict(
                image_corners_px=[[10, 10], [60, 10], [60, 40], [10, 40]],
                origin_camera_m=[0, 0, 0.5],
                T_camera_local=None,
                axes_generated=False,
                rms_px=0.1,
                low_resolution=False,
            )
            candidates = [
                dict(
                    id=str(i),
                    frame_index=i,
                    side_hint="left" if i % 2 else "right",
                    quality=100 - i,
                    pose=pose,
                )
                for i in range(10)
            ]
            folder = work / "search/01_episode"
            write_json(folder / "candidates.json", dict(episode=str(ep), candidates=candidates))
            write_json(
                folder / "scan_complete.json",
                dict(candidate_pool_sha256=digest(folder / "candidates.json")),
            )
            episode = SimpleNamespace(
                camera=SimpleNamespace(resolution=(100, 100)), timestamps=np.arange(10, dtype=float)
            )
            image = np.zeros((100, 100, 3), np.uint8)
            module = "vive_correct.batch.automatic."
            for name, value in [
                ("Episode.load", episode),
                ("diverse_candidates", candidates),
                ("measure_stability", {"passed": True}),
                ("temporal_check", {"passed": True}),
                ("visual_quality", dict(min_edge_px=30, sharpness=100)),
                ("read_frame", image),
                ("draw_poses", image),
                ("save_image", None),
            ]:
                stack.enter_context(patch(module + name, return_value=value))
            stack.enter_context(
                patch(module + "frame_windows", side_effect=lambda *a: iter([(0, image, [])]))
            )
            stack.enter_context(
                patch(module + "fit_candidate", side_effect=lambda *a: deepcopy(pose))
            )
            reviewer = stack.enter_context(
                patch(module + "review_candidates", side_effect=AssertionError("network"))
            )
            locator = stack.enter_context(
                patch(module + "verify_choices", side_effect=AssertionError("network"))
            )
            refinement = stack.enter_context(
                patch(module + "refinement_pool", side_effect=AssertionError("adaptive search"))
            )
            config = SimpleNamespace(model="cv-only", base_url="", api_key="unused")
            result = review_episode(
                dict(index=1, episode=str(ep), report=str(report)),
                work,
                config,
                qwen_review=False,
                fixed_candidates=True,
                limit=10,
                first_pass=2,
                batch_size=2,
            )
            reviewer.assert_not_called()
            locator.assert_not_called()
            refinement.assert_not_called()
            self.assertEqual(result["status"], "cv_only_unverified")
            output = json.loads(Path(result["report"]).read_text())
            self.assertEqual(output["identity"], "cv_image_half_unverified")
            self.assertIsNone(output["model"])
            self.assertEqual(output["model_review"]["evaluated_candidates"], 10)
            self.assertEqual(output["model_review"]["model_candidates"], 0)
            self.assertEqual(output["model_review"]["localization_requests"], 0)
            for obs in output["observations"]:
                for item in obs["stickers"].values():
                    self.assertEqual(item["identity_review"]["source"], "cv_image_half_unverified")
