import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from vive_correct.run_dataset import (
    VIDEO_SUFFIX,
    TIMESTAMP_SUFFIX,
    discover,
    estimate_dataset,
    main,
)
from vive_correct.estimate.find_stable_frame import SelectionConfig


class DatasetRunnerTests(unittest.TestCase):
    def test_rerun_stage_needs_no_model_or_dataset_discovery(self):
        with TemporaryDirectory() as temporary:
            with (
                patch("vive_correct.run_dataset.check_dependencies"),
                patch("vive_correct.run_dataset.export_recordings", return_value=0) as export,
                patch("vive_correct.run_dataset.estimate_dataset") as estimate,
            ):
                self.assertEqual(
                    main(["--out", temporary, "--dataset", "/missing", "--stage", "rerun"]), 0
                )
                export.assert_called_once()
                estimate.assert_not_called()

    def test_baseline_only_skips_credentials_and_preflight(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = self.episode(root / "input", "sample")
            constraints = {str(episode): [{"physical_side": s} for s in ("left", "right")]}
            with (
                patch(
                    "vive_correct.batch.selection_policy.load_constraints", return_value=constraints
                ),
                patch("vive_correct.estimate.openai_vision.load_config") as credentials,
                patch("vive_correct.estimate.openai_vision._post_chat") as network,
                patch("vive_correct.run_dataset.estimate_dataset", return_value=[]),
                patch("vive_correct.run_dataset.scan_manifest"),
                patch("vive_correct.batch.automatic.run_automatic", return_value=0) as repair,
            ):
                self.assertEqual(
                    main(
                        [
                            "--dataset",
                            str(root / "input"),
                            "--out",
                            str(root / "out"),
                            "--axis-baseline",
                            "baseline.json",
                            "--baseline-only",
                        ]
                    ),
                    0,
                )
                credentials.assert_not_called()
                network.assert_not_called()
                self.assertEqual(repair.call_args.kwargs["limit"], 0)

    def test_incomplete_baseline_fails_before_scanning(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.episode(root / "input", "sample")
            with (
                patch("vive_correct.batch.selection_policy.load_constraints", return_value={}),
                patch("vive_correct.run_dataset.estimate_dataset") as estimate,
            ):
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "--dataset",
                            str(root / "input"),
                            "--out",
                            str(root / "out"),
                            "--axis-baseline",
                            "baseline.json",
                            "--baseline-only",
                        ]
                    )
                estimate.assert_not_called()

    def episode(self, dataset, name, video=True):
        episode = dataset / name
        paths = [TIMESTAMP_SUFFIX, Path("calibrations/rectify_params.yaml")]
        if video:
            paths.append(VIDEO_SUFFIX)
        for relative in paths:
            path = episode / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        return episode

    def estimate(self, episode, output, config):
        output.mkdir(parents=True, exist_ok=True)
        image = output / "image.png"
        image.write_bytes(b"fixture")
        report = {
            "mode": "single_hand",
            "observations": [
                {
                    "frame_index": 5,
                    "image": str(image),
                    "overlay": str(image),
                    "stickers": {"left": {}},
                }
            ],
        }
        (output / "sticker_pose_report_new.json").write_text(json.dumps(report))
        return report

    def test_resume_missing_inputs_and_unreviewed_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, output = root / "input", root / "output"
            first = self.episode(dataset, "desk/same")
            self.episode(dataset, "stove/same", video=False)
            self.assertEqual(len(discover(dataset)), 2)
            with patch("vive_correct.run_dataset.estimate", side_effect=self.estimate) as run:
                entries = estimate_dataset(dataset, output, SelectionConfig())
                self.assertEqual([e["status"] for e in entries], ["unreviewed", "missing_input"])
                estimate_dataset(dataset, output, SelectionConfig())
                self.assertEqual(run.call_count, 1)
                (output / "estimates/desk/same/image.png").unlink()
                estimate_dataset(dataset, output, SelectionConfig())
                self.assertEqual(run.call_count, 2)
                (first / VIDEO_SUFFIX).write_text("changed input")
                estimate_dataset(dataset, output, SelectionConfig())
                self.assertEqual(run.call_count, 3)
            manifest = json.loads((output / "scan_manifest.json").read_text())
            self.assertEqual(manifest["entries"][0]["hands"], [])
            self.assertEqual(manifest["entries"][0]["detected_side_hints"], ["left"])
            with self.assertRaisesRegex(ValueError, "settings changed"):
                estimate_dataset(dataset, output, SelectionConfig(stride=5))

    def test_failed_episode_does_not_stop_later_episode(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "input"
            self.episode(dataset, "a")
            self.episode(dataset, "b")

            def run(episode, output, config):
                if episode.name == "a":
                    raise ValueError("bad calibration")
                return self.estimate(episode, output, config)

            with patch("vive_correct.run_dataset.estimate", side_effect=run):
                entries = estimate_dataset(dataset, root / "output", SelectionConfig())
            self.assertEqual([e["status"] for e in entries], ["failed", "unreviewed"])

    def test_finish_requires_review_inputs(self):
        with TemporaryDirectory() as temporary:
            with patch("vive_correct.run_dataset.run_batch") as run:
                with self.assertRaises(SystemExit):
                    main(["--out", temporary, "--stage", "finish"])
                run.assert_not_called()

    def test_gallery_does_not_require_dataset_or_model(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            with (
                patch("vive_correct.batch.bilateral_gallery.build") as build,
                patch("vive_correct.run_dataset.estimate_dataset") as estimate,
            ):
                self.assertEqual(
                    main(["--out", str(output), "--dataset", "/missing", "--stage", "gallery"]), 0
                )
                build.assert_called_once_with(output / "batch")
                estimate.assert_not_called()

    def test_all_uses_coarse_scan_and_independent_model_concurrency(self):
        from types import SimpleNamespace

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            model = SimpleNamespace(model="qwen-test", api_key="secret", base_url="https://test")
            with (
                patch("vive_correct.estimate.openai_vision.load_config", return_value=model),
                patch("vive_correct.estimate.openai_vision._post_chat", return_value='{"ok":true}'),
                patch("vive_correct.run_dataset.estimate_dataset", return_value=[]),
                patch("vive_correct.run_dataset.scan_manifest") as scan,
                patch("vive_correct.batch.automatic.run_automatic", return_value=0) as run,
            ):
                self.assertEqual(main(["--dataset", str(dataset), "--out", str(root / "out")]), 0)
            self.assertEqual(scan.call_args.args[2], 15)
            self.assertEqual(run.call_args.kwargs["fine_stride"], 5)
            self.assertEqual(run.call_args.kwargs["jobs"], 12)
            self.assertEqual(run.call_args.kwargs["model_jobs"], 3)
            self.assertEqual(run.call_args.kwargs["limit"], 24)
