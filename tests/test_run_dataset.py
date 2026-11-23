import json
import io
from contextlib import redirect_stdout
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
    select_episodes,
    check_run_plan,
)
from vive_correct.estimate.find_stable_frame import SelectionConfig


class DatasetRunnerTests(unittest.TestCase):
    def test_adopting_estimates_rejects_stale_inputs_before_review(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = dict(dataset="/data", selection={"window": 11}, source_hashes={"x": "v1"})
            plan = dict(
                settings,
                episodes=[dict(relative_episode="sample", input_signature={"video": [10, 1]})],
            )
            state = dict(
                settings=settings,
                entries=[dict(relative_episode="sample", input_signature={"video": [10, 1]})],
            )
            (root / "batch_state.json").write_text(json.dumps(state))
            check_run_plan(root, plan)
            state["entries"][0]["input_signature"]["video"] = [10, 2]
            (root / "batch_state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                check_run_plan(root, plan)
            self.assertFalse((root / "run_plan.json").exists())

    def test_explicit_subset_is_ordered_and_rejects_duplicates_or_unknown_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            a = self.episode(dataset, "desk/a")
            b = self.episode(dataset, "dining/b")
            selection = root / "episodes.json"
            selection.write_text(json.dumps(["dining/b", "desk/a"]))
            self.assertEqual(select_episodes(dataset, selection), [b, a])
            for names in (["desk/a", "desk/a"], ["../elsewhere"], [], {"episode": "desk/a"}):
                selection.write_text(json.dumps(names))
                with self.assertRaises(ValueError):
                    select_episodes(dataset, selection)

    def test_subset_cannot_silently_change_on_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "input"
            self.episode(dataset, "a")
            self.episode(dataset, "b")
            selection = root / "episodes.json"
            selection.write_text('["b"]')
            with patch("vive_correct.run_dataset.estimate", side_effect=self.estimate) as run:
                entries = estimate_dataset(
                    dataset, root / "out", SelectionConfig(), episodes_file=selection
                )
                self.assertEqual([e["relative_episode"] for e in entries], ["b"])
                self.assertEqual(run.call_count, 1)
                selection.write_text('["a"]')
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    estimate_dataset(
                        dataset, root / "out", SelectionConfig(), episodes_file=selection
                    )

    def test_codex_full_pipeline_passes_exact_model_and_transport_without_api_key(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            episode = self.episode(dataset, "desk/sample")
            with (
                patch("vive_correct.batch.codex_transport.CodexTransport") as transport_class,
                patch("vive_correct.estimate.openai_vision.load_config") as credentials,
                patch("vive_correct.estimate.openai_vision._post_chat") as network,
                patch("vive_correct.run_dataset.estimate_dataset", return_value=[]) as estimate,
                patch("vive_correct.run_dataset.scan_manifest"),
                patch("vive_correct.batch.automatic.run_automatic", return_value=0) as repair,
            ):
                transport_class.return_value.return_value = '{"ok":true}'
                argv = [
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(root / "out"),
                    "--profile",
                    "astra",
                ]
                self.assertEqual(main(argv), 0)
                credentials.assert_not_called()
                network.assert_not_called()
                self.assertEqual(repair.call_args.args[1].model, "gpt-6-astra")
                self.assertIs(repair.call_args.kwargs["transport"], transport_class.return_value)
                self.assertFalse(repair.call_args.kwargs["fixed_candidates"])
                for key, value in dict(
                    guided_frames=6,
                    fine_stride=3,
                    first_pass=4,
                    batch_size=4,
                    limit=24,
                    model_jobs=4,
                    jobs=10,
                    timeout=300,
                    retries=2,
                ).items():
                    self.assertEqual(repair.call_args.kwargs[key], value)
                plan = json.loads((root / "out/run_plan.json").read_text())
                self.assertEqual(plan["model"], "gpt-6-astra")
                self.assertEqual(plan["episodes"][0]["relative_episode"], "desk/sample")
                # Identical settings resume; changes fail before any model call or estimate write.
                self.assertEqual(main(argv), 0)
                for override in (
                    ["--review-confidence", "0.9"],
                    ["--codex-model", "other"],
                    ["--tracker-guided-frames", "3"],
                    ["--first-pass", "2"],
                ):
                    with self.subTest(override=override):
                        transport_class.return_value.reset_mock()
                        estimate.reset_mock()
                        with self.assertRaises(SystemExit):
                            main(argv + override)
                        transport_class.return_value.assert_not_called()
                        estimate.assert_not_called()
                (episode / VIDEO_SUFFIX).write_text("changed input")
                with self.assertRaises(SystemExit):
                    main(argv)
                self.assertEqual(json.loads((root / "out/run_plan.json").read_text()), plan)

    def test_dry_run_is_offline_and_cli_overrides_profile(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.episode(root / "data", "desk/sample")
            self.episode(root / "data", "stove/missing", video=False)
            with (
                patch("vive_correct.batch.codex_transport.CodexTransport") as codex,
                patch("vive_correct.estimate.openai_vision.load_config") as credentials,
                patch("vive_correct.run_dataset.estimate_dataset") as estimate,
            ):
                for flags, guided in (
                    ([], 0),
                    (["--fixed-candidates"], 0),
                    (["--model-review", "off"], 0),
                    (["--qwen-review", "off"], 0),
                ):
                    captured = io.StringIO()
                    with redirect_stdout(captured):
                        self.assertEqual(
                            main(
                                [
                                    "--dataset",
                                    str(root / "data"),
                                    "--out",
                                    str(root / "out"),
                                    "--dry-run",
                                    "--review-batch-size",
                                    "2",
                                    "--jobs",
                                    "1",
                                    *flags,
                                ]
                            ),
                            0,
                        )
                    plan = json.loads(captured.getvalue())
                    self.assertEqual(plan["profile"], "legacy")
                    self.assertEqual(plan["parameters"]["transport"], "api")
                    self.assertEqual(plan["parameters"]["review_batch_size"], 2)
                    self.assertEqual(plan["parameters"]["jobs"], 1)
                    self.assertEqual(plan["parameters"]["tracker_guided_frames"], guided)
                    self.assertIsNone(plan["episodes"][1]["input_signature"][str(VIDEO_SUFFIX)])
                self.assertFalse((root / "out").exists())
                codex.assert_not_called()
                credentials.assert_not_called()
                estimate.assert_not_called()
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "--dataset",
                            str(root / "data"),
                            "--out",
                            str(root / "out"),
                            "--dry-run",
                            "--fixed-candidates",
                            "--tracker-guided-frames",
                            "6",
                        ]
                    )

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
            baseline = root / "baseline.json"
            baseline.write_text("{}")
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
                            str(baseline),
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
            self.episode(dataset, "sample")
            model = SimpleNamespace(model="qwen-test", api_key="secret", base_url="https://test")
            with (
                patch("vive_correct.estimate.openai_vision.load_config", return_value=model),
                patch("vive_correct.estimate.openai_vision._post_chat", return_value='{"ok":true}'),
                patch("vive_correct.run_dataset.estimate_dataset", return_value=[]),
                patch("vive_correct.run_dataset.scan_manifest") as scan,
                patch("vive_correct.batch.automatic.run_automatic", return_value=0) as run,
            ):
                self.assertEqual(
                    main(
                        [
                            "--dataset",
                            str(dataset),
                            "--out",
                            str(root / "out"),
                            "--profile",
                            "legacy",
                        ]
                    ),
                    0,
                )
            self.assertEqual(scan.call_args.args[2], 15)
            self.assertEqual(run.call_args.kwargs["fine_stride"], 5)
            self.assertEqual(run.call_args.kwargs["jobs"], 12)
            self.assertEqual(run.call_args.kwargs["model_jobs"], 3)
            self.assertEqual(run.call_args.kwargs["limit"], 24)
