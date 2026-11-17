"""Fast defaults, explicit full scan, and read-only planning stay consistent."""

import io
import json
import os
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vive_correct.cv_features import run_feature_dataset
from vive_correct.dataset import VIDEO_SUFFIX
from vive_correct.estimate.cv_config import CVConfig
from vive_correct.run_dataset import main


class FastCLITests(unittest.TestCase):
    def test_cv_default_skips_tracking_and_dry_run_writes_nothing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "dataset/scene/episode" / VIDEO_SUFFIX
            video.parent.mkdir(parents=True)
            video.touch()
            output = root / "out"
            args = ["--profile", "cv", "--dataset", str(root / "dataset"), "--out", str(output)]
            stream = io.StringIO()
            with (
                redirect_stdout(stream),
                patch("vive_correct.run_dataset.estimate_dataset") as tracking,
            ):
                self.assertEqual(main([*args, "--dry-run"]), 0)
                tracking.assert_not_called()
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan["mode"], "cv_features")
            self.assertEqual(plan["search_mode"], "fast")
            self.assertFalse(output.exists())
            output.mkdir()
            (output / "feature_plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "Feature settings or inputs changed"):
                run_feature_dataset(
                    SimpleNamespace(
                        episodes_file=None,
                        coarse_stride=15,
                        max_frames=None,
                        feature_search="full",
                        dry_run=False,
                    ),
                    root / "dataset",
                    output,
                    CVConfig.load(),
                )
            with patch("vive_correct.cv_features.run_feature_dataset", return_value=0) as features:
                self.assertEqual(main([*args, "--feature-search", "full"]), 0)
                self.assertEqual(features.call_args.args[0].feature_search, "full")
                self.assertEqual(features.call_args.args[0].stage, "features")
            with patch("vive_correct.cv_dataset.run_cv_dataset", return_value=0) as tracking:
                self.assertEqual(main([*args, "--stage", "all"]), 0)
                tracking.assert_called_once()

    def test_shell_default_and_explicit_stage_override(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_cv.sh"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "python3"
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            for extra, expected in (
                ([], "features"),
                (["--stage", "all", "--feature-search", "full"], "all"),
            ):
                result = subprocess.run(
                    ["bash", str(script), str(root), str(root / "out"), *extra],
                    env=dict(os.environ, PATH=f"{root}:{os.environ['PATH']}"),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                arguments = result.stdout.splitlines()
                stages = [arguments[i + 1] for i, arg in enumerate(arguments) if arg == "--stage"]
                self.assertEqual(stages[-1], expected)
