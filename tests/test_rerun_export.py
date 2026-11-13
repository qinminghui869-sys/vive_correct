"""Export publication and failure handling without launching a viewer."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from vive_correct.batch.report_io import digest
from vive_correct.rerun_export import DEPTH_KEY, export_recordings


class RerunExportTests(unittest.TestCase):
    def fixture(self, root):
        episode = root / "input"
        data = episode / "data/chunk-000/episode_000000"
        data.mkdir(parents=True)
        pq.write_table(pa.table({"timestamp": [0.0, 0.03, 0.06]}), data / f"{DEPTH_KEY}.parquet")
        manifest = root / "batch/results/manifest.json"
        manifest.parent.mkdir(parents=True)
        report = manifest.parent / "observations.json"
        report.write_text("{}")
        manifest.write_text(
            json.dumps(
                {
                    "entries": [
                        dict(
                            index=1,
                            episode=str(episode),
                            report="observations.json",
                            report_sha256=digest(report),
                        )
                    ]
                }
            )
        )
        return report

    def render(self, command, **kwargs):
        self.assertEqual(command[command.index("--pose-euler-sequence") + 1], "zyx")
        self.assertIn("--pose-radians", command)
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"recording")
        output.with_suffix(".json").write_text(
            json.dumps({"frames": 2, "skipped_missing_head_frames": [1]})
        )

    def test_success_publishes_complete_recording_with_skips(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with (
                patch("vive_correct.rerun_export.check_dependencies"),
                patch(
                    "ego_data.vive_origin_alignment.build_alignment",
                    return_value={"pose_euler_sequence": "zyx", "pose_degrees": False},
                ),
                patch("vive_correct.rerun_export.subprocess.run", side_effect=self.render),
            ):
                self.assertEqual(export_recordings(root, root / "mount.json"), 0)
            row = json.loads((root / "rerun/manifest.json").read_text())["entries"][0]
            self.assertEqual(row["frames"], 2)
            self.assertEqual(row["skipped_frames"], [1])
            self.assertTrue(Path(row["recording"]).is_file())
            self.assertFalse(list((root / "rerun").rglob("*.partial.rrd")))

    def test_changed_report_is_rejected_before_alignment(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root).write_text('{"changed": true}')
            with (
                patch("vive_correct.rerun_export.check_dependencies"),
                patch("ego_data.vive_origin_alignment.build_alignment") as build,
            ):
                self.assertEqual(export_recordings(root, root / "mount.json"), 1)
                build.assert_not_called()

    def test_renderer_failure_does_not_publish_or_open(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with (
                patch("vive_correct.rerun_export.check_dependencies"),
                patch(
                    "ego_data.vive_origin_alignment.build_alignment",
                    return_value={"pose_euler_sequence": "xyz", "pose_degrees": True},
                ),
                patch(
                    "vive_correct.rerun_export.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, "renderer"),
                ),
                patch("vive_correct.rerun_export.launch") as viewer,
            ):
                self.assertEqual(export_recordings(root, root / "mount.json", open_viewer=True), 1)
                viewer.assert_not_called()
                self.assertFalse(list((root / "rerun").rglob("recording.rrd")))
