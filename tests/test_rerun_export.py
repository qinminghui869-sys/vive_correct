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
from vive_correct.export.rerun_export import DEPTH_KEY, export_recordings
from vive_correct.estimate.geometry import VIDEO_KEY


class RerunExportTests(unittest.TestCase):
    def fixture(self, root):
        episode = root / "input"
        data = episode / "data/chunk-000/episode_000000"
        data.mkdir(parents=True)
        pq.write_table(pa.table({"timestamp": [0.0, 0.03, 0.06]}), data / f"{DEPTH_KEY}.parquet")
        pq.write_table(pa.table({"timestamp": [0.0, 0.03, 0.06]}), data / f"{VIDEO_KEY}.parquet")
        manifest = root / "batch/results/manifest.json"
        manifest.parent.mkdir(parents=True)
        report = manifest.parent / "observations.json"
        report.write_text(
            json.dumps(
                dict(intrinsics=dict(left=dict(fx=100, fy=100, cx=32, cy=24)), max_depth_m=4)
            )
        )
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
                patch("vive_correct.export.rerun_export.check_dependencies"),
                patch(
                    "ego_data.vive_origin_alignment.build_alignment",
                    return_value={"pose_euler_sequence": "zyx", "pose_degrees": False},
                ),
                patch("vive_correct.export.rerun_export.subprocess.run", side_effect=self.render),
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
                patch("vive_correct.export.rerun_export.check_dependencies"),
                patch("ego_data.vive_origin_alignment.build_alignment") as build,
            ):
                self.assertEqual(export_recordings(root, root / "mount.json"), 1)
                build.assert_not_called()

    def test_renderer_failure_does_not_publish_or_open(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with (
                patch("vive_correct.export.rerun_export.check_dependencies"),
                patch(
                    "ego_data.vive_origin_alignment.build_alignment",
                    return_value={"pose_euler_sequence": "xyz", "pose_degrees": True},
                ),
                patch(
                    "vive_correct.export.rerun_export.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, "renderer"),
                ),
                patch("vive_correct.export.rerun_export.launch") as viewer,
            ):
                self.assertEqual(export_recordings(root, root / "mount.json", open_viewer=True), 1)
                viewer.assert_not_called()
                self.assertFalse(list((root / "rerun").rglob("recording.rrd")))

    def test_mode_selects_matching_timeline_and_frame_coverage(self):
        for missing in (None, "mask", "metadata", "both", "depth_video"):
            with self.subTest(missing=missing), TemporaryDirectory() as temporary:
                root = Path(temporary) / "双手肌电 + MANUS"
                root.mkdir()
                self.fixture(root)
                dataset = root / "input"
                data = dataset / "data/chunk-000"
                # Force indexed episode layout and distinct RGB/depth counts.
                (data / "episode_000001").mkdir()
                episode = data / "episode_000000"
                pq.write_table(
                    pa.table({"timestamp": [0.0, 0.03, 0.06, 0.09, 0.12]}),
                    episode / f"{VIDEO_KEY}.parquet",
                )
                manifest = root / "batch/results/manifest.json"
                content = json.loads(manifest.read_text())
                content["entries"][0]["episode"] = str(episode)
                manifest.write_text(json.dumps(content))
                ann = dataset / "annotation/chunk-000" / DEPTH_KEY / episode.name
                ann.mkdir(parents=True)
                if missing not in ("metadata", "both"):
                    (ann / "fast_foundation_stereo_video_meta.json").write_text(
                        json.dumps(
                            dict(
                                intrinsics=dict(left=dict(fx=100, fy=100, cx=32, cy=24)),
                                max_depth_m=4,
                            )
                        )
                    )
                if missing not in ("mask", "both"):
                    (ann / "valid_mask_ffv1.mkv").touch()
                for key in (VIDEO_KEY, DEPTH_KEY):
                    video = dataset / "videos/chunk-000" / key / f"{episode.name}.mp4"
                    video.parent.mkdir(parents=True)
                    if key != DEPTH_KEY or missing != "depth_video":
                        video.touch()
                depth = missing in (None, "mask")
                expected = 3 if depth else 5

                def render(command, **kwargs):
                    self.assertEqual("--rgb-only" in command, not depth)
                    self.assertEqual(command[command.index("--dataset") + 1], str(episode))
                    self.assertEqual(command[command.index("--end") + 1], str(expected - 1))
                    output = Path(command[command.index("--output") + 1])
                    output.write_bytes(b"recording")
                    output.with_suffix(".json").write_text(
                        json.dumps({"frames": expected, "skipped_missing_head_frames": []})
                    )

                with (
                    patch("vive_correct.export.rerun_export.check_dependencies"),
                    patch(
                        "ego_data.vive_origin_alignment.build_alignment",
                        return_value={"pose_euler_sequence": "xyz", "pose_degrees": True},
                    ),
                    patch("vive_correct.export.rerun_export.subprocess.run", side_effect=render),
                ):
                    self.assertEqual(export_recordings(root, root / "mount.json"), 0)
                row = json.loads((root / "rerun/manifest.json").read_text())["entries"][0]
                self.assertEqual(row["frames"], expected)

    def test_incomplete_render_is_not_published(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)

            def incomplete(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(b"partial")
                output.with_suffix(".json").write_text(
                    json.dumps({"frames": 1, "skipped_missing_head_frames": []})
                )

            with (
                patch("vive_correct.export.rerun_export.check_dependencies"),
                patch(
                    "ego_data.vive_origin_alignment.build_alignment",
                    return_value={"pose_euler_sequence": "xyz", "pose_degrees": True},
                ),
                patch("vive_correct.export.rerun_export.subprocess.run", side_effect=incomplete),
            ):
                self.assertEqual(export_recordings(root, root / "mount.json"), 1)
            self.assertFalse(list((root / "rerun").rglob("recording.rrd")))
