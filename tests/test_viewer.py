import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from vive_correct.viewer import launch, recordings_at


class ViewerTests(unittest.TestCase):
    def test_html_opens_existing_recordings_without_partial_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "review_index.html").write_text("preview")
            recording = root / "recording.rrd"
            recording.write_bytes(b"data")
            (root / "recording.partial.rrd").write_bytes(b"incomplete")
            self.assertEqual(recordings_at(root / "review_index.html"), [recording])

    def test_failed_manifest_does_not_open_stale_recording(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rerun").mkdir()
            (root / "stale.rrd").write_bytes(b"data")
            (root / "rerun/manifest.json").write_text(
                json.dumps({"entries": [{"status": "failed"}]})
            )
            with self.assertRaisesRegex(ValueError, "No completed"):
                recordings_at(root)

    def test_manifest_relative_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording.rrd"
            recording.write_bytes(b"data")
            (root / "manifest.json").write_text(
                json.dumps({"entries": [{"status": "complete", "recording": "recording.rrd"}]})
            )
            self.assertEqual(recordings_at(root), [recording])

    def test_linux_startup_is_detached_and_uses_vulkan(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock(pid=123)
            process.wait.side_effect = subprocess.TimeoutExpired("rerun", 2)
            with (
                patch("vive_correct.viewer.shutil.which", return_value="/bin/rerun"),
                patch("vive_correct.viewer.sys.platform", "linux"),
                patch("vive_correct.viewer.subprocess.Popen", return_value=process) as popen,
            ):
                launch([root / "recording.rrd"])
            self.assertIn("vulkan", popen.call_args.args[0])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_startup_failure_is_reported(self):
        with TemporaryDirectory() as temporary:
            process = Mock()
            process.wait.return_value = 1
            with (
                patch("vive_correct.viewer.shutil.which", return_value="/bin/rerun"),
                patch("vive_correct.viewer.subprocess.Popen", return_value=process),
            ):
                with self.assertRaisesRegex(ValueError, "exited during startup"):
                    launch([Path(temporary) / "recording.rrd"])
