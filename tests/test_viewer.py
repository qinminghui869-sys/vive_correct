import json
import errno
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

from vive_correct.export.viewer import file_watchers_exhausted, launch, recordings_at
from vive_correct.export.viewer_http import serve_and_launch


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
                patch("vive_correct.export.viewer.shutil.which", return_value="/bin/rerun"),
                patch("vive_correct.export.viewer.sys.platform", "linux"),
                patch("vive_correct.export.viewer.file_watchers_exhausted", return_value=False),
                patch("vive_correct.export.viewer.subprocess.Popen", return_value=process) as popen,
            ):
                launch([root / "recording.rrd"])
            self.assertIn("vulkan", popen.call_args.args[0])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_inotify_probe_closes_descriptor_and_detects_resource_exhaustion(self):
        libc = Mock()
        with (
            patch("vive_correct.export.viewer.sys.platform", "linux"),
            patch("vive_correct.export.viewer.ctypes.CDLL", return_value=libc),
            patch("os.close") as close,
        ):
            libc.inotify_init1.return_value = 42
            self.assertFalse(file_watchers_exhausted())
            close.assert_called_once_with(42)
            libc.inotify_init1.return_value = -1
            for error in (errno.EMFILE, errno.ENFILE, errno.ENOSPC):
                with patch("vive_correct.export.viewer.ctypes.get_errno", return_value=error):
                    self.assertTrue(file_watchers_exhausted())
            close.assert_called_once_with(42)

    def test_exhausted_watchers_launch_detached_http_supervisor(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock(pid=123)
            process.wait.side_effect = subprocess.TimeoutExpired("rerun", 2)
            with (
                patch("vive_correct.export.viewer.shutil.which", return_value="/bin/rerun"),
                patch("vive_correct.export.viewer.file_watchers_exhausted", return_value=True),
                patch("vive_correct.export.viewer.subprocess.Popen", return_value=process) as popen,
            ):
                launch([root / "recording.rrd"])
            command = popen.call_args.args[0]
            self.assertIn("vive_correct.export.viewer_http", command)
            self.assertEqual(json.loads(command[3])[0], "/bin/rerun")
            self.assertEqual(command[-1], str(root / "recording.rrd"))

    def test_http_transport_serves_only_selected_files_and_stops_with_viewer(self):
        with TemporaryDirectory() as temporary:
            recording = Path(temporary) / "中文 + recording.rrd"
            recording.write_bytes(b"test recording")
            (recording.parent / "private.txt").write_text("must not be exposed")
            addresses = []

            def viewer(command, **kwargs):
                url = command[-1]
                addresses.append(url)
                self.assertTrue(url.startswith("http://127.0.0.1:"))
                with urlopen(url, timeout=2) as response:
                    self.assertEqual(response.read(), b"test recording")
                for path in ("/", "/private.txt", "/../private.txt"):
                    origin = url.split("/", 3)[:3]
                    with self.assertRaises(HTTPError) as error:
                        urlopen("/".join(origin) + path, timeout=2)
                    self.assertEqual(error.exception.code, 404)
                return Mock(returncode=0)

            with patch("vive_correct.export.viewer_http.subprocess.run", side_effect=viewer):
                self.assertEqual(serve_and_launch(["rerun"], [recording]), 0)
            with self.assertRaises(OSError):
                urlopen(addresses[0], timeout=2)

    def test_startup_failure_is_reported(self):
        with TemporaryDirectory() as temporary:
            process = Mock()
            process.wait.return_value = 1
            with (
                patch("vive_correct.export.viewer.shutil.which", return_value="/bin/rerun"),
                patch("vive_correct.export.viewer.subprocess.Popen", return_value=process),
            ):
                with self.assertRaisesRegex(ValueError, "exited during startup"):
                    launch([Path(temporary) / "recording.rrd"])
