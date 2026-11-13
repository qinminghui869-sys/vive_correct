"""Path dispatch and per-episode output isolation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import open_file

ROOT = Path(__file__).resolve().parents[1]


class OpenFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.episode = Path(self.temp.name) / "desk" / "episode"
        (self.episode / "data/chunk-000/episode_000000").mkdir(parents=True)
        (self.episode / "videos/chunk-000").mkdir(parents=True)

    def test_builtin_episode_keeps_saved_outputs(self):
        with patch.object(open_file, "DEFAULT_DATASET", self.episode):
            command = open_file.command_for(self.episode)
        self.assertEqual(
            command[1:], [str(ROOT / "ego_data/align_l3_wrists.py"), "--dataset", str(self.episode)]
        )
        self.assertNotIn("--output", command)

    def test_other_episode_uses_its_own_files_and_a_synced_frame(self):
        with patch.object(open_file, "synced_initial_frame", return_value=301) as sync:
            command = open_file.command_for(self.episode)
        self.assertIn(str(ROOT / "ego_data/outputs/wrist_alignment_desk_episode.json"), command)
        self.assertIn(str(ROOT / "ego_data/outputs/desk_episode_wrists_aligned.rrd"), command)
        self.assertEqual(command[command.index("--initial-frame") + 1], "301")
        sync.assert_called_once_with(self.episode)

    def test_recording_and_video_dispatch(self):
        for suffix, executable in [(".rrd", "rerun"), (".mp4", "xdg-open")]:
            path = Path(self.temp.name) / ("example" + suffix)
            path.touch()
            with patch("open_file.shutil.which", return_value="/usr/bin/" + executable):
                command = open_file.command_for(path)
            self.assertEqual(command, ["/usr/bin/" + executable, str(path)])

    def test_file_inside_episode_can_open_the_episode(self):
        video = self.episode / "videos/chunk-000/video.mp4"
        video.touch()
        command = open_file.command_for(video, tool="wrists", initial_frame=10)
        self.assertIn(str(self.episode), command)
        stickers = open_file.command_for(self.episode, tool="stickers")
        self.assertIn("sticker_pipeline.py", stickers[1])
        self.assertIn(str(self.episode), stickers)


if __name__ == "__main__":
    unittest.main()
