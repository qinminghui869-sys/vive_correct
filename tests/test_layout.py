"""Folder resolution for single-episode recordings and multi-episode datasets."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vive_correct.data.dataset import TIMESTAMP_SUFFIX, VIDEO_SUFFIX, discover, input_signature
from vive_correct.estimate.geometry import VIDEO_KEY
from vive_correct.estimate.openai_vision import load_config
from vive_correct.data.layout import (
    calibration_file,
    data_directory,
    dataset_root,
    timestamp_file,
    video_file,
)
from vive_correct.run_dataset import DEFAULT_DATASET, DEFAULT_OUTPUT, REPO


def _touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class LayoutTests(unittest.TestCase):
    def test_per_episode_calibration_overrides_shared_calibration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = _touch(root / "calibrations/rectify_params.yaml", "shared")
            specific = _touch(
                root / "calibrations/chunk-000/episode_000001/rectify_params.yaml", "specific"
            )
            episode = root / "data/chunk-000/episode_000001"
            self.assertEqual(calibration_file(episode), specific)
            self.assertIn(str(specific.relative_to(root)), input_signature(episode))
            self.assertEqual(calibration_file(root), shared)
            single = _touch(root / "calibrations/chunk-000/episode_000000/rectify_params.yaml")
            self.assertEqual(calibration_file(root), single)

    def test_single_episode_folder_stays_rooted_at_itself(self):
        with TemporaryDirectory() as temporary:
            episode = Path(temporary) / "desk" / "sample"
            _touch(episode / VIDEO_SUFFIX)
            _touch(episode / TIMESTAMP_SUFFIX)
            _touch(episode / "calibrations/rectify_params.yaml")
            self.assertEqual(discover(episode.parent), [episode])
            self.assertEqual(video_file(episode), episode / VIDEO_SUFFIX)
            self.assertEqual(timestamp_file(episode), episode / TIMESTAMP_SUFFIX)
            self.assertEqual(dataset_root(episode), episode)
            signature = input_signature(episode)
            self.assertEqual(
                set(signature),
                {str(VIDEO_SUFFIX), str(TIMESTAMP_SUFFIX), "calibrations/rectify_params.yaml"},
            )

    def test_multi_episode_dataset_is_one_entry_per_episode(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "task"
            for name in ("episode_000000", "episode_000001"):
                _touch(dataset / "videos/chunk-000" / VIDEO_KEY / f"{name}.mp4", name)
                _touch(dataset / "data/chunk-000" / name / f"{VIDEO_KEY}.parquet", name)
            _touch(dataset / "calibrations/rectify_params.yaml", "camera")
            found = discover(dataset)
            self.assertEqual([path.name for path in found], ["episode_000000", "episode_000001"])
            episode = found[1]
            self.assertEqual(dataset_root(episode), dataset)
            self.assertEqual(data_directory(episode), episode)
            self.assertEqual(video_file(episode).name, "episode_000001.mp4")
            self.assertEqual(
                calibration_file(episode), dataset / "calibrations/rectify_params.yaml"
            )
            signature = input_signature(episode)
            self.assertIn(f"videos/chunk-000/{VIDEO_KEY}/episode_000001.mp4", signature)
            self.assertNotIn(dataset, found)

    def test_collection_keeps_each_single_episode_dataset_whole(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "task-a"
            second = root / "task-b"
            _touch(first / VIDEO_SUFFIX)
            _touch(second / "videos/chunk-000" / VIDEO_KEY / "episode_000000.mp4")
            _touch(second / "videos/chunk-000" / VIDEO_KEY / "episode_000002.mp4")
            _touch(second / "data/chunk-000/episode_000002" / f"{VIDEO_KEY}.parquet")
            found = discover(root)
            self.assertIn(first, found)
            self.assertIn(second / "data/chunk-000/episode_000000", found)
            self.assertIn(second / "data/chunk-000/episode_000002", found)
            self.assertNotIn(second, found)

    def test_local_model_file_overrides_environment_and_cli_overrides_file(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "openai.local.json"
            path.write_text(
                json.dumps(
                    {
                        "api_key": "local-key",
                        "base_url": "https://local.example/v1",
                        "model": "qwen3.8-max",
                    }
                )
            )
            config = load_config(
                environ={
                    "OPENAI_API_KEY": "env-key",
                    "OPENAI_BASE_URL": "https://env.example/v1",
                    "OPENAI_MODEL": "env-model",
                },
                config_path=path,
                dotenv_paths=[],
            )
            self.assertEqual(config.api_key, "local-key")
            self.assertEqual(config.base_url, "https://local.example/v1")
            self.assertEqual(config.model, "qwen3.8-max")
            self.assertEqual(config.key_source, "openai.local.json")
            overridden = load_config(
                api_key="cli-key",
                model="cli-model",
                environ={"OPENAI_API_KEY": "env-key"},
                config_path=path,
                dotenv_paths=[],
            )
            self.assertEqual(overridden.api_key, "cli-key")
            self.assertEqual(overridden.model, "cli-model")
            self.assertEqual(overridden.key_source, "cli")

    def test_default_paths_point_at_ks3_and_repository_outputs(self):
        self.assertEqual(DEFAULT_DATASET, Path("/mnt/nas/synnas/脑电/KS3"))
        self.assertEqual(DEFAULT_OUTPUT, REPO / "outputs" / "ks3")


if __name__ == "__main__":
    unittest.main()
