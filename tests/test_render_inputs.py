import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from ego_data.render_inputs import DEPTH_KEY, depth_input
from vive_correct.estimate.geometry import VIDEO_KEY


class RenderInputsTests(unittest.TestCase):
    def test_indexed_new_format_without_mask_and_legacy_precedence(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "中文 + MANUS"
            episode = root / "data/chunk-000/episode_000004"
            episode.mkdir(parents=True)
            (episode / f"{DEPTH_KEY}.parquet").touch()
            for key in (DEPTH_KEY, VIDEO_KEY):
                path = root / "videos/chunk-000" / key / f"{episode.name}.mp4"
                path.parent.mkdir(parents=True)
                path.touch()
            info = root / "meta/info.json"
            info.parent.mkdir()
            feature = dict(
                shape=[48, 64, 1],
                depth_storage=dict(unit="millimeter", max_depth_m=4, invalid_value=0),
            )
            info.write_text(json.dumps(dict(features={DEPTH_KEY: feature})))
            calibration = root / "calibrations/chunk-000" / episode.name / "rectify_params.yaml"
            calibration.parent.mkdir(parents=True)
            calibration.write_text(
                yaml.safe_dump(
                    dict(
                        cameras={
                            VIDEO_KEY: dict(
                                camera_model="pinhole",
                                distortion_model="none",
                                intrinsics=[100, 100, 32, 24],
                                resolution=[64, 48],
                            )
                        }
                    )
                )
            )
            result = depth_input(episode)
            self.assertEqual(result["mode"], "depth_unmasked_preview")
            self.assertEqual(result["metadata"]["intrinsics"]["left"]["fx"], 100)
            ann = root / "annotation/chunk-000" / DEPTH_KEY / episode.name
            ann.mkdir(parents=True)
            (ann / "valid_mask_ffv1.mkv").touch()
            self.assertEqual(depth_input(episode)["mode"], "depth_masked")
            legacy = ann / "episode_000000_meta.json"
            legacy.write_text("broken")
            with self.assertRaises(json.JSONDecodeError):
                depth_input(episode)
            legacy.unlink()
            feature["depth_storage"]["unit"] = "meter"
            info.write_text(json.dumps(dict(features={DEPTH_KEY: feature})))
            with self.assertRaisesRegex(ValueError, "millimeter"):
                depth_input(episode)
