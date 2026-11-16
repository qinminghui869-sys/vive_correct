import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego_data.corrected_rgb import export_corrected_rgb
from vive_correct.estimate.geometry import VIDEO_KEY


class CorrectedRGBTests(unittest.TestCase):
    def test_depthless_recording_keeps_correction_and_full_frame_coverage(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            episode = root / "episode"
            data = episode / "data/chunk-000/episode_000000"
            data.mkdir(parents=True)
            times = [0.0, 0.03, 0.06]
            pq.write_table(pa.table({"timestamp": times}), data / f"{VIDEO_KEY}.parquet")
            for side in ("left", "right"):
                values = dict(timestamp=times)
                values.update({k: [0.0] * 3 for k in ("x", "y", "z", "roll", "pitch", "yaw")})
                pq.write_table(pa.table(values), data / f"observation.{side}_wrist_pose.parquet")
            video = episode / "videos/chunk-000" / VIDEO_KEY / "episode_000000.mp4"
            video.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in times:
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            alignment = root / "alignment.json"
            alignment.write_text(
                json.dumps(
                    dict(
                        episode=str(episode),
                        pose_euler_sequence="xyz",
                        pose_degrees=True,
                        wrists={
                            s: dict(model="world", offset_m=[0.1, 0.2, 0.3])
                            for s in ("left", "right")
                        },
                    )
                )
            )
            output = root / "test.rrd"
            result = export_corrected_rgb(episode, output, alignment)
            self.assertEqual(result["frames"], 3)
            self.assertTrue(result["vive_correction_displayed"])
            self.assertEqual(result["mode"], "rectified_rgb_corrected_wrists")
            self.assertEqual(result["invalid_wrist_frames"], {"left": [], "right": []})
            self.assertGreater(output.stat().st_size, 100)
