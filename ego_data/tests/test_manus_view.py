import json
import tempfile
import unittest
from pathlib import Path
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from manus_view import export


class ManusViewTests(unittest.TestCase):
    def test_no_depth_or_head_and_invalid_timestamp_still_opens_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ep = root / "episode"
            data = ep / "data/chunk-000/episode_000000"
            data.mkdir(parents=True)
            key = "observation.images.headband_left_rgb_rectified"
            video = ep / "videos/chunk-000" / key / "episode_000000.mp4"
            video.parent.mkdir(parents=True)
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width = 32
                stream.height = 32
                stream.pix_fmt = "yuv420p"
                for i in range(3):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((32, 32, 3), i * 50, dtype=np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            pq.write_table(pa.table({"timestamp": [0.0, np.nan, 0.02]}), data / f"{key}.parquet")
            fields = {"timestamp": [0.02, 0.0, 0.0]}
            fields.update(
                {f"node{i:02d}_{axis}": [i * 0.001] * 3 for i in range(50) for axis in "xyz"}
            )
            pq.write_table(pa.table(fields), data / "observation.hand_skeleton.parquet")
            output = root / "preview.rrd"
            report = export(ep, output, selected=[0, 1, 2])
            self.assertEqual(report["frames"], 3)
            self.assertFalse(report["registered"])
            self.assertEqual([s["valid"] for s in report["hand_sync"]], [True, False, True])
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(
                json.loads(output.with_suffix(".json").read_text())["mode"], "rgb_manus_only"
            )
