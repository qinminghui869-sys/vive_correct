"""Side identification and regeneration must retain the matching Vive pose."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vive_correct.hand_repair import NODES, identify_sides, repair, world_points
from vive_correct.batch.hand_overlay import JOINTS


def fixture():
    count = 80
    raw = np.zeros((count, 50, 3))
    ergonomic = {}
    for side, block, angle in [
        ("right", 0, 0.5 + 0.3 * np.sin(np.arange(count) * 0.2)),
        ("left", 25, 0.5 + 0.3 * np.cos(np.arange(count) * 0.33)),
    ]:
        for finger, nodes in [("Middle", [11, 12, 13]), ("Ring", [16, 17, 18])]:
            raw[:, block + nodes[1], 0] = 0.04
            raw[:, block + nodes[2], 0] = 0.04 + 0.03 * np.cos(angle)
            raw[:, block + nodes[2], 1] = 0.03 * np.sin(angle)
            ergonomic[f"{side}_{finger}PIPStretch"] = angle
    return raw, ergonomic


class HandRepairTests(unittest.TestCase):
    def test_identification_uses_finger_motion(self):
        raw, ergonomic = fixture()
        mapping, _ = identify_sides(raw, ergonomic)
        self.assertEqual(mapping, {"right": 0, "left": 25})
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            identify_sides(raw[:1].repeat(80, axis=0), ergonomic)

    def test_rotation_translation_and_bone_lengths(self):
        raw, _ = fixture()
        pose = np.tile([1.0, 2.0, 3.0, 0.0, 0.0, 90.0], (len(raw), 1))
        points = world_points(raw, pose, 25)
        np.testing.assert_allclose(points[0, 0], [1.085, 2.0, 2.942])
        np.testing.assert_allclose(
            np.linalg.norm(points - points[:, 0:1], axis=2),
            np.linalg.norm(raw[:, NODES + 25], axis=2),
            atol=1e-12,
        )
        pose[0] = np.nan
        self.assertTrue(np.isnan(world_points(raw, pose, 25)[0]).all())

    def test_repair_writes_new_files_and_keeps_source(self):
        raw, ergonomic = fixture()
        times = np.arange(len(raw)) * 0.01
        with tempfile.TemporaryDirectory() as folder:
            episode = Path(folder) / "episode"
            data = episode / "data/chunk-000/episode_000000"
            data.mkdir(parents=True)
            common = {"timestamp": times, "episode_index": np.zeros(len(raw), dtype=int)}
            fields = dict(common)
            fields.update(
                {f"node{i:02d}_{a}": raw[:, i, j] for i in range(50) for j, a in enumerate("xyz")}
            )
            pq.write_table(pa.table(fields), data / "observation.hand_skeleton.parquet")
            pq.write_table(
                pa.table({**common, **ergonomic}), data / "observation.hand_pose.parquet"
            )
            original = {}
            for side, block, x in [("left", 0, 1.0), ("right", 25, -1.0)]:
                pose = np.tile([x, 0.0, 0.0, 0.0, 0.0, 0.0], (len(raw), 1))
                fields = {
                    **common,
                    **{
                        n: pose[:, i] for i, n in enumerate(["x", "y", "z", "roll", "pitch", "yaw"])
                    },
                }
                pq.write_table(pa.table(fields), data / f"observation.{side}_wrist_pose.parquet")
                points = world_points(raw, pose, block).reshape(-1, 63)
                columns = [f"{j}_{a}" for j in JOINTS for a in "xyz"]
                path = data / f"observation.hand_{side}_points.parquet"
                pq.write_table(
                    pa.table({**common, **{n: points[:, i] for i, n in enumerate(columns)}}), path
                )
                original[side] = path.read_bytes()
            out = repair(episode, Path(folder) / "fixed")
            for side, block, x in [("left", 25, 1.0), ("right", 0, -1.0)]:
                t = pq.read_table(out / f"observation.hand_{side}_points.parquet")
                self.assertEqual(t.schema.metadata[b"raw_block_start"], str(block).encode())
                np.testing.assert_allclose(t["wrist_x"].to_numpy(), x)
                self.assertEqual(
                    (data / f"observation.hand_{side}_points.parquet").read_bytes(), original[side]
                )
