"""Raw viewing preserves named joint coordinates and nearest sample values."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from render_rerun import table, sample_nearest
from vive_correct.batch.hand_overlay import JOINTS


class RawHandsTests(unittest.TestCase):
    def test_nearest_does_not_interpolate_and_enforces_tolerance(self):
        stream = (np.array([1.0, 1.01]), np.array([[10.0, 20.0], [30.0, 40.0]]))
        value, delta = sample_nearest(stream, 1.003, 0.005)
        np.testing.assert_array_equal(value, stream[1][0])
        self.assertAlmostEqual(delta, -0.003)
        value, delta = sample_nearest(stream, 1.02, 0.005)
        self.assertTrue(np.isnan(value).all())
        self.assertAlmostEqual(delta, -0.01)

    def test_named_columns_ignore_storage_order(self):
        columns = [f"{joint}_{axis}" for joint in JOINTS for axis in "xyz"]
        values = {name: [float(i)] for i, name in reversed(list(enumerate(columns)))}
        values.update(timestamp=[1.0], episode_index=[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation.hand_left_points.parquet"
            pq.write_table(pa.table(values), path)
            _, points = table(path)
        np.testing.assert_array_equal(points[0], np.arange(63))


if __name__ == "__main__":
    unittest.main()
