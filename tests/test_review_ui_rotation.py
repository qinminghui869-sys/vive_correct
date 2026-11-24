"""Check browser local-axis composition against the export rotation convention."""

import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


class BrowserRotationTests(unittest.TestCase):
    def test_current_axes_compose_in_click_order(self):
        rng = np.random.default_rng(42)
        cases = []
        for angles in (
            [0, 0, 0],
            [30, 40, 50],
            [25, 90, 15],
            [25, -90, 15],
            [10, 89.999999, 170],
            *rng.uniform(-180, 180, (40, 3)).tolist(),
        ):
            for axis in range(3):
                for amount in (-90, -5, 1, 45, 90):
                    cases.append(dict(angles=angles, steps=[[axis, amount], [axis, -amount]]))
            cases.append(dict(angles=angles, steps=[[0, 90], [1, 5], [2, -45], [0, 1]]))
        # End exactly at both gimbal locks, where many Euler triples are equivalent.
        cases.extend(
            dict(angles=[25, 0, 0], steps=[[1, angle], [0, 5], [2, -10]]) for angle in (-90, 90)
        )
        script = (
            Path(__file__).resolve().parents[1] / "vive_correct/review/review_ui/axis_rotation.js"
        )
        result = subprocess.run(
            [
                "node",
                "-e",
                "const fs = require('fs'), vm = require('vm');"
                "vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));"
                "const cases = JSON.parse(fs.readFileSync(0, 'utf8'));"
                "console.log(JSON.stringify(cases.map(({angles, steps}) => "
                "steps.map(([axis, delta]) => (angles = rotateCurrentAxes(angles, axis, delta))))));",
                str(script),
            ],
            input=json.dumps(cases),
            text=True,
            capture_output=True,
            check=True,
        )
        for case, results in zip(cases, json.loads(result.stdout)):
            expected = Rotation.from_euler("XYZ", case["angles"], degrees=True).as_matrix()
            for (axis, delta), angles in zip(case["steps"], results):
                expected = (
                    expected @ Rotation.from_euler("xyz"[axis], delta, degrees=True).as_matrix()
                )
                actual = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
                np.testing.assert_allclose(actual, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
