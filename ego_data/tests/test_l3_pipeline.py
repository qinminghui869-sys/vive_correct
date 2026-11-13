"""The public L3 entry point must keep world offsets, A calibration and full export."""

import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch
from align_wrists import render_style_arguments

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_l3_entry_point_defaults(self):
        with patch("sys.argv", ["align_l3_wrists.py"]), patch("align_wrists.App") as app:
            runpy.run_path(str(ROOT / "align_l3_wrists.py"), run_name="__main__")
        args = app.call_args.args[0]
        self.assertTrue(args.world_only)
        self.assertEqual(args.world_axes, "LUF")
        self.assertEqual(args.initial_frame, 300)
        self.assertEqual(args.start, 0)
        self.assertIsNone(args.end)  # resolved to last depth frame by App
        self.assertEqual(args.calibration.name, "left_camera_to_head_vive.json")
        self.assertEqual(args.output.name, "wrist_alignment_A.json")
        self.assertEqual(args.rerun_output.name, "l3_A_wrists_aligned_all.rrd")
        self.assertTrue(args.render_settings.is_file())

    def test_style_cannot_override_data_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "style.json"
            path.write_text(json.dumps({"render_options": {"output": "elsewhere.rrd"}}))
            with self.assertRaises(ValueError):
                render_style_arguments(path)


if __name__ == "__main__":
    unittest.main()
