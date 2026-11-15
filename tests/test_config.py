"""YAML defaults stay wired into the estimate and batch entry points."""

import copy
import unittest
from unittest.mock import patch

import yaml

from vive_correct.batch.pose_alignment import _CALIBRATION
from vive_correct.batch.workflow import BatchConfig
from vive_correct.config import _ESTIMATE, _load, batch_settings, estimate_settings, run_settings
from vive_correct.estimate.find_stable_frame import SelectionConfig
from vive_correct.estimate.geometry import LABEL_SIZE_M, MAX_RMS_PX, MIN_EDGE_PX
from vive_correct.estimate.visualization import AXIS_LENGTH_M


class ConfigTests(unittest.TestCase):
    def test_run_profiles_reject_malformed_or_inconsistent_parameters(self):
        document = {name: run_settings(name) for name in ("astra", "legacy")}
        for key, value in (
            ("review_confidence", float("nan")),
            ("model_jobs", True),
            ("first_pass", 25),
            ("transport", "unknown"),
            ("tracker_guided_frames", -1),
            ("typo", 4),
        ):
            changed = copy.deepcopy(document)
            changed["astra"][key] = value
            with (
                self.subTest(key=key),
                patch("vive_correct.config.Path.read_text", return_value=yaml.safe_dump(changed)),
                self.assertRaises(ValueError),
            ):
                run_settings("astra")

    def test_yaml_defaults_reach_the_pipeline(self):
        estimate = estimate_settings()
        batch = batch_settings()
        selection = estimate["selection"]
        geometry = estimate["geometry"]
        config = SelectionConfig()
        self.assertEqual(config.window, selection["window"])
        self.assertEqual(config.stride, selection["stride"])
        self.assertEqual(config.max_frames, selection["max_frames"])
        self.assertEqual(config.max_fb_px, selection["max_fb_px"])
        self.assertEqual(config.min_projected_edge_px, selection["min_projected_edge_px"])
        self.assertEqual(LABEL_SIZE_M, (geometry["label_length_m"], geometry["label_width_m"]))
        self.assertEqual(MAX_RMS_PX, geometry["max_rms_px"])
        self.assertEqual(MIN_EDGE_PX, geometry["min_edge_px"])
        self.assertEqual(AXIS_LENGTH_M, geometry["axis_length_m"])
        self.assertEqual(
            BatchConfig.__dataclass_fields__["stride"].default, batch["scan"]["stride"]
        )
        self.assertEqual(_CALIBRATION["max_sync_gap_s"], batch["calibration"]["max_sync_gap_s"])
        self.assertEqual(_CALIBRATION["min_train_observations"], 6)
        self.assertEqual(estimate["detector"]["name"], "cv")
        self.assertEqual(geometry["max_x_from_left_deg"], 40)
        self.assertEqual(geometry["max_y_from_up_deg"], 40)
        self.assertEqual(geometry["max_z_from_forward_deg"], 50)

    def test_unknown_and_invalid_values_fail(self):
        document = copy.deepcopy(estimate_settings())
        document["extra"] = {}
        with self.assertRaises(ValueError):
            _load("estimate.yaml", _ESTIMATE, document)
        document = copy.deepcopy(estimate_settings())
        document["selection"]["window"] = 2
        with self.assertRaises(ValueError):
            _load("estimate.yaml", _ESTIMATE, document)
        document = copy.deepcopy(estimate_settings())
        document["selection"]["stride"] = True
        with self.assertRaises(ValueError):
            _load("estimate.yaml", _ESTIMATE, document)


if __name__ == "__main__":
    unittest.main()
