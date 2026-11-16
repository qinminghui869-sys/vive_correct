"""Load the commented YAML defaults used by estimate and batch."""

from pathlib import Path
import math

import yaml

_DIR = Path(__file__).resolve().parent

_ESTIMATE = {
    "selection": (
        "window",
        "stride",
        "max_frames",
        "max_fb_px",
        "max_median_motion_px",
        "max_p95_motion_px",
        "max_excursion_px",
        "min_projected_edge_px",
    ),
    "geometry": (
        "label_length_m",
        "label_width_m",
        "max_rms_px",
        "min_edge_px",
        "axis_length_m",
        "max_x_from_left_deg",
        "max_y_from_up_deg",
        "max_z_from_forward_deg",
        "corner_noise_sigma_px",
        "corner_noise_trials",
    ),
    "detector": ("name",),
}
_BATCH = {
    "scan": ("stride",),
    "reconstruction": ("window",),
    "calibration": (
        "max_sync_gap_s",
        "euler_sequence",
        "euler_degrees",
        "min_train_observations",
        "min_holdout_observations",
        "holdout_translation_m",
        "holdout_rotation_deg",
        "max_head_camera_offset_m",
        "max_wrist_label_offset_m",
        "max_jacobian_condition",
    ),
}
_INTS = {
    "window",
    "stride",
    "max_frames",
    "corner_noise_trials",
    "min_train_observations",
    "min_holdout_observations",
}
_FLOATS = {
    "max_fb_px",
    "max_median_motion_px",
    "max_p95_motion_px",
    "max_excursion_px",
    "min_projected_edge_px",
    "label_length_m",
    "label_width_m",
    "max_rms_px",
    "min_edge_px",
    "axis_length_m",
    "max_x_from_left_deg",
    "max_y_from_up_deg",
    "max_z_from_forward_deg",
    "corner_noise_sigma_px",
    "max_sync_gap_s",
    "holdout_translation_m",
    "holdout_rotation_deg",
    "max_head_camera_offset_m",
    "max_wrist_label_offset_m",
    "max_jacobian_condition",
}
_OPTIONAL = {"max_frames"}

_RUN_TYPES = {
    "transport": str,
    "codex_model": str,
    "jobs": int,
    "model_jobs": int,
    "coarse_stride": int,
    "scan_stride": int,
    "first_pass": int,
    "review_limit": int,
    "review_batch_size": int,
    "model_timeout": int,
    "model_retries": int,
    "review_confidence": float,
    "tracker_guided_frames": int,
}


def run_settings(profile="astra"):
    """Load a complete preset; reject misspelled or invalid tuning parameters."""
    document = yaml.safe_load((_DIR / "run.yaml").read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"astra", "legacy", "cv"}:
        raise ValueError("run.yaml: expected astra, legacy and cv profiles")
    if profile not in document:
        raise ValueError(f"Unknown run profile: {profile}")
    for name, values in document.items():
        if not isinstance(values, dict) or set(values) != set(_RUN_TYPES):
            raise ValueError(f"run.yaml: {name} expected keys {sorted(_RUN_TYPES)}")
        for key, kind in _RUN_TYPES.items():
            value = values[key]
            if kind is float:
                valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
            elif kind is int:
                valid = type(value) is int and value >= (0 if key == "tracker_guided_frames" else 1)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise ValueError(f"run.yaml: invalid {name}.{key}")
        if values["transport"] not in {"api", "codex"}:
            raise ValueError(f"run.yaml: invalid {name}.transport")
        if values["first_pass"] > values["review_limit"]:
            raise ValueError(f"run.yaml: {name}.first_pass exceeds review_limit")
    return dict(document[profile])


def _load(name, schema, doc=None):
    if doc is None:
        doc = yaml.safe_load((_DIR / name).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != set(schema):
        raise ValueError(f"{name}: expected sections {sorted(schema)}")
    parsed = {}
    for section, keys in schema.items():
        block = doc[section]
        if not isinstance(block, dict) or set(block) != set(keys):
            raise ValueError(f"{name}: {section} expected keys {list(keys)}")
        parsed[section] = {key: _value(name, section, key, block[key]) for key in keys}
    if name == "estimate.yaml":
        _validate_estimate(parsed)
    else:
        _validate_batch(parsed)
    return parsed


def _positive(filename, section, key, value, minimum=0):
    if value <= minimum:
        raise ValueError(f"{filename}: {section}.{key} must be greater than {minimum}")


def _odd_window(filename, section, value):
    if value < 3 or value % 2 != 1:
        raise ValueError(f"{filename}: {section}.window must be an odd integer >= 3")


def _validate_estimate(doc):
    selection, geometry = doc["selection"], doc["geometry"]
    _odd_window("estimate.yaml", "selection", selection["window"])
    _positive("estimate.yaml", "selection", "stride", selection["stride"])
    if selection["max_frames"] is not None and selection["max_frames"] < selection["window"]:
        raise ValueError("estimate.yaml: selection.max_frames must include a complete window")
    for key in ("max_fb_px", "max_median_motion_px", "max_p95_motion_px", "max_excursion_px"):
        _positive("estimate.yaml", "selection", key, selection[key])
    if selection["min_projected_edge_px"] < 8:
        raise ValueError("estimate.yaml: selection.min_projected_edge_px must be at least 8")
    for key in (
        "label_length_m",
        "label_width_m",
        "max_rms_px",
        "axis_length_m",
        "corner_noise_sigma_px",
    ):
        _positive("estimate.yaml", "geometry", key, geometry[key])
    for key in ("max_x_from_left_deg", "max_y_from_up_deg", "max_z_from_forward_deg"):
        _positive("estimate.yaml", "geometry", key, geometry[key])
        if geometry[key] > 90:
            raise ValueError(f"estimate.yaml: geometry.{key} must be at most 90 degrees")
    if geometry["min_edge_px"] < 8:
        raise ValueError("estimate.yaml: geometry.min_edge_px must be at least 8")
    _positive("estimate.yaml", "geometry", "corner_noise_trials", geometry["corner_noise_trials"])
    if doc["detector"]["name"] not in {"cv", "ai"}:
        raise ValueError("estimate.yaml: detector.name must be cv or ai")


def _validate_batch(doc):
    _positive("batch.yaml", "scan", "stride", doc["scan"]["stride"])
    _odd_window("batch.yaml", "reconstruction", doc["reconstruction"]["window"])
    calibration = doc["calibration"]
    for key in (
        "max_sync_gap_s",
        "holdout_translation_m",
        "holdout_rotation_deg",
        "max_head_camera_offset_m",
        "max_wrist_label_offset_m",
        "max_jacobian_condition",
    ):
        _positive("batch.yaml", "calibration", key, calibration[key])
    for key in ("min_train_observations", "min_holdout_observations"):
        _positive("batch.yaml", "calibration", key, calibration[key])
    if len(calibration["euler_sequence"]) != 3 or set(calibration["euler_sequence"].lower()) - set(
        "xyz"
    ):
        raise ValueError(
            "batch.yaml: calibration.euler_sequence must be a three-axis sequence such as xyz"
        )


def _value(filename, section, key, value):
    if key in _OPTIONAL and value is None:
        return None
    if key in _INTS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{filename}: {section}.{key} must be an integer")
        return value
    if key in _FLOATS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{filename}: {section}.{key} must be a number")
        return float(value)
    if key == "euler_degrees":
        if not isinstance(value, bool):
            raise ValueError(f"{filename}: {section}.{key} must be true or false")
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{filename}: {section}.{key} must be text")
    return value


def estimate_settings():
    return _load("estimate.yaml", _ESTIMATE)


def batch_settings():
    return _load("batch.yaml", _BATCH)
