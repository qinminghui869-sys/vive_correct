"""Validated, serializable settings shared by all classic-CV entry points."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class CVConfig:
    proposal_max_width: int = 960
    thresholds: tuple = (55, 75, 95, 120, 145, 175, 205, 225)
    approx_epsilons: tuple = (0.015, 0.025, 0.035)
    max_saturation: float = 0.26
    max_chroma: float = 22.0
    min_white_fraction: float = 0.75
    min_contrast: float = 28.0
    dark_threshold: int = 115
    min_dark_fraction: float = 0.55
    min_side_dark: float = 0.25
    min_edge_support: float = 0.55
    min_fill: float = 0.85
    min_area: float = 180.0
    max_area_fraction: float = 0.10
    border_scale: float = 0.06
    black_channel_max: int = 150
    surround_dark_threshold: int = 105
    shell_scale: float = 1.8
    context_scale: float = 2.8
    min_shell_fraction: float = 0.35
    min_enclosed_fraction: float = 0.70
    feature_min_corner_angle: float = 25.0
    max_label_black_fraction: float = 0.10
    box_dark_threshold: int = 105
    box_channel_max: int = 125
    box_color_spread: int = 35
    box_scale: float = 1.4
    box_min_black: float = 0.75
    box_min_side_black: float = 0.40
    box_min_enclosure: float = 0.95
    pair_quality_ratio: float = 0.70
    body_max_area_ratio: float = 30.0
    body_max_white_neighbors: int = 4
    body_min_fill: float = 0.28
    body_search_scale: float = 5.0
    feature_min_y_fraction: float = 0.40
    refine_radius_scale: float = 0.10
    max_refine_shift: float = 6.0
    max_line_residual: float = 1.5
    nms_iou: float = 0.70
    association_fraction: float = 0.35
    association_margin: float = 0.10

    def __post_init__(self):
        for name in ("thresholds", "approx_epsilons"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            object.__setattr__(self, name, tuple(values))
        for field in fields(self):
            value = getattr(self, field.name)
            values = value if isinstance(value, tuple) else (value,)
            if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Invalid CV parameter: {field.name}")
        for name in (
            "max_label_black_fraction",
            "box_min_black",
            "box_min_side_black",
            "box_min_enclosure",
            "body_min_fill",
            "feature_min_y_fraction",
            "min_shell_fraction",
            "min_enclosed_fraction",
            "pair_quality_ratio",
            "max_saturation",
            "min_white_fraction",
            "min_dark_fraction",
            "min_side_dark",
            "min_edge_support",
            "min_fill",
            "max_area_fraction",
            "border_scale",
            "refine_radius_scale",
            "nms_iou",
            "association_fraction",
            "association_margin",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        if type(self.proposal_max_width) is not int or self.proposal_max_width < 320:
            raise ValueError("proposal_max_width must be an integer >= 320")
        if any(v >= 255 for v in self.thresholds) or not 0 < self.dark_threshold < 255:
            raise ValueError("Brightness thresholds must be between 0 and 255")
        if any(v >= 0.1 for v in self.approx_epsilons):
            raise ValueError("approx_epsilons must be < 0.1")
        if not 1 < self.shell_scale < self.context_scale <= 5:
            raise ValueError("Require 1 < shell_scale < context_scale <= 5")
        if type(self.body_max_white_neighbors) is not int:
            raise ValueError("body_max_white_neighbors must be an integer")
        if self.surround_dark_threshold >= 255:
            raise ValueError("surround_dark_threshold must be < 255")
        if not 1 < self.box_scale < self.context_scale:
            raise ValueError("box_scale must be between 1 and context_scale")
        if any(
            v > 255 for v in (self.box_dark_threshold, self.box_channel_max, self.box_color_spread)
        ):
            raise ValueError("Box pixel thresholds must be <= 255")
        if self.black_channel_max > 255:
            raise ValueError("black_channel_max must be <= 255")
        if self.feature_min_corner_angle >= 90:
            raise ValueError("feature_min_corner_angle must be < 90")
        if self.association_margin >= self.association_fraction:
            raise ValueError("association_margin must be below association_fraction")

    @classmethod
    def load(cls, path=None):
        default = Path(__file__).resolve().parents[1] / "config/cv.yaml"
        values = yaml.safe_load(default.read_text())
        if path is not None:
            override = yaml.safe_load(Path(path).read_text())
            if not isinstance(override, dict):
                raise ValueError("CV config must be a mapping")
            values.update(override)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown CV parameters: {sorted(unknown)}")
        return cls(**values)

    def report(self):
        values = json.loads(json.dumps(asdict(self)))
        return {
            "parameters": values,
            "sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
        }
