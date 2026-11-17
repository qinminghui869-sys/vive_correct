"""Resolve depth metadata consistently across rendering and calibration."""

import json
from pathlib import Path

METADATA_NAMES = (
    "episode_000000_meta.json",
    "fast_foundation_stereo_video_meta.json",
)


def metadata_path(annotation):
    """Prefer the existing episode filename when both names are present."""
    annotation = Path(annotation)
    for name in METADATA_NAMES:
        path = annotation / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Depth metadata missing in {annotation}: expected one of {', '.join(METADATA_NAMES)}"
    )


def load_metadata(annotation):
    """Read the selected file; malformed metadata must not silently fall back."""
    return json.loads(metadata_path(annotation).read_text())
