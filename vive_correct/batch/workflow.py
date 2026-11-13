"""Configuration-driven batch orchestration; geometry lives in stage modules."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from vive_correct.batch.bilateral_gallery import build
from vive_correct.batch.bilateral_reconstruction import reconstruct
from vive_correct.batch.bilateral_search import scan_manifest
from vive_correct.batch.mount_calibration import calibrate_manifest
from vive_correct.config import batch_settings

_SCAN_STRIDE = batch_settings()["scan"]["stride"]


@dataclass(frozen=True)
class BatchConfig:
    baseline: Path
    output: Path
    initial_mount: Path
    review: Path
    stride: int = _SCAN_STRIDE

    @classmethod
    def load(cls, path):
        """Resolve relative paths against the config, never the caller's cwd."""
        path = Path(path).resolve()
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {"baseline", "output", "initial_mount"}
        allowed = required | {"review", "stride"}
        if not isinstance(doc, dict) or required - doc.keys():
            raise ValueError("Config requires baseline, output and initial_mount")
        if doc.keys() - allowed:
            raise ValueError(f"Unknown config keys: {sorted(doc.keys() - allowed)}")

        def resolve(key):
            value = doc[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Config {key} must be a nonempty path")
            return (path.parent / value).resolve()

        output = resolve("output")
        baseline = resolve("baseline")
        if output == baseline.parent or output in baseline.parents:
            raise ValueError("Output must be separate from the baseline directory")
        stride = doc.get("stride", _SCAN_STRIDE)
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("Config stride must be a positive integer")
        review = resolve("review") if "review" in doc else output / "reviews.json"
        return cls(baseline, output, resolve("initial_mount"), review, stride)

    @property
    def manifest(self):
        return self.output / "results/manifest.json"


def search(config):
    return scan_manifest(config.baseline, config.output / "search", config.stride)


def run(config, stage="run"):
    """Run stops before expensive work if required review or inputs are absent."""
    stages = ("reconstruct", "calibrate", "gallery") if stage == "run" else (stage,)
    if stage not in {"scan", "reconstruct", "calibrate", "gallery", "run"}:
        raise ValueError(f"Unknown stage: {stage}")
    required = []
    if stage in {"scan", "reconstruct", "run"}:
        required.append(config.baseline)
    if "reconstruct" in stages:
        required.append(config.review)
    if "calibrate" in stages:
        required.append(config.initial_mount)
    if stage in {"calibrate", "gallery"}:
        required.append(config.manifest)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    cv2.setNumThreads(1)
    for current in stages:
        print(f"[{current}] {config.output}", flush=True)
        if current == "scan":
            search(config)
        elif current == "reconstruct":
            reconstruct(config.baseline, config.review, config.output / "results")
        elif current == "calibrate":
            calibrate_manifest(config.manifest, config.initial_mount, config.output / "calibration")
        else:
            metadata_path = config.output / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            metadata["batch_config"] = {
                "baseline": str(config.baseline),
                "review": str(config.review),
                "initial_mount": str(config.initial_mount),
                "scan_stride": config.stride,
                "note": "Configured values; actual observation windows are recorded per report.",
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            build(config.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Batch YAML; paths are relative to this file")
    parser.add_argument(
        "--stage",
        choices=("scan", "reconstruct", "calibrate", "gallery", "run"),
        default="run",
        help="Default: reconstruct, calibrate, gallery (requires review)",
    )
    args = parser.parse_args(argv)
    try:
        run(BatchConfig.load(args.config), args.stage)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    return 0
