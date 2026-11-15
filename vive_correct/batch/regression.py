"""Repeat the six-episode retention regression using an existing detection cache."""

import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from vive_correct.batch.automatic import run_automatic
from vive_correct.batch.device_axes import bind_anchor
from vive_correct.batch.report_io import digest, write_json
from vive_correct.estimate.openai_vision import load_config

INDICES = {1, 4, 8, 11, 18, 23}


def verify(work, baseline):
    """Check exact source poses, coverage, report integrity and archive integrity."""
    original = {e["index"]: e for e in json.loads(baseline.read_text())["entries"]}
    rows = []
    for entry in json.loads((work / "batch/results/manifest.json").read_text())["entries"]:
        source = Path(original[entry["index"]]["report"])
        if not source.is_absolute():
            source = baseline.parent / source
        assert digest(source) == original[entry["index"]]["report_sha256"]
        before = json.loads(source.read_text())
        after = json.loads(Path(entry["report"]).read_text())
        poses = {
            (o["frame_index"], s): p
            for o in after["observations"]
            for s, p in o["stickers"].items()
        }
        checks = []
        for observation in before["observations"]:
            for side, old in observation["stickers"].items():
                new = poses.get((observation["frame_index"], side))
                expected = bind_anchor(
                    old,
                    side,
                    observation["frame_index"],
                    {"baseline_source": str(source), "baseline_sha256": digest(source)},
                )
                expected_origin = expected["origin_camera_m"]
                checks.append(
                    {
                        "side": side,
                        "frame_index": observation["frame_index"],
                        "origin_retained": new is not None
                        and np.array_equal(expected_origin, new["origin_camera_m"]),
                        "reference_policy_passed": new is not None
                        and new["axes_generated"] == expected["axes_generated"]
                        and np.array_equal(expected["T_camera_local"], new["T_camera_local"]),
                    }
                )
        assert digest(Path(entry["report"])) == entry["report_sha256"]
        review = after["model_review"]
        rows.append(
            {
                "index": entry["index"],
                "hands": entry["hands"],
                "baseline_checks": checks,
                "requests": review["requested_batches"],
                "cached_batches": review["cached_batches"],
                "localization_requests": review["localization_requests"],
                "errors": review["errors"],
                "warnings": review["warnings"],
            }
        )
    with zipfile.ZipFile(work / "batch/bilateral_results.zip") as archive:
        assert archive.testzip() is None
    result = {
        "scope": "six episodes; reused coarse detections; fresh baseline diagnostics and missing-hand search",
        "entries": rows,
        "all_reference_checks_passed": all(
            c["origin_retained"] and c["reference_policy_passed"]
            for r in rows
            for c in r["baseline_checks"]
        ),
        "bilateral_count": sum(len(r["hands"]) == 2 for r in rows),
        "single_count": sum(len(r["hands"]) == 1 for r in rows),
        "zip_integrity_passed": True,
    }
    write_json(work / "verification_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/l3_batch_run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/l3_axis_review_20261113/gallery_manifest.json"),
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("vive_correct/openai.example.json")
    )
    parser.add_argument(
        "--initial-mount",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "calibrations/left_camera_to_head_vive.json",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = args.out.resolve()
    if not args.verify_only:
        cv2.setNumThreads(1)
        entries = [
            e
            for e in json.loads((args.source / "scan_manifest.json").read_text())["entries"]
            if e["index"] in INDICES
        ]
        if {e["index"] for e in entries} != INDICES:
            raise ValueError("Source must contain all six regression episodes")
        for entry in entries:
            name = f"{entry['index']:02d}_{Path(entry['episode']).name}"
            source = args.source / "search" / name / "candidates.json"
            pool = json.loads(source.read_text())
            pool.update(
                candidates=[c for c in pool["candidates"] if c["frame_index"] % 15 == 0],
                stride=15,
                source_pool_sha256=digest(source),
            )
            target = work / "search" / name
            write_json(target / "candidates.json", pool)
            write_json(
                target / "scan_complete.json",
                {"candidate_pool_sha256": digest(target / "candidates.json")},
            )
        write_json(work / "scan_manifest.json", {"entries": entries})
        status = run_automatic(
            work,
            load_config(config_path=args.model_config),
            args.initial_mount,
            axis_baseline=args.baseline,
            jobs=12,
            model_jobs=3,
            timeout=60,
            retries=2,
        )
    else:
        status = 0
    result = verify(work, args.baseline)
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}))
    return status or (0 if result["all_reference_checks_passed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
