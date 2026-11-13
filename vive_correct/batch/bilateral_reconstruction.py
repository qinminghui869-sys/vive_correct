"""Merge reviewed opposite-wrist observations while retaining per-wrist quality."""

import argparse
import json
from pathlib import Path

import cv2

from vive_correct.batch.report_io import digest
from vive_correct.config import batch_settings
from vive_correct.estimate.find_stable_frame import (
    Episode,
    SelectionConfig,
    fit_candidate,
    frame_windows,
    measure_stability,
)
from vive_correct.estimate.geometry import corner_noise_sensitivity
from vive_correct.estimate.visualization import draw_poses, save_image

_RECONSTRUCTION_WINDOW = batch_settings()["reconstruction"]["window"]


def reviewed_observation(episode_path, decision, folder):
    """Recheck geometry and stability after verifying the reviewed pool identity."""
    pool_path = Path(decision["pool"])
    if digest(pool_path) != decision["candidate_pool_sha256"]:
        raise ValueError("Reviewed candidate pool changed")
    pool = json.loads(pool_path.read_text())
    if Path(pool["episode"]).resolve() != Path(episode_path).resolve():
        raise ValueError("Review refers to a different episode")
    candidate = next(
        (item for item in pool["candidates"] if item["id"] == decision["candidate_id"]),
        None,
    )
    if candidate is None:
        raise ValueError(f"Reviewed candidate is missing: {decision['candidate_id']}")

    episode = Episode.load(episode_path)
    config = SelectionConfig(window=_RECONSTRUCTION_WINDOW)
    frame = candidate["frame_index"]
    side = decision["side"]
    _, image, gray = next(frame_windows(episode, config, frame))
    pose = fit_candidate(image, candidate["pose"]["image_corners_px"], episode.camera, refine=False)
    pose["stability"] = measure_stability(
        gray, pose["image_corners_px"], episode.camera.resolution, config
    )
    if not pose["stability"]["passed"] and not decision.get("allow_motion"):
        return None
    if pose.get("axes_generated", True):
        pose["corner_noise_sensitivity"] = corner_noise_sensitivity(pose, episode.camera)
    pose["source_candidate_id"] = candidate["id"]
    image_path = folder / f"{side}_{frame:06d}.png"
    overlay_path = folder / f"{side}_{frame:06d}_axes.png"
    save_image(image_path, image)
    save_image(overlay_path, draw_poses(image, {side: pose}, episode.camera))
    return {
        "frame_index": frame,
        "timestamp_s": float(episode.timestamps[frame]),
        "image": str(image_path.resolve()),
        "overlay": str(overlay_path.resolve()),
        "stickers": {side: pose},
        "stability_window_frames": _RECONSTRUCTION_WINDOW,
        "quality_status": "verified" if pose["stability"]["passed"] else "conditional_motion",
        "review": decision,
    }


def reconstruct(manifest_path, review_path, output):
    manifest_path, review_path, output = map(Path, (manifest_path, review_path, output))
    baseline = json.loads(manifest_path.read_text())
    choices = json.loads(review_path.read_text())["entries"]
    review_hash = digest(review_path)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in baseline["entries"]:
        source = manifest_path.parent / item["report"]
        source_hash = digest(source)
        if source_hash != item["report_sha256"]:
            raise ValueError("Baseline report hash mismatch")
        report = json.loads(source.read_text())
        original_count = len(report["observations"])
        folder = output / f"{item['index']:02d}_{Path(item['episode']).name}"
        folder.mkdir(exist_ok=True)
        known = {side for observation in report["observations"] for side in observation["stickers"]}
        for observation in report["observations"]:
            observation.setdefault("quality_status", item["status"])

        rejected = []
        for decision in (choice for choice in choices if choice["index"] == item["index"]):
            if not decision.get("identity_verified") or not decision.get("clear_label"):
                rejected.append(
                    {
                        "candidate_id": decision["candidate_id"],
                        "reason": "identity_or_clarity_not_verified",
                    }
                )
                continue
            side = decision["side"]
            if side not in {"left", "right"} or side in known:
                raise ValueError("Review must identify a missing physical wrist")
            observation = reviewed_observation(item["episode"], decision, folder)
            if observation is None:
                rejected.append(
                    {"candidate_id": decision["candidate_id"], "reason": "motion_not_authorized"}
                )
                continue
            report["observations"].append(observation)
            known.add(side)

        report["observations"].sort(key=lambda observation: observation["frame_index"])
        if any(
            set(observation["stickers"]) == {"left", "right"}
            for observation in report["observations"]
        ):
            mode = "both_hands_same_frame"
        else:
            mode = "separate_frames" if known == {"left", "right"} else "single_hand"
        report.update(
            mode=mode,
            baseline_report=str(source.resolve()),
            baseline_report_sha256=source_hash,
            review_source=str(review_path.resolve()),
            review_sha256=review_hash,
            rejected_additions=rejected,
            same_time_propagation_validated=False,
            note="Separate-frame observations are not a simultaneous relative pose. Per-observation quality is authoritative.",
        )
        path = folder / "observations.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        entry = {
            "index": item["index"],
            "episode": item["episode"],
            "report": str(path.resolve()),
            "report_sha256": digest(path),
            "hands": sorted(known),
            "missing_hands": sorted({"left", "right"} - known),
            "mode": mode,
            "baseline_status": item["status"],
            "new_observations": len(report["observations"]) - original_count,
        }
        entries.append(entry)
        print(item["index"], mode, entry["new_observations"], flush=True)
    result = {
        "entries": entries,
        "image_source": "headband_left_rgb_rectified",
        "bilateral_observation_count": sum(len(entry["hands"]) == 2 for entry in entries),
        "same_time_calibrated_count": 0,
        "baseline_manifest": str(manifest_path.resolve()),
        "note": "Bilateral observation coverage includes quality-flagged previews; it is not a metric accuracy or calibrated-trajectory success count.",
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    reconstruct(args.manifest, args.review, args.out)


if __name__ == "__main__":
    main()
