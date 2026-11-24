"""Reconstruct explicitly reviewed candidates with fresh geometry and stability checks."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from vive_correct.estimate.find_stable_frame import SelectionConfig
from vive_correct.estimate.geometry import Camera, corner_noise_sensitivity
from vive_correct.data.layout import calibration_file
from vive_correct.estimate.sticker_pipeline import run


def load_choices(pool_path, review_path):
    pool_path = Path(pool_path)
    pool = json.loads(pool_path.read_text())
    review = json.loads(Path(review_path).read_text())
    digest = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if review.get("candidate_pool_sha256") != digest:
        raise ValueError("Review does not match this candidate pool; review again after rescanning")
    by_id = {candidate["id"]: candidate for candidate in pool["candidates"]}
    frames = defaultdict(dict)
    for decision in review["accepted"]:
        if decision["side"] not in {"left", "right"}:
            raise ValueError("Reviewed physical side must be left or right")
        candidate = by_id[decision["candidate_id"]]
        if not candidate["stability"]["passed"]:
            raise ValueError("An unstable candidate cannot be accepted")
        frame, side = candidate["frame_index"], decision["side"]
        if side in frames[frame]:
            raise ValueError("Two candidates cannot represent the same wrist in one frame")
        frames[frame][side] = candidate
    return pool, frames


def reconstruct(pool_path, review_path, output):
    """Prefer verified same-frame pairs; never construct a cross-time relative pose."""
    pool, frames = load_choices(pool_path, review_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observations, rejected = [], []
    config = SelectionConfig(**pool["selection_config"])
    for frame, labels in sorted(frames.items()):
        path = output / f"reviewed_frame_{frame:06d}.json"
        path.write_text(
            json.dumps(
                {
                    "episode": pool["episode"],
                    "frame_index": frame,
                    "labels": {
                        side: item["pose"]["image_corners_px"] for side, item in labels.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        result = run(
            pool["episode"], output / f"frame_{frame:06d}", observations=path, config=config
        )
        if not result["observations"]:
            rejected.append(frame)
            continue
        for observation in result["observations"]:
            observation["reviewed_candidates"] = {
                side: labels[side]["id"] for side in observation["stickers"]
            }
            observation["review_quality"] = min(
                labels[side]["quality"] for side in observation["stickers"]
            )
            observations.append(observation)
    if not observations:
        raise ValueError(f"No reviewed candidate passed revalidation; rejected frames: {rejected}")
    pairs = [obs for obs in observations if set(obs["stickers"]) == {"left", "right"}]
    if pairs:
        selected = [max(pairs, key=lambda obs: obs["review_quality"])]
        mode = "both_hands_same_frame"
    else:
        selected = []
        for side in ("left", "right"):
            available = [obs for obs in observations if side in obs["stickers"]]
            if available:
                selected.append(max(available, key=lambda obs: obs["review_quality"]))
        selected.sort(key=lambda obs: obs["frame_index"])
        mode = "separate_frames" if len(selected) == 2 else "single_hand"
    result.update(
        mode=mode,
        observations=selected,
        identity="visually_reviewed_and_revalidated",
        candidate_pool=str(Path(pool_path).resolve()),
        review=str(Path(review_path).resolve()),
        examined_windows=pool["examined_windows"],
        rejected=pool.get("rejected", {}),
        rejected_reviewed_frames=rejected,
    )
    camera = Camera.load(calibration_file(pool["episode"]))
    for observation in selected:
        for pose in observation["stickers"].values():
            if pose.get("low_resolution"):
                pose["corner_noise_sensitivity"] = corner_noise_sensitivity(pose, camera)
    (output / "reconstruction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = reconstruct(args.pool, args.review, args.out)
    print(report["mode"])
