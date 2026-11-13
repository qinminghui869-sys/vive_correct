"""Find additional wrist observations without confusing image side with hand identity."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import cv2
import numpy as np

from vive_correct.batch.candidate_pool import review_order, render_review
from vive_correct.batch.report_io import digest
from vive_correct.config import batch_settings
from vive_correct.estimate.find_stable_frame import Episode, fit_candidate
from vive_correct.estimate.label_detection import detect_labels, has_tracker_body

_SCAN_STRIDE = batch_settings()["scan"]["stride"]


def scan(episode_path, output, missing_sides, stride=_SCAN_STRIDE, frame_ranges=None):
    """Scan both image halves; only explicit review may assign physical wrists.

    Geometry proposals include moving frames. Stability is evaluated on the
    reviewed observations, not on thousands of unreviewed background rectangles.
    """
    if stride < 1 or not set(missing_sides) <= {"left", "right"}:
        raise ValueError("Invalid scan stride or missing wrist sides")
    episode = Episode.load(episode_path)
    output = Path(output)
    (output / "crops").mkdir(parents=True, exist_ok=True)
    width, height = episode.camera.resolution
    rois = {"left": [0, 0, width // 2, height], "right": [width // 2, 0, width, height]}
    candidates = []
    cap = cv2.VideoCapture(str(episode.video))
    try:
        for frame in range(len(episode.timestamps)):
            if not cap.grab():
                raise ValueError(f"Video decode failed at frame {frame}")
            if frame % stride or (
                frame_ranges is not None and not any(a <= frame < b for a, b in frame_ranges)
            ):
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f"Video retrieval failed at frame {frame}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for hint, proposals in detect_labels(image, rois, preserve_variants=True).items():
                # Threshold variants at the same location share one review slot.
                centres = []
                for quad in proposals:
                    centre = np.mean(quad, axis=0)
                    if any(np.linalg.norm(centre - old) < 10 for old in centres):
                        continue
                    centres.append(centre)
                    try:
                        pose = fit_candidate(image, quad, episode.camera)
                    except (ValueError, cv2.error):
                        continue
                    q = np.asarray(pose["image_corners_px"])
                    body = has_tracker_body(gray, q)
                    area = float(cv2.contourArea(q.astype(np.float32)))
                    x, y = q.mean(0).astype(int)
                    xa, ya = max(0, x - 150), max(0, y - 140)
                    crop = image[ya : min(height, y + 160), xa : min(width, x + 150)].copy()
                    key = f"f{frame:06d}_{hint}_{len(candidates):05d}"
                    cv2.polylines(
                        crop, [np.round(q - [xa, ya]).astype(np.int32)], True, (0, 220, 255), 1
                    )
                    if not cv2.imwrite(str(output / "crops" / f"{key}.jpg"), crop):
                        raise OSError("Cannot write candidate image")
                    quality = np.sqrt(area) * (2 if body else 1) / (1 + pose["rms_px"])
                    candidates.append(
                        {
                            "id": key,
                            "frame_index": frame,
                            "side_hint": hint,
                            "quality": float(quality),
                            "area_px2": area,
                            "compact_body": body,
                            "pose": pose,
                            "stability": {
                                "passed": None,
                                "reason": "deferred_until_identity_review",
                            },
                        }
                    )
    finally:
        cap.release()
    report = {
        "episode": str(episode.path),
        "video": str(episode.video),
        "stride": stride,
        "frame_ranges": frame_ranges,
        "missing_physical_sides": sorted(missing_sides),
        "identity": "unreviewed",
        "camera_source": episode.camera.source,
        "candidates": candidates,
        "note": "No candidate is accepted; side_hint is an image-half hint only.",
    }
    path = output / "candidates.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    render_review(output, limit=96)
    # Prioritize the missing image half, while retaining cross-hand candidates.
    priority = [c for c in review_order(candidates) if c["side_hint"] in missing_sides]
    (output / "missing_side_order.json").write_text(
        json.dumps([c["id"] for c in priority], indent=2) + "\n"
    )
    (output / "scan_complete.json").write_text(
        json.dumps(
            {
                "candidate_pool_sha256": digest(path),
                "video_frames": len(episode.timestamps),
                "sampled_frames": (len(episode.timestamps) - 1) // stride + 1,
                "candidates": len(candidates),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def _init_worker():
    cv2.setNumThreads(1)


def _scan_one(entry, folder, missing, stride):
    result = scan(entry["episode"], folder, missing, stride)
    return entry["index"], len(result["candidates"])


def scan_manifest(manifest_path, output, stride=_SCAN_STRIDE, indices=None, jobs=1):
    """Resume unchanged scans; reject stale pools rather than silently mixing runs.

    Pending episodes are independent (each writes only its own folder), so they
    run in ``jobs`` worker processes.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    pending = []
    for entry in manifest["entries"]:
        missing = {"left", "right"} - set(entry["hands"])
        if not missing or (indices and entry["index"] not in indices):
            continue
        folder = output / f"{entry['index']:02d}_{Path(entry['episode']).name}"
        complete = folder / "scan_complete.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            pool_path = folder / "candidates.json"
            if digest(pool_path) != record["candidate_pool_sha256"]:
                raise ValueError(f"Changed completed pool: {folder}")
            pool = json.loads(pool_path.read_text())
            if (
                pool["stride"] != stride
                or Path(pool["episode"]).resolve() != Path(entry["episode"]).resolve()
            ):
                raise ValueError(
                    f"Search configuration changed; use a new output directory: {folder}"
                )
            print(entry["index"], "already complete", flush=True)
            continue
        pending.append((entry, folder, missing))
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs == 1 or len(pending) <= 1:
        for entry, folder, missing in pending:
            _, count = _scan_one(entry, folder, missing, stride)
            print(entry["index"], "candidates", count, flush=True)
        return
    with ProcessPoolExecutor(min(jobs, len(pending)), initializer=_init_worker) as pool:
        futures = [
            pool.submit(_scan_one, entry, folder, missing, stride)
            for entry, folder, missing in pending
        ]
        try:
            for future in as_completed(futures):
                index, count = future.result()
                print(index, "candidates", count, flush=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--stride", type=int, default=_SCAN_STRIDE)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes")
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    scan_manifest(args.manifest, args.out, args.stride, args.indices, args.jobs)


if __name__ == "__main__":
    main()
