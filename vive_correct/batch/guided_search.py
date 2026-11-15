"""Recover labels inside independently localized Tracker bodies, with pixel geometry.

Model boxes only define search regions. They never become label corners or origins.
Every proposal still requires the normal label review and independent identity guard.
"""

from contextlib import nullcontext
import hashlib
import json

import cv2
import numpy as np

from vive_correct.batch.qwen_review import locate_wrist_trackers, matches_tracker
from vive_correct.estimate.find_stable_frame import fit_candidate
from vive_correct.estimate.label_detection import detect_labels

VERSION = 2


def search_frames(frame_count, count, half_window=5):
    if count < 1 or frame_count <= 2 * half_window:
        return []
    return sorted(
        set(np.linspace(half_window, frame_count - half_window - 1, count).astype(int).tolist())
    )


def proposals_in_trackers(image, locations, camera, frame):
    height, width = image.shape[:2]
    proposals = []
    for side in ("left", "right"):
        if locations.get(side) is None:
            continue
        box = np.asarray(locations[side]["bbox"]) * [width, height, width, height] / 1000
        low = np.maximum(0, np.floor(box[:2] - 12)).astype(int)
        high = np.minimum([width, height], np.ceil(box[2:] + 12)).astype(int)
        if np.any(high - low < 5):
            continue
        quads = detect_labels(
            image, {side: [*low.tolist(), *high.tolist()]}, preserve_variants=True
        )[side]
        centers = []
        for quad in quads:
            try:
                pose = fit_candidate(image, quad, camera)
            except (ValueError, cv2.error):
                continue
            if not matches_tracker(pose, side, locations, camera.resolution):
                continue
            center = np.asarray(pose["image_corners_px"]).mean(0)
            if any(np.linalg.norm(center - previous) < 8 for previous in centers):
                continue
            centers.append(center)
            area = cv2.contourArea(np.asarray(pose["image_corners_px"], np.float32))
            proposals.append(
                dict(
                    id=f"guided_f{frame:06d}_{side}_{len(proposals):03d}",
                    frame_index=frame,
                    side_hint=side,
                    quality=float(np.sqrt(area) / (1 + pose["rms_px"])),
                    pose=pose,
                    proposal_source="tracker_guided_pixel_detection",
                    tracker_box_normalized=locations[side]["bbox"],
                )
            )
    return proposals


def guided_pool(
    episode,
    cache_dir,
    config,
    read_frame,
    *,
    count=6,
    window=11,
    timeout=180,
    retries=3,
    transport=None,
    semaphore=None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates, records, errors = [], [], []
    requests = 0
    for frame in search_frames(len(episode.timestamps), count, window // 2):
        try:
            image = read_frame(episode, frame)
            signature = dict(
                version=VERSION,
                model=config.model,
                endpoint=config.base_url,
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            )
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            path = cache_dir / f"{key}.json"
            if path.exists():
                locations = json.loads(path.read_text())["locations"]
            else:
                with semaphore if semaphore is not None else nullcontext():
                    requests += 1
                    locations = locate_wrist_trackers(
                        image, config, timeout=timeout, retries=retries, transport=transport
                    )
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(dict(signature=signature, locations=locations), ensure_ascii=False)
                )
                temporary.replace(path)
            found = proposals_in_trackers(image, locations, episode.camera, frame)
            candidates.extend(found)
            records.append(dict(frame_index=frame, locations=locations, candidates=len(found)))
        except (RuntimeError, ValueError, OSError, KeyError, cv2.error) as error:
            errors.append(
                dict(frame_index=frame, reason=str(error).replace(config.api_key, "[REDACTED]"))
            )
    report = dict(
        candidates=candidates,
        frames=records,
        errors=errors,
        requests=requests,
        model=config.model,
        version=VERSION,
    )
    (cache_dir / "guided_search.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
