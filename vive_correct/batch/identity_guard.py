"""Independently locate wrist devices before accepting the selected tag observations."""

from contextlib import nullcontext
import hashlib
import json

import cv2

from vive_correct.batch.qwen_review import locate_wrist_trackers, matches_tracker, failure_kind
from vive_correct.batch.selection_policy import choose_pair

LOCATOR_VERSION = 2


def verify_choices(
    accepted,
    episode,
    config,
    cache_dir,
    read_frame,
    *,
    semaphore=None,
    timeout=180,
    retries=3,
    transport=None,
    max_checks=8,
):
    remaining = {side: list(items) for side, items in accepted.items()}
    approved, records, errors = set(), [], []

    def token(side, item):
        return (side, item["frame_index"], json.dumps(item["pose"]["image_corners_px"]))

    requests = 0
    choices = choose_pair(remaining)
    for _ in range(max_checks):
        pending = [
            (side, item) for side, item in choices.items() if token(side, item) not in approved
        ]
        if not pending:
            return choices, records, errors, requests
        side, item = pending[0]
        pose = item["pose"]
        if pose.get("baseline_retention"):
            # Inherited identity is not a new model or human verification.
            approved.add(token(side, item))
            continue
        if pose.get("orientation_status") == "user_constraint_applied":
            pose["identity_guard"] = {"status": "user_reference", "physical_axes_verified": False}
            approved.add(token(side, item))
            continue
        path = None
        try:
            image = read_frame(episode, item["frame_index"])
            key = {
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "model": config.model,
                "endpoint": config.base_url,
                "version": LOCATOR_VERSION,
            }
            name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            path = cache_dir / f"{name}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get("rejection"):
                    raise RuntimeError(cached["rejection"])
                locations = cached["locations"]
            else:
                with semaphore if semaphore is not None else nullcontext():
                    requests += 1
                    locations = locate_wrist_trackers(
                        image, config, timeout=timeout, retries=retries, transport=transport
                    )
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"signature": key, "locations": locations}, ensure_ascii=False)
                )
                temporary.replace(path)
            passed = matches_tracker(pose, side, locations, episode.camera.resolution)
            record = {
                "frame_index": item["frame_index"],
                "side": side,
                "passed": passed,
                "method": "independent_full_frame_tracker_localization",
                "tracker": locations.get(side),
            }
            records.append(record)
            if passed:
                pose["identity_guard"] = record
                approved.add(token(side, item))
                continue
        except (RuntimeError, ValueError, KeyError, OSError, cv2.error) as error:
            message = str(error).replace(config.api_key, "[REDACTED]")
            record = {
                "frame_index": item["frame_index"],
                "side": side,
                "kind": failure_kind(message),
                "reason": message,
            }
            records.append(record)
            if record["kind"] == "content_rejected" and path is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"signature": key, "rejection": message}))
            else:
                errors.append(record)
        remaining[side] = [candidate for candidate in remaining[side] if candidate is not item]
        choices = choose_pair(remaining)
    # A bounded check never silently accepts candidates that were not checked.
    return {s: i for s, i in choices.items() if token(s, i) in approved}, records, errors, requests
