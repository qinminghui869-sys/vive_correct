"""Identity review, constrained geometry and adaptive search with explicit uncertainty."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
from threading import BoundedSemaphore

import cv2
import numpy as np

from vive_correct.batch.bilateral_gallery import build
from vive_correct.batch.identity_guard import verify_choices
from vive_correct.batch.device_axes import bind_anchor, origin_only, CONVENTION
from vive_correct.batch.bilateral_search import scan
from vive_correct.batch.candidate_pool import review_order
from vive_correct.batch.mount_calibration import calibrate, track_reference
from vive_correct.batch.qwen_review import (
    PROMPT_VERSION,
    review_candidates,
    failure_kind,
    validate_decisions,
)
from vive_correct.batch.report_io import digest, write_json
from vive_correct.batch.selection_policy import (
    load_constraints,
    temporal_check,
    visual_quality,
    choose_pair,
    good_pair,
)
from vive_correct.estimate.find_stable_frame import (
    Episode,
    SelectionConfig,
    fit_candidate,
    frame_windows,
    measure_stability,
)
from vive_correct.estimate.visualization import draw_poses, save_image


def diverse_candidates(candidates, limit):
    """Time/position diversity; paired frame candidates stay adjacent for review."""
    groups = {side: [] for side in ("left", "right")}
    for candidate in review_order(candidates):
        group = groups[candidate["side_hint"]]
        center = np.asarray(candidate["pose"]["image_corners_px"]).mean(0)
        if any(
            abs(candidate["frame_index"] - old["frame_index"]) < 90
            and np.linalg.norm(center - np.asarray(old["pose"]["image_corners_px"]).mean(0)) < 80
            for old in group
        ):
            continue
        group.append(candidate)
    ordered = []
    for i in range(max(map(len, groups.values()), default=0)):
        for side in groups:
            if i < len(groups[side]):
                ordered.append(groups[side][i])
    selected, seen = [], set()
    by_frame = {}
    for item in review_order(candidates):
        by_frame.setdefault(item["frame_index"], []).append(item)
    for item in ordered:
        for candidate in [item] + [
            c for c in by_frame[item["frame_index"]] if c["side_hint"] != item["side_hint"]
        ][:1]:
            if candidate["id"] not in seen:
                selected.append(candidate)
                seen.add(candidate["id"])
            if len(selected) == limit:
                return selected
    return selected


def refinement_pool(episode, work, index, pool, stride):
    folder = work / "search_refined" / f"{index:02d}_{episode.path.name}"
    # Missing-hand recovery covers the full video; local seeds can miss valid wrists.
    ranges = None
    path = folder / "candidates.json"
    complete = folder / "scan_complete.json"
    if path.exists() and complete.exists():
        cached = json.loads(path.read_text())
        if (
            digest(path) == json.loads(complete.read_text())["candidate_pool_sha256"]
            and cached.get("frame_ranges") == ranges
            and cached["stride"] == stride
            and Path(cached["episode"]).resolve() == episode.path.resolve()
        ):
            return cached
    return scan(episode.path, folder, {"left", "right"}, stride, frame_ranges=ranges)


def read_frame(episode, frame):
    cap = cv2.VideoCapture(str(episode.video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            raise ValueError("Cannot decode selected frame")
        return image
    finally:
        cap.release()


def review_episode(
    entry,
    work,
    config,
    *,
    limit=24,
    batch_size=1,
    timeout=180,
    retries=3,
    confidence=0.8,
    window=11,
    transport=None,
    semaphore=None,
    constraints=(),
    first_pass=8,
    fine_stride=5,
    qwen_review=True,
    fixed_candidates=False,
    guided_frames=0,
):
    if not qwen_review and constraints:
        raise ValueError("CV-only review cannot inherit reviewed identities")
    folder = work / "search" / f"{entry['index']:02d}_{Path(entry['episode']).name}"
    pool_path = folder / "candidates.json"
    if (
        digest(pool_path)
        != json.loads((folder / "scan_complete.json").read_text())["candidate_pool_sha256"]
    ):
        raise ValueError("Candidate pool hash mismatch")
    pool = json.loads(pool_path.read_text())
    if Path(pool["episode"]).resolve() != Path(entry["episode"]).resolve():
        raise ValueError("Candidate pool episode mismatch")
    reviewer = "qwen" if "qwen" in config.model.lower() else "gpt"
    episode = Episode.load(entry["episode"])
    selection = SelectionConfig(window=window)
    report = json.loads(Path(entry["report"]).read_text())
    report.update(
        observations=[],
        local_axes_convention=CONVENTION,
        axis_convention="Device-local axes fixed by per-wrist reference; no image-direction relabelling",
        identity=f"{reviewer}_visual_review_not_human_verified"
        if qwen_review
        else "cv_image_half_unverified",
        model=config.model if qwen_review else None,
        reviewer_type=f"{reviewer}_visual_review" if qwen_review else "cv_only",
        human_verified=False,
        pose_accuracy_validated=False,
        same_time_propagation_validated=False,
    )
    if limit == 0:
        report.update(
            identity="retained_reviewed_baseline", reviewer_type="baseline_replay", model=None
        )
    cache_dir = work / "qwen_reviews" / f"{entry['index']:02d}"
    decisions, errors, warnings = [], [], []
    accepted = {"left": [], "right": []}
    consumed, processed = set(), 0
    signature = dict(
        prompt_version=PROMPT_VERSION,
        model=config.model,
        endpoint=config.base_url,
        window=window,
        confidence=confidence,
        implementation={
            name: digest(Path(__file__).with_name(name))
            for name in ("automatic.py", "qwen_review.py", "selection_policy.py")
        },
    )
    # Preserve historical observations independently of new model votes or tracking.
    for source in constraints:
        frame, side = source["frame_index"], source["physical_side"]
        pose = deepcopy(source["pose"])
        try:
            image = read_frame(episode, frame)
            if pose.get("axes_generated") is False:
                origin = np.asarray(pose["origin_camera_m"], float)
                if origin.shape != (3,) or not np.isfinite(origin).all() or origin[2] <= 0:
                    raise ValueError("Invalid baseline origin")
            else:
                transform = np.asarray(pose["T_camera_local"], float)
                if (
                    transform.shape != (4, 4)
                    or not np.isfinite(transform).all()
                    or not np.allclose(
                        transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
                    )
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)
                ):
                    raise ValueError("Invalid baseline rigid transform")
                projected = episode.camera.project(
                    np.asarray(pose["object_corners_local_m"]), transform[:3, :3], transform[:3, 3]
                )
                residual = float(
                    np.sqrt(np.mean(np.sum((projected - pose["image_corners_px"]) ** 2, axis=1)))
                )
                if not np.isfinite(residual) or residual > max(3.0, 2 * pose["rms_px"]):
                    raise ValueError("Baseline reprojection mismatch")
            pose = bind_anchor(pose, side, frame, source)
            try:
                _, _, gray = next(frame_windows(episode, selection, frame))
                temporal = temporal_check(gray, pose, episode.camera)
            except (ValueError, OSError, cv2.error, StopIteration) as error:
                temporal = {"passed": False, "reason": str(error)}
            pose["temporal_consistency"] = temporal
            pose["visual_quality"] = visual_quality(image, pose)
            pose["baseline_retention"] = {
                "source_report": source["baseline_source"],
                "source_report_sha256": source["baseline_sha256"],
                "scope": "exact_episode_side_frame",
                "reason": "retained_without_new_verification",
                "original_quality_status": pose.get("quality_status"),
            }
            accepted[side].append({"frame_index": frame, "pose": pose})
            decisions.append(
                {"candidate_id": source["id"], "accepted": True, "reason": "baseline_retained"}
            )
        except (ValueError, KeyError, OSError, cv2.error) as error:
            warnings.append(
                {"kind": "baseline_invalid", "candidate_id": source["id"], "reason": str(error)}
            )
    # Only spend model calls on missing hands. Existing axes are never silently replaced.
    queue = (
        []
        if all(accepted.values())
        else diverse_candidates(pool["candidates"], len(pool["candidates"]))
    )
    model_candidates = 0
    refined = False
    requested_batches = cached_batches = 0
    identity_checks, localization_requests = [], 0
    guided_search = None

    def recovery_queue():
        nonlocal guided_search
        fine = refinement_pool(episode, work, entry["index"], pool, fine_stride)
        extra = []
        if qwen_review and guided_frames:
            from vive_correct.batch.guided_search import guided_pool

            guided_search = guided_pool(
                episode,
                cache_dir / "guided",
                config,
                read_frame,
                count=guided_frames,
                window=window,
                timeout=timeout,
                retries=retries,
                transport=transport,
                semaphore=semaphore,
            )
            extra = guided_search["candidates"]
            errors.extend(guided_search["errors"])
        proposals = extra + diverse_candidates(fine["candidates"], len(fine["candidates"]))
        return [
            c
            for c in proposals
            if (
                c["frame_index"],
                tuple(np.round(np.asarray(c["pose"]["image_corners_px"]).mean(0), 1)),
            )
            not in consumed
        ]

    while True:
        while queue and model_candidates < limit:
            group, queue = queue[:batch_size], queue[batch_size:]
            prepared = []
            for candidate in group:
                token = (
                    candidate["frame_index"],
                    tuple(np.round(np.asarray(candidate["pose"]["image_corners_px"]).mean(0), 1)),
                )
                if token in consumed:
                    continue
                consumed.add(token)
                processed += 1
                try:
                    frame = candidate["frame_index"]
                    try:
                        _, image, gray = next(frame_windows(episode, selection, frame))
                    except (ValueError, StopIteration):
                        image, gray = read_frame(episode, frame), []
                    if candidate.get("constraint_source"):
                        pose = deepcopy(candidate["pose"])
                        transform = np.asarray(pose["T_camera_local"])
                        projected = episode.camera.project(
                            np.asarray(pose["object_corners_local_m"]),
                            transform[:3, :3],
                            transform[:3, 3],
                        )
                        if not np.allclose(projected, pose["image_corners_px"], atol=3):
                            raise ValueError("Constraint reprojection mismatch")
                    else:
                        pose = fit_candidate(
                            image, candidate["pose"]["image_corners_px"], episode.camera
                        )
                    pose["stability"] = measure_stability(
                        gray, pose["image_corners_px"], episode.camera.resolution, selection
                    )
                    pose["temporal_consistency"] = temporal_check(gray, pose, episode.camera)
                    pose["visual_quality"] = visual_quality(image, pose)
                    prepared.append(
                        dict(
                            id=candidate["id"],
                            frame_index=frame,
                            image=image,
                            pose=pose,
                            candidate=candidate,
                        )
                    )
                except (ValueError, OSError, cv2.error, StopIteration) as error:
                    decisions.append(
                        {"candidate_id": candidate["id"], "accepted": False, "reason": str(error)}
                    )
            if prepared:
                # Exact user-reviewed reference identity is not re-voted by the model.
                response = [
                    {
                        "candidate_id": i["id"],
                        "physical_side": i["candidate"]["physical_side"],
                        "is_label": True,
                        "clear": True,
                        "confidence": 1.0,
                        "visible_features": "Existing user direction reference; geometry and temporal checks passed",
                    }
                    for i in prepared
                    if i["candidate"].get("constraint_source")
                ]
                model_items = [i for i in prepared if not i["candidate"].get("constraint_source")]
                model_items = model_items[: max(0, limit - model_candidates)]
                model_candidates += len(model_items)
                if model_items and not qwen_review:
                    response.extend(
                        dict(
                            candidate_id=i["id"],
                            physical_side=i["candidate"]["side_hint"],
                            is_label=True,
                            clear=True,
                            confidence=None,
                            visible_features="CV proposal; image-half side assumption, identity unverified",
                        )
                        for i in model_items
                    )
                if model_items and qwen_review:
                    key = dict(
                        signature,
                        candidates=[
                            {"id": i["id"], "frame": i["frame_index"], "pose": i["pose"]}
                            for i in model_items
                        ],
                    )
                    import hashlib

                    cache_name = hashlib.sha256(
                        json.dumps(key, sort_keys=True).encode()
                    ).hexdigest()
                    cache_path = cache_dir / f"{cache_name}.json"
                    try:
                        if cache_path.exists():
                            cached = json.loads(cache_path.read_text())
                            cached_batches += 1
                            if cached.get("rejection"):
                                raise RuntimeError(cached["rejection"])
                            model_response = validate_decisions(
                                {"decisions": cached["decisions"]}, [i["id"] for i in model_items]
                            )
                        else:
                            with semaphore if semaphore is not None else nullcontext():
                                requested_batches += 1
                                model_response = review_candidates(
                                    model_items,
                                    episode.camera,
                                    config,
                                    timeout=timeout,
                                    retries=retries,
                                    transport=transport,
                                )
                            write_json(cache_path, {"signature": key, "decisions": model_response})
                        response.extend(model_response)
                    except (RuntimeError, ValueError, OSError) as error:
                        message = str(error).replace(config.api_key, "[REDACTED]")
                        record = {
                            "candidate_ids": [i["id"] for i in model_items],
                            "kind": failure_kind(message),
                            "reason": message,
                        }
                        if record["kind"] == "content_rejected":
                            write_json(cache_path, {"signature": key, "rejection": message})
                        (warnings if record["kind"] == "content_rejected" else errors).append(
                            record
                        )
                by_id = {item["id"]: item for item in prepared}
                for decision in response:
                    decisions.append(decision)
                    side = decision["physical_side"]
                    if not (
                        decision["is_label"]
                        and decision["clear"]
                        and side in accepted
                        and (not qwen_review or decision["confidence"] >= confidence)
                    ):
                        continue
                    item = by_id[decision["candidate_id"]]
                    source = item["candidate"]
                    if source.get("physical_side") and source["physical_side"] != side:
                        warnings.append(
                            {"kind": "constraint_identity_conflict", "candidate_id": item["id"]}
                        )
                        continue
                    pose = origin_only(item["pose"])
                    if source.get("constraint_source"):
                        pose["axis_review"].update(
                            source_report=source["constraint_source"],
                            source_report_sha256=source["constraint_sha256"],
                            scope="exact_episode_side_frame_only",
                        )
                    else:
                        pose.update(
                            orientation_status="model_uncertain"
                            if pose.get("axes_generated", True)
                            else "axes_not_generated",
                            axis_review={
                                "reviewer_type": f"{reviewer}_identity_only"
                                if qwen_review
                                else "cv_only",
                                "model": config.model if qwen_review else None,
                                "old_from_new": None,
                                "method": "Identity only; device installation reference missing",
                                "human_verified": False,
                            },
                        )
                    pose.update(
                        physical_axes_verified=False,
                        true_vive_origin_calibrated=False,
                        T_label_vive=None,
                        source_candidate_id=item["id"],
                        proposal_source=source.get("proposal_source", "full_frame_cv"),
                        identity_review={
                            "source": "user_reference"
                            if source.get("constraint_source")
                            else reviewer
                            if qwen_review
                            else "cv_image_half_unverified",
                            "model": config.model
                            if qwen_review and not source.get("constraint_source")
                            else None,
                            "visible_features": decision["visible_features"],
                        },
                    )
                    pose["quality_status"] = (
                        "low_resolution_preview"
                        if pose.get("low_resolution")
                        else ("verified" if pose["stability"]["passed"] else "conditional_motion")
                    )
                    accepted[side].append(dict(frame_index=item["frame_index"], pose=pose))
            print(
                f"[review {entry['index']:02d}] candidates={processed} accepted="
                + str({s: len(v) for s, v in accepted.items()}),
                flush=True,
            )
            if (
                not fixed_candidates
                and not refined
                and model_candidates < limit
                and (not queue or model_candidates >= first_pass)
                and not good_pair(accepted)
            ):
                refined = True
                queue = recovery_queue()
            if (
                not fixed_candidates
                and model_candidates >= first_pass
                and good_pair(accepted)
                and not any(c.get("constraint_source") for c in queue)
            ):
                break
        if qwen_review:
            choices, checks, identity_errors, location_count = verify_choices(
                accepted,
                episode,
                config,
                cache_dir / "localization",
                read_frame,
                semaphore=semaphore,
                timeout=timeout,
                retries=retries,
                transport=transport,
            )
        else:
            choices, checks, identity_errors, location_count = choose_pair(accepted), [], [], 0
        errors.extend(identity_errors)
        identity_checks.extend(checks)
        localization_requests += location_count
        # Do not repeatedly select a device mismatch on subsequent recovery passes.
        rejected_identity = {
            (c["side"], c["frame_index"]) for c in checks if c.get("passed") is False
        }
        for side in accepted:
            accepted[side] = [
                i for i in accepted[side] if (side, i["frame_index"]) not in rejected_identity
            ]
        if (
            not fixed_candidates
            and refined
            and len(choices) < 2
            and queue
            and model_candidates < limit
        ):
            continue
        # If independent localization invalidates an otherwise good coarse pair,
        # use the remaining budget for fine search instead of silently stopping.
        if not fixed_candidates and not refined and model_candidates < limit and len(choices) < 2:
            refined = True
            queue = recovery_queue()
            if queue:
                continue
        break
    report["device_tracks"] = {}
    for side, choice in choices.items():
        if not choice["pose"].get("axes_generated", False):
            continue
        anchor = {"frame_index": choice["frame_index"], "stickers": {side: choice["pose"]}}
        try:
            report["device_tracks"][side] = track_reference(episode, anchor, side, radius=15)
        except (ValueError, OSError, cv2.error) as error:
            report["device_tracks"][side] = {"observations": [], "stops": [{"reason": str(error)}]}
            warnings.append({"kind": "device_tracking_failed", "side": side, "reason": str(error)})
    selected = {}
    for side, choice in choices.items():
        frame = choice["frame_index"]
        observation = selected.setdefault(
            frame,
            {
                "frame_index": frame,
                "timestamp_s": float(episode.timestamps[frame]),
                "stability_window_frames": window,
                "stickers": {},
            },
        )
        observation["stickers"][side] = choice["pose"]
    result_dir = work / "batch/results" / f"{entry['index']:02d}_{Path(entry['episode']).name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    for frame, observation in sorted(selected.items()):
        image = read_frame(episode, frame)
        raw_path, overlay_path = (
            result_dir / f"frame_{frame:06d}.png",
            result_dir / f"frame_{frame:06d}_axes.png",
        )
        save_image(raw_path, image)
        save_image(overlay_path, draw_poses(image, observation["stickers"], episode.camera))
        statuses = [p["quality_status"] for p in observation["stickers"].values()]
        observation.update(
            image=str(raw_path),
            overlay=str(overlay_path),
            reviewer_type=f"{reviewer}_identity_only" if qwen_review else "cv_only",
            human_verified=False,
            quality_status="conditional_motion"
            if "conditional_motion" in statuses
            else "low_resolution_preview"
            if "low_resolution_preview" in statuses
            else "verified",
        )
        report["observations"].append(observation)
    hands = sorted(choices)
    stop_reason = (
        "baseline_replay"
        if limit == 0
        else "bilateral_stable"
        if len(choices) == 2 and good_pair({s: [i] for s, i in choices.items()})
        else "candidate_budget_exhausted"
        if model_candidates >= limit
        else "candidate_pool_exhausted"
        if not queue
        else "bounded_identity_review"
    )
    report.update(
        mode="both_hands_same_frame"
        if any(len(o["stickers"]) == 2 for o in report["observations"])
        else "separate_frames"
        if len(hands) == 2
        else "single_hand"
        if hands
        else "no_usable_label",
        model_review={
            "candidate_count": len(pool["candidates"]),
            "review_limit": limit,
            "first_pass": first_pass,
            "review_batch_size": batch_size,
            "fine_stride": fine_stride,
            "confidence_threshold": confidence,
            "stop_reason": stop_reason,
            "missing_hands": sorted({"left", "right"} - set(hands)),
            "selected_for_review": processed,
            "model_candidates": model_candidates if qwen_review else 0,
            "evaluated_candidates": model_candidates,
            "qwen_review_enabled": qwen_review,
            "fixed_candidates": fixed_candidates,
            "retained_baseline_hands": [
                s for s, i in choices.items() if i["pose"].get("baseline_retention")
            ],
            "refined_search": refined,
            "guided_search": {
                "enabled": bool(guided_frames),
                "requests": guided_search["requests"] if guided_search else 0,
                "candidate_count": len(guided_search["candidates"]) if guided_search else 0,
                "frames": guided_search["frames"] if guided_search else [],
            },
            "requested_batches": requested_batches,
            "cached_batches": cached_batches,
            "localization_requests": localization_requests,
            "identity_checks": identity_checks,
            "errors": errors,
            "warnings": warnings,
            "human_verified": False,
        },
    )
    path = result_dir / "observations.json"
    write_json(path, report)
    write_json(
        cache_dir / "review_summary.json",
        {
            "decisions": decisions,
            "errors": errors,
            "warnings": warnings,
            "requested_batches": requested_batches,
            "cached_batches": cached_batches,
            "localization_requests": localization_requests,
            "identity_checks": identity_checks,
            "scope": "bounded identity review, no model axis rotation"
            if qwen_review
            else "CV only; image-half identity unverified",
            "human_verified": False,
        },
    )
    return {
        "index": entry["index"],
        "episode": entry["episode"],
        "report": str(path),
        "report_sha256": digest(path),
        "hands": hands,
        "missing_hands": sorted({"left", "right"} - set(hands)),
        "mode": report["mode"],
        "status": "review_error"
        if errors
        else ("model_reviewed" if qwen_review else "cv_only_unverified")
        if hands
        else "no_accepted_label",
    }


def run_automatic(
    work,
    config,
    initial_mount,
    *,
    jobs=12,
    limit=24,
    batch_size=1,
    timeout=180,
    retries=3,
    confidence=0.8,
    window=11,
    transport=None,
    model_jobs=3,
    axis_baseline=None,
    first_pass=8,
    fine_stride=5,
    qwen_review=True,
    fixed_candidates=False,
    guided_frames=0,
):
    work = Path(work).resolve()
    entries = json.loads((work / "scan_manifest.json").read_text())["entries"]
    results, errors = [], []
    semaphore = BoundedSemaphore(model_jobs)
    constraints = load_constraints(axis_baseline, retain_observations=True)

    def process(entry):
        try:
            result = review_episode(
                entry,
                work,
                config,
                limit=limit,
                batch_size=batch_size,
                timeout=timeout,
                retries=retries,
                confidence=confidence,
                window=window,
                transport=transport,
                semaphore=semaphore,
                constraints=constraints.get(str(Path(entry["episode"]).resolve()), ()),
                first_pass=first_pass,
                fine_stride=fine_stride,
                qwen_review=qwen_review,
                fixed_candidates=fixed_candidates,
                guided_frames=guided_frames,
            )
        except Exception as error:
            # Publish a failure record rather than quietly dropping an episode.
            report = json.loads(Path(entry["report"]).read_text())
            report.update(
                observations=[],
                mode="no_usable_label",
                model_review={"errors": [str(error).replace(config.api_key, "[REDACTED]")]},
            )
            path = (
                work
                / "batch/results"
                / f"{entry['index']:02d}_{Path(entry['episode']).name}"
                / "observations.json"
            )
            write_json(path, report)
            result = dict(
                index=entry["index"],
                episode=entry["episode"],
                report=str(path),
                report_sha256=digest(path),
                hands=[],
                missing_hands=["left", "right"],
                mode="no_usable_label",
                status="review_error",
            )
        target = work / "batch/calibration" / f"{entry['index']:02d}"
        try:
            cal = calibrate(result["report"], initial_mount, target)
        except Exception as error:
            cal = {
                "episode": entry["episode"],
                "accepted": False,
                "reason": "calibration_error: " + str(error),
                "source_report_sha256": result["report_sha256"],
            }
            write_json(target / "calibration.json", cal)
            result["calibration_error"] = str(error)
        return result

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process, entry): entry for entry in entries}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "review_error" or result.get("calibration_error"):
                errors.append(result["index"])
            print(
                f"[Repair {len(results)}/{len(entries)}] {result['index']:02d} {result['status']} hands={result['hands']}",
                flush=True,
            )
            write_json(
                work / "automatic_state.json",
                {"entries": sorted(results, key=lambda e: e["index"]), "errors": errors},
            )
    results.sort(key=lambda e: e["index"])
    batch = work / "batch"
    write_json(batch / "results/manifest.json", {"entries": results})
    write_json(
        batch / "run_metadata.json",
        {
            "model": config.model,
            "jobs": jobs,
            "model_jobs": model_jobs,
            "first_pass": first_pass,
            "review_limit": limit,
            "review_batch_size": batch_size,
            "fine_stride": fine_stride,
            "model_timeout": timeout,
            "model_retries": retries,
            "confidence_threshold": confidence,
            "stability_window": window,
            "model_review_is_not_human_verification": True,
            "qwen_review_enabled": qwen_review,
            "fixed_candidates": fixed_candidates,
            "guided_frames": guided_frames,
            "run_plan": {
                "path": str(work / "run_plan.json"),
                "sha256": digest(work / "run_plan.json"),
            }
            if (work / "run_plan.json").is_file()
            else None,
        },
    )
    build(batch)
    print(f"Delivery: {batch / 'delivery/index.html'}", flush=True)
    return 1 if errors else 0
