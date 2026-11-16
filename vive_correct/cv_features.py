"""Whole-video feature-image selection, with same-frame pairs preferred.

These are appearance candidates, not accepted poses or verified physical hands.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import html
import json
import hashlib
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from vive_correct.batch.report_io import write_json
from vive_correct.dataset import VIDEO_SUFFIX, input_signature, select_episodes, source_hashes
from vive_correct.estimate.label_detection import (
    black_context_metrics,
    detect_label_candidates,
    label_metrics,
)


def review_config(config):
    return replace(
        config,
        max_saturation=max(config.max_saturation, 0.30),
        max_chroma=max(config.max_chroma, 25.0),
        min_white_fraction=min(config.min_white_fraction, 0.68),
        min_contrast=min(config.min_contrast, 22.0),
        min_dark_fraction=min(config.min_dark_fraction, 0.45),
        min_side_dark=min(config.min_side_dark, 0.18),
        min_edge_support=min(config.min_edge_support, 0.45),
    )


def feature_candidate_allowed(candidate, config):
    q = np.asarray(candidate.get("quad", []), float)
    if q.shape != (4, 2):
        return False
    a, b = np.roll(q, 1, axis=0) - q, np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(lengths <= 0):
        return False
    angles = np.degrees(np.arccos(np.clip((a * b).sum(1) / lengths, -1, 1)))
    return bool(
        angles.min() >= config.feature_min_corner_angle
        and candidate.get("accepted")
        and candidate.get("box_supported")
        and candidate.get("compact_body")
        and candidate.get("nearby_white_components", 0) < config.body_max_white_neighbors
        and candidate.get("label_black_fraction", 1) <= config.max_label_black_fraction
    )


def choose_features(best, pairs, ratio):
    eligible = [
        pair
        for pair in pairs
        if all(
            pair[side]["selection_score"] >= ratio * best[side]["selection_score"]
            for side in ("left", "right")
        )
    ]
    if eligible:
        selected = max(eligible, key=lambda pair: min(c["selection_score"] for c in pair.values()))
        return selected, "same_frame"
    return best, "separate_frames" if len(best) == 2 else "partial"


def _serialize(candidate):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in candidate.items()}


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def feature_image(image, candidate, side, config):
    evidence = black_context_metrics(image, candidate["quad"], config, masks=True)
    patch = evidence["patch"].copy()
    quad = np.round(evidence["quad"]).astype(np.int32)
    cv2.polylines(patch, [quad], True, (0, 255, 255), 2)
    mask = np.full_like(patch, 40)
    mask[evidence["context_mask"]] = (80, 80, 80)
    mask[evidence["shell_mask"]] = (120, 120, 120)
    mask[evidence["context_black_mask"] & evidence["context_mask"]] = (180, 120, 40)
    mask[evidence["black_mask"] & evidence["shell_mask"]] = (40, 200, 40)
    cv2.polylines(mask, [quad], True, (0, 255, 255), 2)
    panels = []
    for panel in (patch, mask):
        scale = min(460 / panel.shape[1], 300 / panel.shape[0])
        resized = cv2.resize(panel, None, fx=scale, fy=scale)
        canvas = np.full((310, 470, 3), 25, np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        panels.append(canvas)
    output = np.full((420, 940, 3), 25, np.uint8)
    output[110:] = np.hstack(panels)
    lines = [
        f"{side} | frame {candidate['frame_index']} | {candidate['status']}",
        f"shell black={evidence['shell_black_fraction']:.2f}  enclosure={evidence['enclosed_fraction']:.2f}",
        f"box={candidate.get('box_black_fraction', 0):.2f}  weakest side={min(candidate.get('box_side_black', [0])):.2f}  surround={evidence['surround_black_fraction']:.2f}",
        "Green: black shell pixels. Blue: black surroundings. Side identity UNVERIFIED.",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (10, 23 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return output


def _missing_feature(side, reason):
    image = np.full((420, 940, 3), 30, np.uint8)
    cv2.putText(
        image,
        f"{side}: NO CONFIRMED FEATURE",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 180, 255),
        2,
    )
    cv2.putText(image, reason, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    return image


RAW_LEFT_SUFFIX = Path("videos/chunk-000/observation.images.headband_bleft_rgb/episode_000000.mp4")


def feature_video(episode):
    return episode / VIDEO_SUFFIX, "rectified"


def feature_video_signature(episode):
    path, space = feature_video(episode)
    stat = path.stat() if path.is_file() else None
    return dict(
        path=str(path),
        space=space,
        size=stat.st_size if stat else None,
        mtime_ns=stat.st_mtime_ns if stat else None,
    )


def feature_scan_phases(limit, fps, stride, config):
    """Half-open frame ranges; sampling is anchored at each range start."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid_video_fps")
    if type(stride) is not int or stride < 1:
        raise ValueError("Feature scan stride must be positive")
    split = min(limit, int(np.ceil(config.feature_initial_seconds * fps)))
    return [
        dict(
            name="initial",
            ranges=[
                dict(start=0, stop=split, stride=config.feature_initial_stride),
                dict(start=split, stop=limit, stride=stride),
            ],
        ),
        dict(
            name="late_recovery",
            ranges=[
                dict(start=split, stop=limit, stride=config.feature_recovery_stride),
            ],
        ),
        dict(name="full_recovery", ranges=[dict(start=0, stop=limit, stride=1)]),
    ]


def scan_features(episode, output, config, stride=15, max_frames=None):
    cv2.setNumThreads(1)
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    video, video_space = feature_video(episode)
    report = dict(
        video_space=video_space,
        episode=str(episode),
        video=str(video),
        features={},
        missing_hands=[],
        mode="partial",
        human_verified=False,
        identity_status="image_half_unverified",
        side_assignment="image-side hints only; crossing may invert physical hands",
        detector_config=config.report(),
        review_config=review_config(config).report(),
        examined_frames=0,
        decoded_frames=0,
        stride=stride,
    )
    best, pairs, error = {}, [], None
    candidate_pool = []
    capture = cv2.VideoCapture(str(video)) if video.is_file() else None
    loose = review_config(config)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture and capture.isOpened() else 0
    limit = min(total, max_frames) if max_frames else total
    fps = capture.get(cv2.CAP_PROP_FPS) if capture and capture.isOpened() else 0.0
    report.update(
        video_fps=float(fps) if np.isfinite(fps) else None,
        sampling_time_basis="frame_index / video_fps",
        scan_phases=[],
    )
    if not limit:
        error = "missing_video" if not video.is_file() else "cannot_decode_video"
    elif not np.isfinite(fps) or fps <= 0:
        error = "invalid_video_fps"
    else:
        seen = set()
        for phase in feature_scan_phases(limit, fps, stride, config):
            if report["scan_phases"] and len(best) == 2:
                break
            if all(r["start"] >= r["stop"] for r in phase["ranges"]):
                continue
            phase["examined_frames"] = 0
            report["scan_phases"].append(phase)
            if phase["name"] != "initial":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for index in range(limit):
                if not capture.grab():
                    error = f"decode_failed_at_{index}"
                    break
                report["decoded_frames"] += 1
                sampled = any(
                    r["start"] <= index < r["stop"] and (index - r["start"]) % r["stride"] == 0
                    for r in phase["ranges"]
                )
                if not sampled or index in seen:
                    continue
                seen.add(index)
                ok, image = capture.retrieve()
                if not ok:
                    error = f"decode_failed_at_{index}"
                    break
                report["examined_frames"] += 1
                phase["examined_frames"] += 1
                h, w = image.shape[:2]
                found = detect_label_candidates(
                    image, {"left": [0, 0, w, h]}, config=loose, preserve_variants=True
                )["candidates"]["left"]
                current = {}
                for candidate in found:
                    if not feature_candidate_allowed(candidate, config):
                        continue
                    if (
                        candidate.get("nearby_white_components", 0)
                        >= config.body_max_white_neighbors
                    ):
                        continue
                    if candidate["quad"][:, 1].mean() < h * config.feature_min_y_fraction:
                        continue
                    if not candidate.get("compact_body"):
                        continue
                    strict = label_metrics(image, candidate["quad"], config)
                    accepted = strict["accepted"] and strict.get("compact_body", False)
                    item = dict(
                        candidate,
                        frame_index=index,
                        status="appearance_candidate" if accepted else "review_candidate",
                        standard_rejection=strict["reason"],
                    )
                    if strict["accepted"]:
                        item.update(strict)
                    if not item.get("compact_body"):
                        item["standard_rejection"] = "noncompact_body_LED_supported"
                    # Black surroundings/enclosure participate even for review-only candidates.
                    item["selection_score"] = item["score"] * (1 if accepted else 0.75)
                    candidate_pool.append(_serialize(item))
                    cx = candidate["quad"][:, 0].mean()
                    side = "left" if cx < w / 2 else "right"
                    if (
                        side not in current
                        or item["selection_score"] > current[side]["selection_score"]
                    ):
                        current[side] = item
                for side, item in current.items():
                    if side not in best or item["selection_score"] > best[side]["selection_score"]:
                        best[side] = item
                if len(current) == 2:
                    pairs.append(current)
            if error:
                break

    if capture:
        capture.release()
    selected, report["mode"] = choose_features(best, pairs, config.pair_quality_ratio)
    return _finish_features(report, selected, output, config, candidate_pool, start, error)


def _finish_features(report, selected, output, config, candidate_pool, start, error=None):
    video = report["video"]
    loose = review_config(config)
    output.mkdir(parents=True, exist_ok=True)
    report.update(
        schema_version=2,
        frame_index_base=0,
        coordinate_system="rectified video pixels; origin top-left; x right; y down",
        corner_order="q0,q1,q2,q3 cyclic in image coordinates; start is not guaranteed top-left",
        center_definition="intersection of diagonals q0-q2 and q1-q3; label centre, not VIVE origin",
    )
    # Reopen at the selected native frames instead of retaining video-sized images in RAM.
    cap = cv2.VideoCapture(str(video)) if selected else None
    frames = {}
    for side in ("left", "right"):
        candidate = selected.get(side)
        feature_path = output / f"{side}_feature.jpg"
        if candidate is not None:
            index = candidate["frame_index"]
            if index not in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    error = f"selected_frame_decode_failed_{index}"
                    candidate = None
                else:
                    frames[index] = image
                    _write_image(output / f"frame_{index:06d}_clean.jpg", image)
            if candidate is not None:
                item = _serialize(candidate)
                q = np.asarray(item["quad"], dtype=float)
                homogeneous = np.column_stack((q, np.ones(4)))
                center = np.cross(
                    np.cross(homogeneous[0], homogeneous[2]),
                    np.cross(homogeneous[1], homogeneous[3]),
                )
                item["label_center_px"] = (center[:2] / center[2]).tolist()
                item["image_size_px"] = list(frames[index].shape[1::-1])
                item["clean_frame_image"] = str(output / f"frame_{index:06d}_clean.jpg")
                item["observation_id"] = hashlib.sha256(
                    json.dumps(
                        [
                            report["video"],
                            feature_video_signature(Path(report["episode"])),
                            side,
                            index,
                            item["quad"],
                        ],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                item["feature_image"] = str(feature_path)
                item["frame_image"] = str(output / f"frame_{index:06d}.jpg")
                report["features"][side] = item
                _write_image(
                    feature_path,
                    feature_image(
                        frames[index],
                        candidate,
                        side,
                        config if item["status"] == "appearance_candidate" else loose,
                    ),
                )
        if candidate is None:
            report["missing_hands"].append(side)
            _write_image(
                feature_path,
                _missing_feature(side, error or "no candidate passed appearance gates"),
            )
    if cap:
        cap.release()
    for index, image in frames.items():
        for side, item in report["features"].items():
            if item["frame_index"] == index:
                quad = np.round(item["quad"]).astype(np.int32)
                color = (0, 255, 0) if side == "left" else (0, 165, 255)
                cv2.polylines(image, [quad], True, color, 3)
                for number, point in enumerate(quad):
                    cv2.putText(
                        image, f"q{number}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )
                cv2.drawMarker(
                    image,
                    tuple(np.round(item["label_center_px"]).astype(int)),
                    color,
                    cv2.MARKER_CROSS,
                    12,
                    2,
                )
                cv2.putText(
                    image,
                    f"{side}: unverified",
                    tuple(quad[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
        _write_image(output / f"frame_{index:06d}.jpg", image)
    if error:
        logging.getLogger(__name__).error(
            "Feature detection failed: episode=%s video=%s reason=%s; rectified video required; raw fallback disabled",
            report["episode"],
            video,
            error,
        )
    review_path = output / "human_review.json"
    if not review_path.exists():
        write_json(
            review_path,
            dict(
                schema_version=1,
                instructions="Review clean frame, overlay and feature image. Verify observation_id before reuse. Never infer physical side from image half alone.",
                observations=[
                    dict(
                        observation_id=item["observation_id"],
                        image_side=side,
                        frame_index=item["frame_index"],
                        status="pending",
                        label_correct=None,
                        corners_correct=None,
                        physical_side="unknown",
                        corrected_quad_px=None,
                        reviewer=None,
                        reviewed_at=None,
                        notes="",
                    )
                    for side, item in report["features"].items()
                ],
            ),
        )
    report["human_review_file"] = str(review_path)
    report.update(
        error=error,
        seconds=time.perf_counter() - start,
        status="error" if error else "features_ready" if not report["missing_hands"] else "partial",
        feature_images={s: str(output / f"{s}_feature.jpg") for s in ("left", "right")},
    )
    write_json(output / "candidate_pool.json", candidate_pool)
    write_json(output / "features.json", report)
    return report


def render_features(output, reports, config_report):
    from os.path import relpath

    delivery = output / "batch/delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    cards = []
    if (delivery / "tracked.html").is_file():
        cards.append('<p><a href="tracked.html">连续跟踪与几何观测报告（独立于特征候选）</a></p>')
    for report in reports:
        name = Path(report["episode"]).parent.name + "/" + Path(report["episode"]).name
        cards.append(
            "<article><h2>"
            + html.escape(name)
            + "</h2><p>"
            + html.escape(report["mode"])
            + " · 侧别未核验 · "
            + html.escape(report.get("video_space", "rectified"))
            + " · "
            + html.escape(report["error"] or report["status"])
            + "</p>"
        )
        for side, item in report["features"].items():
            cards.append(
                "<p>"
                + html.escape(
                    f"{side}: frame_index={item['frame_index']} (0-based); "
                    f"quad={item['quad']}; label_center_px={item.get('label_center_px', 'legacy: unavailable')}"
                )
                + "</p>"
            )
            if item.get("clean_frame_image"):
                cards.append(
                    '<p><a href="'
                    + html.escape(relpath(item["clean_frame_image"], delivery), quote=True)
                    + '">查看无标注原帧</a></p>'
                )
        if report.get("human_review_file"):
            cards.append(
                '<p><a href="'
                + html.escape(relpath(report["human_review_file"], delivery), quote=True)
                + '">人工审核记录模板</a></p>'
            )
        frame_images = sorted({item["frame_image"] for item in report["features"].values()})
        for path in frame_images:
            cards.append(
                '<img class="frame" loading="lazy" src="'
                + html.escape(relpath(path, delivery), quote=True)
                + '">'
            )
        cards.append('<div class="features">')
        for side in ("left", "right"):
            path = report["feature_images"][side]
            cards.append(
                '<img loading="lazy" src="'
                + html.escape(relpath(path, delivery), quote=True)
                + '">'
            )
        cards.append("</div></article>")
    counts = dict(
        episodes=len(reports),
        bilateral=sum(len(r["features"]) == 2 for r in reports),
        same_frame=sum(r["mode"] == "same_frame" and len(r["features"]) == 2 for r in reports),
        missing_video=sum(r["error"] == "missing_video" for r in reports),
        rectified_video_required=True,
    )
    write_json(
        delivery / "features_summary.json",
        dict(counts=counts, detector_config=config_report, human_verified=False, episodes=reports),
    )
    (delivery / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>VIVE CV 特征图</title>'
        "<style>body{font:16px/1.6 sans-serif;margin:24px;background:#eee}article{background:white;padding:20px;margin:24px 0}"
        "img{max-width:100%}.frame{max-height:640px}.features{display:flex;gap:12px}.features img{width:49%;object-fit:contain}</style>"
        "<h1>VIVE 白标签候选与黑色外壳特征</h1><p>同帧双手优先；黄色轮廓为标签，绿色为外壳黑像素，蓝色为外围黑像素。"
        "特征图属于待复核候选，画面左右不保证等于交叉后的物理左右手。缺视频或未检出时明确显示缺失。</p>"
        "<p>" + html.escape(str(counts)) + "</p>" + "".join(cards),
        encoding="utf-8",
    )
    print(f"Feature delivery: {delivery / 'index.html'} {counts}", flush=True)
    return counts


def run_feature_dataset(args, dataset, output, config):
    episodes = select_episodes(dataset, args.episodes_file)
    if not episodes:
        raise ValueError("No episodes found")
    stride = args.coarse_stride
    if stride < 1:
        raise ValueError("Feature scan stride must be positive")
    plan = dict(
        mode="cv_features",
        detector_config=config.report(),
        source_hashes=source_hashes(),
        stride=stride,
        max_frames=args.max_frames,
        episodes=[
            dict(
                path=str(p),
                input_signature=input_signature(p),
                feature_video=feature_video_signature(p),
            )
            for p in episodes
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "feature_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Feature settings or inputs changed; use a new output directory")
    write_json(plan_path, plan)
    reports, pending = {}, []
    for episode in episodes:
        folder = output / "features" / episode.relative_to(dataset)
        path = folder / "features.json"
        if path.exists():
            report = json.loads(path.read_text())
            paths = list(report["feature_images"].values()) + [
                v["frame_image"] for v in report["features"].values()
            ]
            paths += [
                v["clean_frame_image"]
                for v in report["features"].values()
                if "clean_frame_image" in v
            ]
            if report.get("human_review_file"):
                paths.append(report["human_review_file"])
            if (
                report.get("schema_version") == 2
                and all(Path(p).is_file() for p in paths)
                and not report.get("error")
            ):
                reports[str(episode)] = report
                continue
        pending.append((episode, folder))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(scan_features, ep, folder, config, stride, args.max_frames): ep
            for ep, folder in pending
        }
        for future in as_completed(futures):
            ep = futures[future]
            reports[str(ep)] = future.result()
            print(
                f"[{len(reports)}/{len(episodes)}] {ep.parent.name}/{ep.name}: "
                f"{reports[str(ep)]['mode']}, sides={list(reports[str(ep)]['features'])}",
                flush=True,
            )
            render_features(
                output, [reports[str(p)] for p in episodes if str(p) in reports], config.report()
            )
    render_features(output, [reports[str(p)] for p in episodes], config.report())
    return int(any(r["error"] for r in reports.values()))


def refilter_features(source_folder, output, config, rescan_missing=True):
    """Tighten feature gates using an auditable, unchanged-input candidate pool."""
    start = time.perf_counter()
    previous = json.loads((source_folder / "features.json").read_text())
    expected_video, _ = feature_video(Path(previous["episode"]))
    if (
        previous.get("video_space") != "rectified"
        or Path(previous["video"]).resolve() != expected_video.resolve()
    ):
        logging.getLogger(__name__).error(
            "Rejecting non-rectified candidate cache: %s", source_folder
        )
        raise ValueError("Only rectified video candidate caches are supported; rescan required")
    if not expected_video.is_file():
        return scan_features(Path(previous["episode"]), output, config, previous["stride"])
    cache_key = hashlib.sha256(
        (
            source_folder.joinpath("features.json").read_text()
            + source_folder.joinpath("candidate_pool.json").read_text()
            + json.dumps(config.report(), sort_keys=True)
            + json.dumps(feature_video_signature(Path(previous["episode"])), sort_keys=True)
        ).encode()
    ).hexdigest()
    saved = output / "features.json"
    if saved.exists():
        existing = json.loads(saved.read_text())
        paths = list(existing.get("feature_images", {}).values()) + [
            v["frame_image"] for v in existing.get("features", {}).values()
        ]
        paths += [
            v["clean_frame_image"]
            for v in existing.get("features", {}).values()
            if "clean_frame_image" in v
        ]
        if existing.get("human_review_file"):
            paths.append(existing["human_review_file"])
        if (
            existing.get("schema_version") == 2
            and existing.get("refilter_cache_key") == cache_key
            and not existing.get("error")
            and (not rescan_missing or not existing.get("missing_hands"))
            and paths
            and all(Path(p).is_file() for p in paths)
        ):
            return existing
    before = previous["detector_config"]["parameters"].copy()
    after = config.report()["parameters"].copy()
    old_angle = before.pop("feature_min_corner_angle", 0.0)
    new_angle = after.pop("feature_min_corner_angle")
    old_limit = before.pop("max_label_black_fraction", 1.0)
    new_limit = after.pop("max_label_black_fraction")
    if before != after or new_limit > old_limit or new_angle < old_angle:
        raise ValueError("Cached filtering only supports tightening feature black/angle gates")
    pool = json.loads((source_folder / "candidate_pool.json").read_text())
    cap = cv2.VideoCapture(previous["video"])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    if width <= 0:
        logging.getLogger(__name__).error(
            "Cannot decode rectified cached source video: %s; raw fallback disabled",
            previous["video"],
        )
        raise ValueError("Cannot decode cached source video")
    kept = [c for c in pool if feature_candidate_allowed(c, config)]
    by_frame, best = {}, {}
    for candidate in kept:
        side = "left" if np.asarray(candidate["quad"])[:, 0].mean() < width / 2 else "right"
        frame = by_frame.setdefault(candidate["frame_index"], {})
        if side not in frame or candidate["selection_score"] > frame[side]["selection_score"]:
            frame[side] = candidate
        if side not in best or candidate["selection_score"] > best[side]["selection_score"]:
            best[side] = candidate
    if len(best) < 2 and rescan_missing:
        result = scan_features(Path(previous["episode"]), output, config, previous["stride"])
        result["refilter_cache_key"] = cache_key
        result["cache_source"] = str(source_folder)
        result["cache_action"] = "rescanned_missing_side"
        write_json(output / "features.json", result)
        return result
    pairs = [v for v in by_frame.values() if len(v) == 2]
    selected, mode = choose_features(best, pairs, config.pair_quality_ratio)
    report = dict(
        previous,
        features={},
        missing_hands=[],
        mode=mode,
        detector_config=config.report(),
        review_config=review_config(config).report(),
        cache_source=str(source_folder),
        cache_action="tightened_feature_gates",
        removed_candidates=len(pool) - len(kept),
        refilter_cache_key=cache_key,
    )
    return _finish_features(report, selected, output, config, kept, start)


def refilter_main():
    import argparse
    from vive_correct.estimate.cv_config import CVConfig

    parser = argparse.ArgumentParser(
        description="Tighten cached CV candidates; rescan missing sides"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cv-config", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-rescan-missing", action="store_true")
    parser.add_argument("--expanded-source", type=Path)
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    config = CVConfig.load(args.cv_config)
    plan = json.loads((args.source / "feature_plan.json").read_text())
    summary = json.loads((args.source / "batch/delivery/features_summary.json").read_text())
    if len(summary["episodes"]) != len(plan["episodes"]):
        raise ValueError("Source scan is incomplete")
    for item in plan["episodes"]:
        ep = Path(item["path"])
        if item["input_signature"] != input_signature(ep) or item[
            "feature_video"
        ] != feature_video_signature(ep):
            raise ValueError(f"Cached input changed: {ep}")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "refilter_plan.json").exists():
        raise ValueError("Use a fresh output directory for cache refinement")
    write_json(
        args.out / "refilter_plan.json",
        dict(
            source_plan=plan,
            source=str(args.source.resolve()),
            expanded_source=str(args.expanded_source.resolve()) if args.expanded_source else None,
            detector_config=config.report(),
            source_hashes=source_hashes(),
        ),
    )
    reports = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for r in summary["episodes"]:
            relative = Path("features") / Path(r["episode"]).parent.name / Path(r["episode"]).name
            source_folder = args.source / relative
            if args.expanded_source:
                expanded = args.expanded_source / relative
                if (expanded / "features.json").exists():
                    extra = json.loads((expanded / "features.json").read_text())
                    if (
                        extra.get("cache_action") == "rescanned_missing_side"
                        and extra["episode"] == r["episode"]
                    ):
                        source_folder = expanded
            futures[
                pool.submit(
                    refilter_features,
                    source_folder,
                    args.out / relative,
                    config,
                    not args.no_rescan_missing,
                )
            ] = r["episode"]
        for future in as_completed(futures):
            episode = futures[future]
            reports[episode] = future.result()
            render_features(args.out, [reports[k] for k in sorted(reports)], config.report())
    return int(any(r["error"] for r in reports.values()))


if __name__ == "__main__":
    raise SystemExit(refilter_main())
