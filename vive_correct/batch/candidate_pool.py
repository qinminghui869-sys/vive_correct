"""Exhaustive candidate export for identity review before final pose selection.

Unlike the quick preview pipeline, a candidate pair never stops this scan.
All candidates retain their stability result; only passing candidates can be
accepted for reconstruction. Image-side hints are not physical wrist identities.
"""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vive_correct.estimate.find_stable_frame import (
    Episode,
    SelectionConfig,
    fit_candidate,
    frame_windows,
    measure_stability,
)
from vive_correct.estimate.label_detection import detect_labels, has_tracker_body


def scan(
    episode_path,
    output,
    config=None,
    *,
    preserve_variants=False,
    rois=None,
    frame_range=None,
    include_unstable=False,
):
    episode = Episode.load(episode_path)
    config = config or SelectionConfig()
    if frame_range is not None:
        start, stop = frame_range
        if not 0 <= start < stop <= len(episode.timestamps):
            raise ValueError("Frame range must be within the video, with an exclusive stop")
    output = Path(output)
    (output / "crops").mkdir(parents=True, exist_ok=True)
    width, height = episode.camera.resolution
    # Full-height proposals; identity review replaces the old wrist-light shortcut.
    rois = rois or {"left": [0, 0, width // 2, height], "right": [width // 2, 0, width, height]}
    candidates, rejected = [], Counter()
    examined = 0
    for index, image, grays in frame_windows(episode, config):
        if frame_range is not None and not start <= index < stop:
            continue
        examined += 1
        for hint, quads in detect_labels(
            image,
            rois,
            preserve_variants=preserve_variants,
            min_edge_px=config.min_projected_edge_px,
        ).items():
            for quad in quads:
                try:
                    pose = fit_candidate(
                        image, quad, episode.camera, min_edge_px=config.min_projected_edge_px
                    )
                except (ValueError, cv2.error):
                    rejected["geometry"] += 1
                    continue
                stability = measure_stability(
                    grays, pose["image_corners_px"], episode.camera.resolution, config
                )
                if not stability["passed"]:
                    rejected[stability["reason"]] += 1
                    if not include_unstable:
                        continue
                q = np.array(pose["image_corners_px"])
                centre = q.mean(0)
                area = cv2.contourArea(q.astype(np.float32))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                body = has_tracker_body(gray, q)
                # Ranking controls review order only; it does not establish identity.
                quality = np.sqrt(area) * (2 if body else 1) / (1 + pose["rms_px"])
                key = f"f{index:06d}_{hint}_{len(candidates):05d}"
                x, y = centre.astype(int)
                xa, ya = max(0, x - 150), max(0, y - 140)
                xb, yb = min(width, x + 150), min(height, y + 160)
                crop = image[ya:yb, xa:xb].copy()
                cv2.polylines(
                    crop, [np.round(q - [xa, ya]).astype(np.int32)], True, (0, 220, 255), 1
                )
                if not cv2.imwrite(str(output / "crops" / f"{key}.jpg"), crop):
                    raise OSError("Cannot save candidate crop")
                candidates.append(
                    {
                        "id": key,
                        "frame_index": index,
                        "side_hint": hint,
                        "quality": float(quality),
                        "area_px2": area,
                        "compact_body": body,
                        "pose": pose,
                        "stability": stability,
                    }
                )
    report = {
        "episode": str(episode.path),
        "video": str(episode.video),
        "selection_config": asdict(config),
        "examined_windows": examined,
        "rejected": dict(rejected),
        "identity": "unreviewed",
        "candidates": candidates,
        "preserve_threshold_variants": preserve_variants,
        "search_rois": rois,
        "frame_range": frame_range,
        "includes_unstable_diagnostics": include_unstable,
    }
    (output / "candidates.json").write_text(json.dumps(report, indent=2) + "\n")
    render_review(output)
    return report


def review_order(candidates):
    """Show diverse spatial/time candidates first, without deleting alternatives."""
    ranked = sorted(candidates, key=lambda c: -c["quality"])
    first, rest = [], []
    for candidate in ranked:
        centre = np.array(candidate["pose"]["image_corners_px"]).mean(0)
        similar = any(
            abs(candidate["frame_index"] - old["frame_index"]) < 150
            and np.linalg.norm(centre - np.array(old["pose"]["image_corners_px"]).mean(0)) < 90
            for old in first
        )
        (rest if similar else first).append(candidate)
    return first + rest


def render_review(output, limit=96):
    output = Path(output)
    report = json.loads((output / "candidates.json").read_text())
    ordered = review_order(report["candidates"])
    (output / "review_order.json").write_text(
        json.dumps([c["id"] for c in ordered], indent=2) + "\n"
    )
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for start in range(0, min(limit, len(ordered)), 24):
        group = ordered[start : start + 24]
        sheet = Image.new("RGB", (6 * 240, 4 * 270), "#eef0f2")
        draw = ImageDraw.Draw(sheet)
        for offset, candidate in enumerate(group):
            x, y = offset % 6 * 240, offset // 6 * 270
            crop = Image.open(output / "crops" / f"{candidate['id']}.jpg")
            crop.thumbnail((236, 236))
            sheet.paste(crop, (x + (240 - crop.width) // 2, y + 32))
            draw.text(
                (x + 3, y + 3),
                f"{start + offset:03d} f{candidate['frame_index']} {candidate['side_hint']}",
                font=font,
                fill="#17232b",
            )
        sheet.save(output / f"review_{start // 24:02d}.jpg", quality=94)


if __name__ == "__main__":
    import argparse

    from vive_correct.config import estimate_settings

    _settings = estimate_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=_settings["selection"]["stride"])
    parser.add_argument(
        "--window",
        type=int,
        default=_settings["selection"]["window"],
        help="Odd stability window in frames; exported in the report",
    )
    parser.add_argument("--preserve-variants", action="store_true")
    parser.add_argument(
        "--min-edge-px",
        type=float,
        default=_settings["geometry"]["min_edge_px"],
        help="Minimum projected edge; values below geometry.min_edge_px flag low-resolution poses",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report = scan(
        args.episode,
        args.out,
        SelectionConfig(
            stride=args.stride, window=args.window, min_projected_edge_px=args.min_edge_px
        ),
        preserve_variants=args.preserve_variants,
    )
    print(f"{len(report['candidates'])} stable candidates; {report['examined_windows']} windows")
