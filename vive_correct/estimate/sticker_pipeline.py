"""Estimate stable white-label poses from rectified headband-left video."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from vive_correct.estimate.find_stable_frame import Episode, SelectionConfig, select_frames
from vive_correct.estimate.geometry import LABEL_SIZE_M, VIDEO_KEY
from vive_correct.estimate.label_detection import default_rois, detect_labels, validate_rois
from vive_correct.config import estimate_settings
from vive_correct.estimate.visualization import AXIS_LENGTH_M, draw_poses, save_image

_SETTINGS = estimate_settings()

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
DEFAULT_EPISODE = REPO / "samples/dining/L3-001-2026-10-26-13-48-50"


def read_observations(path, episode):
    """Accept a small review file or a previously exported experiment result."""
    doc = json.loads(Path(path).read_text())
    if doc.get("episode") and Path(doc["episode"]).resolve() != episode.path:
        raise ValueError("Reviewed observations belong to a different episode")
    frame = doc.get("frame_index", doc.get("frame_zero_based"))
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise ValueError("Observations require an integer frame_index")
    labels = doc.get("labels")
    if labels is None:
        labels = {
            side: hand["corners_px_in_object_order"] for side, hand in doc.get("hands", {}).items()
        }
    if not labels or set(labels) - {"left", "right"}:
        raise ValueError("Observations require left and/or right label corners")
    return frame, {side: [corners] for side, corners in labels.items()}


def run(
    episode,
    out_dir=None,
    *,
    detector="cv",
    frame=None,
    observations=None,
    rois=None,
    config=None,
    api_key=None,
    base_url=None,
    model=None,
):
    episode = Episode.load(episode)
    config = config or SelectionConfig()
    if detector not in {"cv", "ai"}:
        raise ValueError("Detector must be cv or ai")
    if observations is not None and (detector != "cv" or frame is not None or rois is not None):
        raise ValueError("--observations cannot be combined with --frame, --rois or --detector ai")
    if rois is not None:
        validate_rois(rois, (episode.camera.resolution[1], episode.camera.resolution[0]))
    identity = "cv_candidate_requires_visual_review"
    model_name = None
    detector_errors = {}
    if observations is not None:
        frame, labels = read_observations(observations, episode)
        provider = lambda image: labels
        identity = "user_supplied_observations"
    elif detector == "ai":
        if frame is None:
            raise ValueError(
                "--detector ai requires --frame; automatic scanning never sends video to a model"
            )
        from vive_correct.estimate.openai_vision import load_config, locate_stickers

        ai_config = load_config(api_key, base_url, model)
        if ai_config is None:
            raise ValueError("AI detector requires an explicitly configured API key")
        model_name = ai_config.model

        def provider(image):
            result = locate_stickers(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                rois or default_rois(image.shape),
                ai_config,
            )
            detector_errors.update(result.get("errors", {}))
            return {
                side: [item["quad"]] for side, item in result["sides"].items() if item["visible"]
            }

        identity = "model_candidate_requires_visual_review"
    else:
        provider = lambda image: detect_labels(
            image, rois, min_edge_px=config.min_projected_edge_px
        )
    selected = select_frames(episode, provider, config, frame, refine=observations is None)
    out_dir = (
        Path(out_dir)
        if out_dir
        else PACKAGE / "outputs" / f"{episode.path.parent.name}_{episode.path.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "episode": str(episode.path),
        "video_key": VIDEO_KEY,
        "camera": {
            "K": episode.camera.K.tolist(),
            "resolution": list(episode.camera.resolution),
            "source": episode.camera.source,
            "distortion": "none; already rectified",
        },
        "label_size_m": list(LABEL_SIZE_M),
        "origin": "label centre; Vive mounting transform is not calibrated",
        "local_axes_convention": "unresolved",
        "axis_convention": "Device installation reference required; origin only without an anchor",
        "axis_length_m": AXIS_LENGTH_M,
        "selection_config": asdict(config),
        "mode": selected["mode"],
        "examined_windows": selected["examined_windows"],
        "rejected": selected["rejected"],
        "identity": identity,
        "model": model_name,
        "detector_errors": detector_errors,
        "side_assignment": "provided labels"
        if observations
        else "image-side assumption; uncrossed hands",
        "pose_accuracy_validated": False,
        "observations": [],
    }
    for i, observation in enumerate(selected["observations"]):
        image = observation["image"]
        labels = observation["labels"]
        source_path = out_dir / f"frame_{observation['frame_index']:06d}.png"
        overlay_path = out_dir / f"frame_{observation['frame_index']:06d}_axes.png"
        overlay = draw_poses(image, labels, episode.camera)
        save_image(source_path, image)
        save_image(overlay_path, overlay)
        if i == 0:
            save_image(out_dir / "sticker_pose_rgb.png", overlay)
        item = {
            key: observation[key]
            for key in ("frame_index", "timestamp_s", "window", "window_duration_s")
        }
        item.update(image=str(source_path), overlay=str(overlay_path), stickers=labels)
        if set(labels) == {"left", "right"} and all(
            p.get("axes_generated", True) for p in labels.values()
        ):
            item["T_left_local_right_local"] = (
                np.linalg.inv(labels["left"]["T_camera_local"]) @ labels["right"]["T_camera_local"]
            ).tolist()
        report["observations"].append(item)
    if not report["observations"]:
        (out_dir / "sticker_pose_rgb.png").unlink(missing_ok=True)
    (out_dir / "sticker_pose_report_new.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--frame", type=int, help="Check this frame and its complete stability window"
    )
    parser.add_argument(
        "--observations", type=Path, help="Reviewed frame and left/right corner JSON"
    )
    parser.add_argument(
        "--rois", type=Path, help="Search boxes: {left: [x0,y0,x1,y1], right: [...]}"
    )
    parser.add_argument("--stride", type=int, default=_SETTINGS["selection"]["stride"])
    parser.add_argument("--max-frames", type=int, default=_SETTINGS["selection"]["max_frames"])
    parser.add_argument("--detector", choices=("cv", "ai"), default=_SETTINGS["detector"]["name"])
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        config = SelectionConfig(stride=args.stride, max_frames=args.max_frames)
        rois = json.loads(args.rois.read_text()) if args.rois else None
        report = run(
            args.episode,
            args.out,
            detector=args.detector,
            frame=args.frame,
            observations=args.observations,
            rois=rois,
            config=config,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(
        f"{report['mode']}: {len(report['observations'])} observed frame(s), "
        f"{report['examined_windows']} windows checked"
    )
    return 0 if report["observations"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
