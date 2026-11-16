"""Prepare video-disjoint CV evaluation and score explicitly reviewed annotations.

python -m vive_correct.cv_evaluation prepare --dataset samples --out outputs/cv_eval
python -m vive_correct.cv_evaluation score --dataset samples --annotations FILE --out DIR
"""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np

from vive_correct.dataset import select_episodes, VIDEO_SUFFIX
from vive_correct.batch.report_io import write_json
from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.label_detection import detect_labels, quad_iou
from vive_correct.estimate.cv_tracking import align_quad


def prepare(dataset, output):
    groups = defaultdict(list)
    for episode in select_episodes(dataset):
        groups[episode.parent.name].append(episode)
    splits, rows = {}, []
    for scene, episodes in sorted(groups.items()):
        # Whole episodes, never adjacent frames, define the split.
        split_at = max(1, int(len(episodes) * 0.7))
        for index, episode in enumerate(sorted(episodes)):
            relative = episode.relative_to(dataset).as_posix()
            split = "tuning" if index < split_at else "holdout"
            splits[relative] = split
            capture = cv2.VideoCapture(str(episode / VIDEO_SUFFIX))
            try:
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            for frame in sorted(set(np.linspace(0, max(count - 1, 0), 12).astype(int))):
                rows.append(
                    dict(
                        episode=relative,
                        frame_index=int(frame),
                        split=split,
                        human_reviewed=False,
                        labels=[],
                    )
                )
    write_json(output / "split.json", splits)
    write_json(output / "annotations.json", {"schema_version": 1, "frames": rows})
    return splits


def score_frames(dataset, annotations, detector, config=None):
    counts = dict(tp=0, fp=0, fn=0, frames=0)
    errors, elapsed = [], 0.0
    for row in annotations:
        if row.get("human_reviewed") is not True:
            continue
        episode = (dataset / row["episode"]).resolve()
        if not episode.is_relative_to(dataset.resolve()):
            raise ValueError("Annotation episode escapes dataset")
        capture = cv2.VideoCapture(str(episode / VIDEO_SUFFIX))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode annotated frame: {row}")
        finally:
            capture.release()
        start = time.perf_counter()
        kwargs = {} if config is None else {"config": config}
        result = detector(image, {"left": [0, 0, image.shape[1], image.shape[0]]}, **kwargs)
        elapsed += time.perf_counter() - start
        predictions = result["left"]
        truth = [np.asarray(label["quad"], float) for label in row["labels"]]
        if any(q.shape != (4, 2) or not np.isfinite(q).all() for q in truth):
            raise ValueError("Annotations require finite four-corner quads in cyclic order")
        pairs = sorted(
            (
                (quad_iou(p, q), i, j)
                for i, p in enumerate(predictions)
                for j, q in enumerate(truth)
            ),
            reverse=True,
        )
        used_p, used_q = set(), set()
        for overlap, i, j in pairs:
            if overlap < 0.5 or i in used_p or j in used_q:
                continue
            used_p.add(i)
            used_q.add(j)
            aligned = min(
                (align_quad(truth[j], p) for p in (predictions[i], predictions[i][::-1])),
                key=lambda q: np.linalg.norm(truth[j] - q, axis=1).mean(),
            )
            errors.extend(np.linalg.norm(truth[j] - aligned, axis=1).tolist())
        counts["tp"] += len(used_p)
        counts["fp"] += len(predictions) - len(used_p)
        counts["fn"] += len(truth) - len(used_q)
        counts["frames"] += 1
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return dict(
        **counts,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        median_corner_error_px=float(np.median(errors)) if errors else None,
        seconds=elapsed,
        acceptance="pending" if not counts["frames"] else "requires_best_observation_audit",
    )


def audit_best(summary, annotations):
    lookup = {
        (row["episode"], row["frame_index"]): row
        for row in annotations
        if row.get("human_reviewed") is True
    }
    checked = correct = false_labels = wrong_side = unchecked = 0
    relevant = {row["episode"] for row in annotations}
    for episode in summary["episodes"]:
        if not any(Path(episode["episode"]).as_posix().endswith("/" + name) for name in relevant):
            continue
        for side, hand in episode["hands"].items():
            # Summary carries absolute episode paths; match the manifest suffix exactly.
            matches = [
                row
                for (name, index), row in lookup.items()
                if Path(episode["episode"]).as_posix().endswith("/" + name)
                and index == hand["frame_index"]
            ]
            if len(matches) != 1:
                unchecked += 1
                continue
            checked += 1
            q = np.asarray(hand["pose"]["image_corners_px"], float)
            labels = matches[0]["labels"]
            match = max(labels, key=lambda label: quad_iou(q, label["quad"]), default=None)
            if match is None or quad_iou(q, match["quad"]) < 0.5:
                false_labels += 1
            elif match["side"] != side:
                wrong_side += 1
            else:
                correct += 1
    return dict(
        checked=checked,
        correct=correct,
        false_labels=false_labels,
        wrong_side=wrong_side,
        unchecked=unchecked,
        accepted_hand_sides=sum(len(e["hands"]) for e in summary["episodes"]),
        identity_switches=None,
        identity_switch_note="Best-only outputs cannot measure per-frame identity switches",
    )


def audit_tracks(report, annotations):
    lookup = {
        row["frame_index"]: row
        for row in annotations
        if row.get("human_reviewed") is True
        and Path(report["episode"]).as_posix().endswith("/" + row["episode"])
    }
    previous, switches, wrong, checked = {}, 0, 0, 0
    for item in report.get("track_observations", []):
        row = lookup.get(item["frame_index"])
        if row is None:
            continue
        for side, q in item["labels"].items():
            match = max(row["labels"], key=lambda label: quad_iou(q, label["quad"]), default=None)
            if match is None or quad_iou(q, match["quad"]) < 0.5:
                continue
            checked += 1
            physical_side = match["side"]
            wrong += int(side != physical_side)
            switches += int(side in previous and previous[side] != physical_side)
            previous[side] = physical_side
    return dict(
        checked_associations=checked,
        wrong_side=wrong,
        identity_switches=switches if checked else None,
        scope="Reviewed sampled frames only; unannotated intervals are not validated",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--baseline-module", type=Path)
    parser.add_argument(
        "--results", type=Path, help="CV delivery summary.json for best-observation audit"
    )
    parser.add_argument("--cv-config", type=Path)
    parser.add_argument("--tracking-report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.dataset.resolve(), args.out)
        return 0
    if args.annotations is None:
        parser.error("score requires --annotations")
    rows = json.loads(args.annotations.read_text())["frames"]
    by_episode = defaultdict(set)
    for row in rows:
        if row.get("split") not in ("tuning", "holdout"):
            raise ValueError("Each annotation requires a tuning/holdout split")
        by_episode[row["episode"]].add(row["split"])
    if any(len(splits) != 1 for splits in by_episode.values()):
        raise ValueError("Video leakage between tuning and holdout")
    config = CVConfig.load(args.cv_config)
    result = {"detector_config": config.report(), "splits": {}}
    baseline = None
    if args.baseline_module:
        spec = importlib.util.spec_from_file_location("cv_baseline", args.baseline_module)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    for split in ("tuning", "holdout"):
        selected = [row for row in rows if row["split"] == split]
        metrics = {"current": score_frames(args.dataset, selected, detect_labels, config)}
        if baseline:
            metrics["baseline"] = score_frames(args.dataset, selected, baseline.detect_labels)
        if args.results:
            metrics["best_observations"] = audit_best(
                json.loads(args.results.read_text()), selected
            )
        if args.tracking_report:
            metrics["tracking"] = audit_tracks(
                json.loads(args.tracking_report.read_text()), selected
            )
        result["splits"][split] = metrics
    write_json(args.out / "metrics.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
