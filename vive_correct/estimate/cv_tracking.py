"""Sequential identity association and conservative per-hand best observations."""

from collections import Counter, deque

import cv2
import numpy as np

from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.geometry import MIN_EDGE_PX
from vive_correct.estimate.label_detection import detect_label_candidates


SIDE_SOURCE = "first reliable bilateral window assumed uncrossed; LK association thereafter"


def align_quad(reference, candidate):
    return min(
        (np.roll(candidate, i, axis=0) for i in range(4)),
        key=lambda q: np.linalg.norm(reference - q, axis=1).mean(),
    )


class SideTracker:
    def __init__(self, window=11, max_fb_px=1.5, config=None):
        self.window, self.max_fb_px = window, max_fb_px
        self.config = config or CVConfig.load()
        self.previous = None
        self.tracks = {}
        self.pending = 0
        self.initialized = False
        self.index = None
        self.events = Counter()
        self.missed = {}

    def _flow(self, gray, quad):
        source = np.asarray(quad, np.float32).reshape(-1, 1, 2)
        target, ok, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, source, None, winSize=(21, 21), maxLevel=3
        )
        if target is None or not ok.all():
            return None
        back, valid, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous, target, None, winSize=(21, 21), maxLevel=3
        )
        if (
            back is None
            or not valid.all()
            or not np.isfinite(back).all()
            or np.max(np.linalg.norm(back - source, axis=2)) > self.max_fb_px
        ):
            return None
        q = target.reshape(4, 2).astype(float)
        if not np.isfinite(q).all() or not cv2.isContourConvex(q.astype(np.float32)):
            return None
        if np.any(q < 2) or np.any(q > np.array(gray.shape[::-1]) - 3):
            return None
        return q

    def update(self, index, gray, candidates):
        if self.index is not None and index != self.index + 1:
            # Skipped frames invalidate identity propagation, including pending initialization.
            self.tracks.clear()
            self.pending = 0
            self.events["frame_gap"] += 1
        matched, next_tracks, used = {}, {}, {}
        if self.previous is not None:
            for side, quad in self.tracks.items():
                predicted = self._flow(gray, quad)
                if predicted is None:
                    self.events["tracking_failed"] += 1
                    continue
                scale = max(
                    np.linalg.norm(predicted - np.roll(predicted, -1, axis=0), axis=1).min(), 1
                )
                distances = sorted(
                    (
                        float(
                            np.linalg.norm(
                                predicted - align_quad(predicted, c["quad"]), axis=1
                            ).mean()
                            / scale
                        ),
                        i,
                    )
                    for i, c in enumerate(candidates)
                )
                if not distances or distances[0][0] > self.config.association_fraction:
                    # Short flow-only bridge; never selected as an observation.
                    self.missed[side] = self.missed.get(side, 0) + 1
                    if self.missed[side] <= 3:
                        next_tracks[side] = predicted
                    self.events["no_detection"] += 1
                    continue
                if (
                    len(distances) > 1
                    and distances[1][0] - distances[0][0] < self.config.association_margin
                ):
                    self.events["ambiguous_association"] += 1
                    continue
                i = distances[0][1]
                used[side] = i
                next_tracks[side] = align_quad(predicted, candidates[i]["quad"])
                matched[side] = dict(candidates[i], quad=next_tracks[side])
                self.missed[side] = 0
            if len(used) == 2 and len(set(used.values())) == 1:
                next_tracks.clear()
                matched.clear()
                self.events["identity_collision"] += 1
        if not self.initialized:
            if set(matched) == {"left", "right"}:
                self.pending += 1
            else:
                self.pending = 0
                matched, next_tracks = {}, {}
                # Exactly two strong candidates: distractors cannot seed a handedness guess.
                if len(candidates) == 2:
                    ordered = sorted(candidates, key=lambda c: c["quad"][:, 0].mean())
                    lx, rx = [c["quad"][:, 0].mean() for c in ordered]
                    if lx < gray.shape[1] * 0.45 and rx > gray.shape[1] * 0.55:
                        matched = dict(zip(("left", "right"), ordered))
                        next_tracks = {s: c["quad"] for s, c in matched.items()}
                        self.pending = 1
            if self.pending >= self.window:
                self.initialized = True
                self.events["initialized"] += 1
        self.tracks = next_tracks
        self.previous, self.index = gray, index
        # Pending matches are returned for buffering, never accepted before initialized.
        return matched


def select_tracked_frames(frames, *, selection, camera=None, detector_config=None):
    from vive_correct.estimate.find_stable_frame import fit_candidate, measure_stability

    config = detector_config or CVConfig.load()
    tracker = SideTracker(selection.window, selection.max_fb_px, config)
    history = deque(maxlen=selection.window)
    best, rejected, examined = {}, Counter(), 0
    trace = []
    pairs = []
    for index, image in frames:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        detected = detect_label_candidates(
            image,
            {"left": [0, 0, width, height]},
            config=config,
            preserve_variants=True,
            min_edge_px=max(MIN_EDGE_PX, selection.min_projected_edge_px),
        )
        rejected.update(detected["rejected"])
        matched = tracker.update(
            index,
            gray,
            [
                c
                for c in detected["candidates"]["left"]
                if c.get("compact_body")
                and c.get("box_supported")
                and c.get("nearby_white_components", 0) < config.body_max_white_neighbors
            ],
        )
        if tracker.initialized:
            trace.append(
                {
                    "frame_index": index,
                    "labels": {
                        side: candidate["quad"].tolist() for side, candidate in matched.items()
                    },
                }
            )
        history.append((index, image, gray, matched))
        examined += 1
        if not tracker.initialized or len(history) != selection.window:
            continue
        if history[-1][0] - history[0][0] != selection.window - 1:
            continue
        middle = history[selection.window // 2]
        frame_index, frame, _, labels = middle
        frame_pair = {}
        for side, candidate in labels.items():
            if any(side not in item[3] for item in history):
                continue
            try:
                stability = measure_stability(
                    [item[2] for item in history], candidate["quad"], (width, height), selection
                )
                if not stability["passed"]:
                    rejected[stability["reason"]] += 1
                    continue
                pose = (
                    fit_candidate(
                        frame,
                        candidate["quad"],
                        camera,
                        refine=False,
                        min_edge_px=max(MIN_EDGE_PX, selection.min_projected_edge_px),
                    )
                    if camera
                    else None
                )
            except (ValueError, cv2.error):
                rejected["geometry_gate"] += 1
                continue
            temporal_penalty = (
                0.1 * stability["fb_max_px"] / selection.max_fb_px
                + 0.1 * stability["p95_motion_px_per_frame"] / selection.max_p95_motion_px
            )
            quality = candidate["score"] / (1 + temporal_penalty)
            diagnostics = {k: v for k, v in candidate.items() if k != "quad"}
            diagnostics.update(
                side_source=SIDE_SOURCE,
                tracking_status="associated",
                identity_verified=False,
                selection_score=quality,
            )
            if pose is not None:
                pose.update(stability=stability, cv_diagnostics=diagnostics)
            frame_pair[side] = dict(
                frame_index=frame_index,
                image=frame,
                quad=candidate["quad"],
                score=quality,
                pose=pose,
                diagnostics=diagnostics,
                window=[history[0][0], history[-1][0]],
            )
            if side not in best or quality > best[side]["score"]:
                best[side] = frame_pair[side]
        if len(frame_pair) == 2:
            pairs.append(frame_pair)
            pairs.sort(key=lambda p: min(item["score"] for item in p.values()), reverse=True)
            pairs = pairs[:8]
    eligible = [
        p
        for p in pairs
        if all(
            p[s]["score"] >= config.pair_quality_ratio * best[s]["score"] for s in ("left", "right")
        )
    ]
    if eligible:
        best = eligible[0]
    return {
        "best": best,
        "same_frame_preferred": bool(eligible),
        "examined_frames": examined,
        "rejected": dict(rejected),
        "tracking_events": dict(tracker.events),
        "side_assignment": SIDE_SOURCE,
        "track_observations": trace,
        "identity_verified": False,
        "detector_config": config.report(),
    }


def select_episode(episode, selection, detector_config=None):
    def frames():
        cap = cv2.VideoCapture(str(episode.video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot decode {episode.video}")
            count = len(episode.timestamps)
            if selection.max_frames is not None:
                count = min(count, selection.max_frames)
            for index in range(count):
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Video decode failed at frame {index}")
                yield index, image
        finally:
            cap.release()

    result = select_tracked_frames(
        frames(), selection=selection, camera=episode.camera, detector_config=detector_config
    )
    observations = []
    for side, best in result.pop("best").items():
        a, b = best["window"]
        observations.append(
            dict(
                frame_index=best["frame_index"],
                timestamp_s=float(episode.timestamps[best["frame_index"]]),
                window=[a, b],
                window_duration_s=float(episode.timestamps[b] - episode.timestamps[a]),
                image=best["image"],
                labels={side: best["pose"]},
            )
        )
    if len(observations) == 2 and observations[0]["frame_index"] == observations[1]["frame_index"]:
        observations[0]["labels"].update(observations[1]["labels"])
        observations = observations[:1]
    result.update(
        observations=sorted(observations, key=lambda item: item["frame_index"]),
        mode="both_hands_same_frame"
        if observations and len(observations[0]["labels"]) == 2
        else {0: "no_usable_label", 1: "single_hand", 2: "independent_best_frames"}[
            len(observations)
        ],
        examined_windows=result["examined_frames"],
    )
    return result
