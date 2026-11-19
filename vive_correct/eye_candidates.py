"""Frame-level eye candidates from high-quality labels in opposite image halves."""

from itertools import product

import numpy as np

from vive_correct.estimate.label_detection import quad_iou


def sample_indices(total, fps):
    """Start at or after video time 10 s; examine at most 100 frames, stride 5."""
    start = int(np.ceil(10 * fps))
    return range(start, min(total, start + 500), 5)


def distinct_labels(a, b):
    qa, qb = np.asarray(a["quad"], np.float32), np.asarray(b["quad"], np.float32)
    return quad_iou(qa, qb) < 0.05 and np.linalg.norm(qa.mean(0) - qb.mean(0)) > max(
        np.linalg.norm(np.ptp(qa, axis=0)), np.linalg.norm(np.ptp(qb, axis=0))
    )


def joint_candidate(image, labels, frame_index):
    """A single label proposes a common frame; the missing hand needs human review."""
    halves = {"left": [], "right": []}
    for item in labels:
        if item["status"] != "appearance_candidate":
            continue
        x = np.asarray(item["quad"])[:, 0]
        # A label crossing the midline does not establish either image half.
        side = (
            "left"
            if x.max() < image.shape[1] / 2
            else "right"
            if x.min() >= image.shape[1] / 2
            else None
        )
        if side:
            halves[side].append(item)
    pairs = [(a, b) for a, b in product(halves["left"], halves["right"]) if distinct_labels(a, b)]
    if pairs:
        a, b = max(pairs, key=lambda pair: min(c["selection_score"] for c in pair))
        hands = dict(left=a, right=b)
    else:
        available = [(s, c) for s, items in halves.items() for c in items]
        if not available:
            return None
        side, item = max(available, key=lambda v: v[1]["selection_score"])
        hands = {side: item}
    return dict(
        frame_index=frame_index,
        hands=hands,
        evidence="dual_label" if len(hands) == 2 else "single_label_pending_other_hand",
        label_count=len(hands),
        score=float(min(c["selection_score"] for c in hands.values())),
        other_hand_visibility_verified=False,
    )


def rank_frames(candidates, fps=None, limit=100):
    """Rank whole frames: bilateral labels first, then the weaker label's quality."""
    return sorted(candidates, key=lambda c: (-c["label_count"], -c["score"], c["frame_index"]))[
        :limit
    ]
