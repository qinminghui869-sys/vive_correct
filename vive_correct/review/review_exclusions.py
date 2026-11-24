"""Persistent episode exclusions; the root JSON is the authoritative decision log."""

from pathlib import Path

from vive_correct.review.review_io import read_json

FILENAME = "rejected_episodes.json"


def exclusion(path, episode):
    path = Path(path)
    if not path.exists():
        return None
    episode = str(Path(episode).resolve())
    return next((row for row in read_json(path)["entries"] if row["episode"] == episode), None)


def skipped(state, index):
    return dict(
        index=index,
        episode=state["episode"],
        status="excluded",
        reason=state["pending_reason"],
    )
