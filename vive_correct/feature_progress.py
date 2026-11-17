"""Durable progress shared by the independent scanner and review server."""

import fcntl
from pathlib import Path

from vive_correct.review_io import read_json


def scan_running(output):
    lock = Path(output) / ".feature_scan.lock"
    if not lock.exists():
        return False
    with lock.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def scan_progress(output):
    output = Path(output)
    path = output / "feature_progress.json"
    if not path.exists():
        return dict(status="waiting", total=0, entries=[])
    progress = read_json(path)
    running = scan_running(output)
    for row in progress["entries"]:
        if row["status"] == "queued":
            status = output / row["folder"] / "scan_status.json"
            if status.is_file():
                saved = read_json(status)
                if saved.get("run_id") == progress["run_id"]:
                    row.update({k: v for k, v in saved.items() if k != "run_id"})
        if not running and row["status"] in ("queued", "scanning"):
            row.update(status="interrupted", error="选帧已中断，可使用原命令续跑")
    if not running and progress["status"] == "running":
        progress["status"] = "interrupted"
    progress["total"] = len(progress["entries"])
    return progress
