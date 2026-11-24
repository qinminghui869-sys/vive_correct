"""Review artifact I/O and manifest updates; no geometry or workflow dependencies."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from vive_correct.batch.report_io import digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value, *, durable=False):
    """Publish complete JSON; durable writes also sync the file and parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(output, *, name=".human_review.lock", blocking=False):
    """Hold an episode lock for work; hold the root lock only while updating indices."""
    with (Path(output) / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise ValueError("正在保存或导出，请稍后重试") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def export_code_signature():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "vive_correct").rglob("*.py")) + sorted((root / "ego_data").glob("*.py"))
    return hashlib.sha256("".join(digest(path) for path in files).encode()).hexdigest()


def merge_rows(path, episodes, rows, **metadata):
    """Replace only affected episodes. Caller must hold the output root lock."""
    previous = read_json(path) if path.exists() else {}
    remaining = [row for row in previous.get("entries", []) if row.get("episode") not in episodes]
    previous.update(
        metadata, entries=sorted(remaining + rows, key=lambda row: row.get("episode", ""))
    )
    if path.name == "repair_manifest.json":
        states = {row["status"] for row in previous["entries"]}
        previous["status"] = (
            "running"
            if states & {"running", "prepared", "rendering"}
            else "complete"
            if states == {"complete"}
            else "incomplete"
        )
    atomic_json(path, previous)
