"""Original six-file baseline and recoverable in-place replacements."""

import fcntl
import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from vive_correct.batch.report_io import digest
from vive_correct.review_io import atomic_json, read_json, sync_directory

DATA = Path("data/chunk-000/episode_000000")
FILES = (
    "observation.left_wrist_pose.parquet",
    "observation.right_wrist_pose.parquet",
    "action.parquet",
    "observation.state.parquet",
    "observation.hand_left_points.parquet",
    "observation.hand_right_points.parquet",
)


def durable_json(path, value):
    """Transaction records must survive process and machine interruptions."""
    atomic_json(path, value, durable=True)


def copy_file(source, destination, *, exclusive=False):
    with Path(source).open("rb") as src, Path(destination).open("xb" if exclusive else "wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    shutil.copystat(source, destination)


@contextmanager
def source_guard(episode, *, exclusive=False):
    directory = Path(tempfile.gettempdir()) / f"vive-rebuild-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    name = hashlib.sha256(str(Path(episode).resolve()).encode()).hexdigest()
    with (directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("这条源数据正在读取或重建，请稍后重试") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def baseline_read(episode):
    recover(episode)
    with source_guard(episode):
        yield


@contextmanager
def source_lock(episode):
    with source_guard(episode, exclusive=True):
        with _source_lock(episode) as directory:
            yield directory


@contextmanager
def _source_lock(episode):
    directory = Path(episode) / ".vive_rebuild"
    directory.mkdir(exist_ok=True)
    with (directory / "lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("这条源数据正在重建，请稍后重试") from error
        try:
            yield directory
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def recover_locked(episode):
    directory = Path(episode) / ".vive_rebuild"
    journal_path = directory / "transaction.json"
    if not journal_path.exists():
        return
    job = read_json(journal_path)
    if job["status"] != "applying":
        return
    transaction = directory / "transactions" / job["id"]
    # Never discard an unrelated edit made after interruption.
    for name, row in job["files"].items():
        current = Path(episode) / DATA / name
        if current.exists() and digest(current) not in (row["before"], row["after"]):
            raise ValueError(f"中断恢复冲突：{current} 已被外部修改")
        if digest(transaction / "previous" / name) != row["before"]:
            raise ValueError(f"中断恢复副本损坏：{name}")
    for name in FILES:
        target = Path(episode) / DATA / name
        temporary = target.with_name(target.name + ".restore")
        copy_file(transaction / "previous" / name, temporary)
        os.replace(temporary, target)
    state_path = directory / "state.json"
    if job["previous_state"] is None:
        state_path.unlink(missing_ok=True)
        for name in job["created_backups"]:
            backup = Path(episode) / DATA / (name + ".bak")
            if backup.exists() and os.path.samefile(backup, transaction / "previous" / name):
                backup.unlink()
    else:
        durable_json(state_path, job["previous_state"])
    sync_directory(Path(episode) / DATA)
    durable_json(journal_path, dict(job, status="rolled_back"))


def recover(episode):
    path = Path(episode) / ".vive_rebuild/transaction.json"
    if path.exists() and read_json(path)["status"] == "applying":
        with source_lock(episode):
            recover_locked(episode)


def original_files(episode):
    episode = Path(episode)
    recover(episode)
    state_path = episode / ".vive_rebuild/state.json"
    state = read_json(state_path) if state_path.exists() else None
    result = {}
    for name in FILES:
        current = episode / DATA / name
        backup = current.with_name(name + ".bak")
        if state:
            row = state["files"][name]
            if not backup.exists() or digest(backup) != row["baseline_sha256"]:
                raise ValueError(f"原始备份缺失或损坏：{backup}")
            if not current.exists() or digest(current) != row["current_sha256"]:
                raise ValueError(f"源文件被外部修改：{current}")
            result[name] = backup
        else:
            if backup.exists():
                raise ValueError(f"存在未登记备份，不会覆盖：{backup}")
            result[name] = current
    return result


def original_path(path):
    path = Path(path)
    if (
        path.name in FILES
        and path.parent.name == "episode_000000"
        and path.parent.parent.name == "chunk-000"
        and path.parent.parent.parent.name == "data"
    ):
        return original_files(path.parents[3])[path.name]
    return path


def apply_files(episode, files, baseline_hashes, bundle):
    """Stage all six files before touching source; one journal protects the whole set."""
    episode = Path(episode)
    with source_lock(episode) as directory:
        recover_locked(episode)
        originals = original_files(episode)
        if {n: digest(p) for n, p in originals.items()} != baseline_hashes:
            raise ValueError("原始输入已变化，请重新审核")
        state_path = directory / "state.json"
        previous = read_json(state_path) if state_path.exists() else None
        identifier = uuid.uuid4().hex
        transaction = directory / "transactions" / identifier
        for sub in ("new", "previous"):
            (transaction / sub).mkdir(parents=True)
        rows, target = {}, {}
        for name in FILES:
            current = episode / DATA / name
            row = files[name]
            if digest(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"修正产物已变化：{name}")
            copy_file(current, transaction / "previous" / name)
            copy_file(Path(row["path"]), transaction / "new" / name)
            rows[name] = dict(before=digest(current), after=row["sha256"])
            target[name] = dict(baseline_sha256=baseline_hashes[name], current_sha256=row["sha256"])
        # Check again after copying, before publishing a journal or backups.
        if any(
            digest(episode / DATA / n) != rows[n]["before"]
            or digest(transaction / "previous" / n) != rows[n]["before"]
            for n in FILES
        ):
            raise ValueError("准备覆盖期间源文件发生变化")
        job = dict(
            id=identifier,
            status="applying",
            files=rows,
            previous_state=previous,
            created_backups=list(FILES) if previous is None else [],
            bundle=str(bundle),
        )
        journal = directory / "transaction.json"
        durable_json(journal, job)
        try:
            if previous is None:
                for name in FILES:
                    backup = episode / DATA / (name + ".bak")
                    # Same-filesystem link publishes a complete backup without clobbering
                    # another process's file; inode identity also proves rollback ownership.
                    os.link(transaction / "previous" / name, backup)
                    if digest(backup) != baseline_hashes[name]:
                        raise ValueError(f"备份校验失败：{name}")
                sync_directory(episode / DATA)
            for name in FILES:
                if digest(episode / DATA / name) != rows[name]["before"]:
                    raise ValueError(f"覆盖期间源文件发生变化：{name}")
                os.replace(transaction / "new" / name, episode / DATA / name)
            sync_directory(episode / DATA)
            if any(digest(episode / DATA / n) != rows[n]["after"] for n in FILES):
                raise ValueError("覆盖结果校验失败")
            durable_json(
                state_path,
                dict(version=1, files=target, bundle=str(bundle), transaction=identifier),
            )
            durable_json(journal, dict(job, status="committed"))
        except Exception:
            recover_locked(episode)
            raise
        return dict(
            status="complete",
            transaction=identifier,
            backups=[str(episode / DATA / (n + ".bak")) for n in FILES],
        )
