"""Original six-file baseline and recoverable in-place replacements.

The dataset directory only receives the six replaced Parquets. Baseline copies and
the crash journal live under the first ``--out`` that performed an in-place rebuild.
"""

import errno
import fcntl
import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from vive_correct.batch.report_io import digest
from vive_correct.data.layout import data_directory, is_indexed_episode
from vive_correct.review.review_io import atomic_json, read_json, sync_directory

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


def publish_file(source, target):
    """Publish atomically even when the output journal and dataset are on different mounts."""
    try:
        os.replace(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        temporary = target.with_name(target.name + ".restore")
        try:
            copy_file(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def index_path():
    override = os.environ.get("VIVE_REBUILD_INDEX")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "outputs" / "rebuild_index.json"


def episode_key(episode):
    return hashlib.sha256(str(Path(episode).resolve()).encode()).hexdigest()


def _read_index():
    path = index_path()
    if not path.is_file():
        return {}
    return read_json(path).get("episodes", {})


def _write_index(episode, directory):
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            current = read_json(path) if path.is_file() else {"version": 1, "episodes": {}}
            current.setdefault("episodes", {})[str(Path(episode).resolve())] = str(directory)
            durable_json(path, current)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def lookup_journal(episode):
    """Return the output-side journal for this episode, if one was registered."""
    location = _read_index().get(str(Path(episode).resolve()))
    if not location:
        return None
    path = Path(location)
    if not path.is_dir():
        raise ValueError(f"重建记录缺失：{path}")
    return path


def journal_directory(output, episode):
    """Create the journal on first in-place rebuild; later runs keep that same directory."""
    existing = lookup_journal(episode)
    if existing is not None:
        return existing
    directory = Path(output).resolve() / "rebuild_state" / episode_key(episode)
    directory.mkdir(parents=True, exist_ok=True)
    _write_index(episode, directory)
    return directory


def rebuild_journal(episode):
    found = lookup_journal(episode)
    if found is None:
        raise ValueError("没有这条数据的重建记录")
    return found


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
def _journal_lock(directory):
    with (directory / "lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("这条源数据正在重建，请稍后重试") from error
        try:
            yield directory
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def source_lock(episode, output=None):
    """Lock one episode without creating a journal directory beside its data."""
    with source_guard(episode, exclusive=True):
        directory = (
            journal_directory(output, episode) if output is not None else lookup_journal(episode)
        )
        if directory is None:
            yield None
            return
        with _journal_lock(directory):
            yield directory


def recover_locked(episode):
    directory = lookup_journal(episode)
    if directory is None:
        return
    journal_path = directory / "transaction.json"
    if not journal_path.exists():
        return
    job = read_json(journal_path)
    if job["status"] != "applying":
        return
    transaction = directory / "transactions" / job["id"]
    # Never discard an unrelated edit made after interruption.
    for name, row in job["files"].items():
        current = data_directory(episode) / name
        if current.exists() and digest(current) not in (row["before"], row["after"]):
            raise ValueError(f"中断恢复冲突：{current} 已被外部修改")
        if digest(transaction / "previous" / name) != row["before"]:
            raise ValueError(f"中断恢复副本损坏：{name}")
    for name in FILES:
        target = data_directory(episode) / name
        temporary = target.with_name(target.name + ".restore")
        copy_file(transaction / "previous" / name, temporary)
        os.replace(temporary, target)
    state_path = directory / "state.json"
    if job["previous_state"] is None:
        state_path.unlink(missing_ok=True)
        for name in job["created_baselines"]:
            baseline = directory / "baseline" / name
            previous = transaction / "previous" / name
            if baseline.exists() and previous.exists() and os.path.samefile(baseline, previous):
                baseline.unlink()
    else:
        durable_json(state_path, job["previous_state"])
    sync_directory(data_directory(episode))
    if (directory / "baseline").is_dir():
        sync_directory(directory / "baseline")
    if job.get("meta_files"):
        from vive_correct.review.dataset_meta import dataset_lock, restore_dataset_meta

        with dataset_lock(episode):
            restore_dataset_meta(episode, transaction, job["meta_files"])
    durable_json(journal_path, dict(job, status="rolled_back"))
    shutil.rmtree(transaction, ignore_errors=True)


def recover(episode):
    directory = lookup_journal(episode)
    if directory is None:
        return
    path = directory / "transaction.json"
    if path.exists() and read_json(path)["status"] == "applying":
        with source_lock(episode):
            recover_locked(episode)


def original_files(episode):
    episode = Path(episode)
    recover(episode)
    directory = lookup_journal(episode)
    state_path = directory / "state.json" if directory is not None else None
    state = read_json(state_path) if state_path is not None and state_path.exists() else None
    result = {}
    for name in FILES:
        current = data_directory(episode) / name
        backup = current.with_name(name + ".bak")
        if state:
            row = state["files"][name]
            baseline = directory / "baseline" / name
            if not baseline.exists() or digest(baseline) != row["baseline_sha256"]:
                raise ValueError(f"原始基线缺失或损坏：{baseline}")
            if not current.exists() or digest(current) != row["current_sha256"]:
                raise ValueError(f"源文件被外部修改：{current}")
            result[name] = baseline
        else:
            if backup.exists():
                raise ValueError(f"存在未登记备份，不会覆盖：{backup}")
            result[name] = current
    return result


def original_path(path):
    path = Path(path)
    parent = path.parent
    if (
        path.name in FILES
        and parent.name.startswith("episode_")
        and parent.parent.name == "chunk-000"
        and parent.parent.parent.name == "data"
    ):
        episode = parent if is_indexed_episode(parent) else path.parents[3]
        return original_files(episode)[path.name]
    return path


def register_existing_journal(episode, directory, baseline_hashes):
    """Register relocated process records without replacing an existing history."""
    directory = Path(directory).resolve()
    with source_guard(episode, exclusive=True), _journal_lock(directory):
        existing = lookup_journal(episode)
        if existing is not None and existing.resolve() != directory:
            raise ValueError("当前索引指向另一份重建记录，请先核对历史")
        state = read_json(directory / "state.json")
        if read_json(directory / "transaction.json")["status"] not in ("committed", "rolled_back"):
            raise ValueError("存在未完成事务，请先恢复")
        for name in FILES:
            row = state["files"][name]
            if (
                row["baseline_sha256"] != baseline_hashes[name]
                or digest(directory / "baseline" / name) != baseline_hashes[name]
            ):
                raise ValueError(f"原始基线缺失或损坏：{name}")
            if digest(data_directory(episode) / name) != row["current_sha256"]:
                raise ValueError(f"源文件被外部修改：{name}")
        _write_index(episode, directory)


def apply_files(episode, files, baseline_hashes, bundle, output, *, expected_current=None):
    """Replace six source files. Baseline and journal stay in the output directory."""
    episode = Path(episode)
    with source_lock(episode, output) as directory:
        recover_locked(episode)
        originals = original_files(episode)
        if (
            expected_current is not None
            and {n: digest(data_directory(episode) / n) for n in FILES} != expected_current
        ):
            raise ValueError("准备后源文件发生变化")
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
            current = data_directory(episode) / name
            row = files[name]
            if digest(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"修正产物已变化：{name}")
            copy_file(current, transaction / "previous" / name)
            copy_file(Path(row["path"]), transaction / "new" / name)
            rows[name] = dict(before=digest(current), after=row["sha256"])
            target[name] = dict(baseline_sha256=baseline_hashes[name], current_sha256=row["sha256"])
        if any(
            digest(data_directory(episode) / n) != rows[n]["before"]
            or digest(transaction / "previous" / n) != rows[n]["before"]
            for n in FILES
        ):
            raise ValueError("准备覆盖期间源文件发生变化")
        job = dict(
            id=identifier,
            status="applying",
            files=rows,
            previous_state=previous,
            created_baselines=list(FILES) if previous is None else [],
            bundle=str(bundle),
        )
        journal = directory / "transaction.json"
        durable_json(journal, job)
        try:
            if previous is None:
                baseline_dir = directory / "baseline"
                baseline_dir.mkdir(exist_ok=True)
                for name in FILES:
                    baseline = baseline_dir / name
                    # Same-filesystem link publishes the original without clobbering
                    # another process's file; inode identity also proves rollback ownership.
                    os.link(transaction / "previous" / name, baseline)
                    if digest(baseline) != baseline_hashes[name]:
                        raise ValueError(f"原始基线校验失败：{name}")
                sync_directory(baseline_dir)
            for name in FILES:
                if digest(data_directory(episode) / name) != rows[name]["before"]:
                    raise ValueError(f"覆盖期间源文件发生变化：{name}")
                publish_file(transaction / "new" / name, data_directory(episode) / name)
            sync_directory(data_directory(episode))
            if any(digest(data_directory(episode) / n) != rows[n]["after"] for n in FILES):
                raise ValueError("覆盖结果校验失败")
            from vive_correct.review.dataset_meta import (
                dataset_lock,
                refresh_dataset_meta,
                snapshot_dataset_meta,
            )

            with dataset_lock(episode):
                job["meta_files"] = snapshot_dataset_meta(episode, transaction)
                durable_json(journal, job)
                meta = refresh_dataset_meta(episode, transaction / "previous")
            durable_json(
                state_path,
                dict(version=1, files=target, bundle=str(bundle), transaction=identifier),
            )
            durable_json(journal, dict(job, status="committed"))
            shutil.rmtree(transaction, ignore_errors=True)
        except Exception:
            recover_locked(episode)
            raise
        return dict(
            status="complete",
            transaction=identifier,
            baseline=[str(directory / "baseline" / name) for name in FILES],
            meta=meta,
        )
