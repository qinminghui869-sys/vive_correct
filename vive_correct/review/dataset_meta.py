"""Refresh dataset stats and byte size after an in-place rebuild.

Only the six replaced features are recomputed. The reduction matches the numbers
already stored for that feature, so unchanged columns keep their original stats.
"""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vive_correct.data.layout import data_directory, dataset_root, is_indexed_episode
from vive_correct.data.source_data import FILES, copy_file

STATS_NAME = "episodes_stats.jsonl"
README_NAME = "readme.json"
_SKIP = {"timestamp", "episode_index", "frame_index", "index"}


def episode_index(episode):
    episode = Path(episode)
    if is_indexed_episode(episode):
        return int(episode.name.removeprefix("episode_"))
    return 0


def meta_directory(episode):
    return dataset_root(episode) / "meta"


@contextmanager
def dataset_lock(episode):
    """Serialize metadata edits for one dataset without leaving a lock in the dataset."""
    directory = Path(tempfile.gettempdir()) / f"vive-rebuild-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    name = hashlib.sha256(b"meta" + str(dataset_root(episode).resolve()).encode()).hexdigest()
    with (directory / name).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def snapshot_dataset_meta(episode, transaction):
    """Copy the two metadata files before either one is replaced."""
    root = meta_directory(episode)
    saved = []
    for name in (STATS_NAME, README_NAME):
        source = root / name
        if not source.is_file():
            continue
        destination = Path(transaction) / "meta" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_file(source, destination)
        saved.append(name)
    return saved


def restore_dataset_meta(episode, transaction, names):
    root = meta_directory(episode)
    for name in names:
        source = Path(transaction) / "meta" / name
        target = root / name
        temporary = target.with_name(target.name + ".restore")
        copy_file(source, temporary)
        os.replace(temporary, target)
    if names:
        from vive_correct.review.review_io import sync_directory

        sync_directory(root)


def refresh_dataset_meta(episode, previous_directory):
    """Recompute one episode's six feature stats and the dataset byte total."""
    root = meta_directory(episode)
    summary = {}
    stats_path = root / STATS_NAME
    if stats_path.is_file():
        _refresh_stats(episode, stats_path, Path(previous_directory))
        summary["episodes_stats"] = str(stats_path)
    readme_path = root / README_NAME
    if readme_path.is_file():
        _refresh_readme(episode, readme_path, Path(previous_directory))
        summary["readme"] = str(readme_path)
    if summary:
        from vive_correct.review.review_io import sync_directory

        sync_directory(root)
    return summary


def _info(path):
    info_path = path.with_name("info.json")
    if not info_path.is_file():
        return None
    return json.loads(info_path.read_text(encoding="utf-8"))


def _feature_columns(table, key, info):
    names = None
    features = (info or {}).get("features", {})
    described = features.get(key, {}).get("names") if isinstance(features, dict) else None
    if (
        isinstance(described, list)
        and described
        and all(isinstance(name, str) for name in described)
    ):
        names = described
    if names is None:
        names = [
            name
            for name in table.column_names
            if name not in _SKIP and pa.types.is_floating(table.schema.field(name).type)
        ]
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise ValueError(f"{key}: 统计列缺失：{missing[0]}")
    return names


def _values(table, names):
    if not names:
        return np.empty((len(table), 0))
    return np.column_stack([table[name].to_numpy(zero_copy_only=False) for name in names])


def _reductions(values, *, wide):
    data = values.astype(np.float64) if wide else values
    if data.size and not np.isfinite(data).all():
        data = np.asarray(data, dtype=np.float64)
        return (
            np.nanmin(data, axis=0),
            np.nanmax(data, axis=0),
            np.nanmean(data, axis=0),
            np.nanstd(data, axis=0),
        )
    return np.min(data, axis=0), np.max(data, axis=0), np.mean(data, axis=0), np.std(data, axis=0)


def _error(got, reference):
    worst = 0.0
    for name, values in zip(("min", "max", "mean", "std"), got):
        if name not in reference:
            return np.inf
        expected = np.asarray(reference[name], dtype=np.float64)
        actual = np.asarray(values, dtype=np.float64)
        if expected.shape != actual.shape:
            return np.inf
        worst = max(worst, float(np.max(np.abs(actual - expected))))
    return worst


def _numbers(values):
    result = [float(value) for value in np.asarray(values).reshape(-1)]
    if not np.isfinite(result).all():
        raise ValueError("统计结果不是有限数值")
    return result


def _use_wide(previous, reference):
    """Pick float64 accumulation only when it reproduces the stored stats more closely."""
    native = _error(_reductions(previous, wide=False), reference)
    wide = _error(_reductions(previous, wide=True), reference)
    return wide < native


def _block(values, *, wide):
    reduced = _reductions(values, wide=wide)
    return {
        "min": _numbers(reduced[0]),
        "max": _numbers(reduced[1]),
        "mean": _numbers(reduced[2]),
        "std": _numbers(reduced[3]),
        "count": [int(len(values))],
    }


def _refresh_stats(episode, path, previous_directory):
    info = _info(path)
    text = path.read_text(encoding="utf-8")
    index = episode_index(episode)
    lines, found = [], False
    for line in text.splitlines():
        row = json.loads(line)
        if row.get("episode_index") != index:
            lines.append(line)
            continue
        found = True
        stats = row.setdefault("stats", {})
        for name in FILES:
            key = name.removesuffix(".parquet")
            if key not in stats:
                raise ValueError(f"episodes_stats.jsonl 缺少 {key}")
            previous = pq.read_table(previous_directory / name)
            current = pq.read_table(data_directory(episode) / name)
            columns = _feature_columns(current, key, info)
            wide = _use_wide(_values(previous, columns), stats[key])
            stats[key] = _block(_values(current, columns), wide=wide)
        lines.append(json.dumps(row, ensure_ascii=False))
    if not found:
        raise ValueError(f"episodes_stats.jsonl 没有 episode_index {index}")
    _publish(path, text, lines)


def _refresh_readme(episode, path, previous_directory):
    text = path.read_text(encoding="utf-8")
    readme = json.loads(text)
    if "size_bytes" not in readme:
        return
    delta = 0
    current_dir = data_directory(episode)
    for name in FILES:
        delta += (current_dir / name).stat().st_size - (previous_directory / name).stat().st_size
    readme["size_bytes"] = int(readme["size_bytes"]) + delta
    indent = 2 if text.startswith("{\n") else None
    published = json.dumps(readme, ensure_ascii=False, indent=indent)
    if text.endswith("\n"):
        published += "\n"
    _publish(path, text, published)


def _publish(path, original, content):
    if isinstance(content, list):
        content = "\n".join(content)
        if original.endswith("\n"):
            content += "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
