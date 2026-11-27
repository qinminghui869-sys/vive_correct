"""Bounded process-local caches; file identity changes invalidate cached inputs."""

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
import threading


def file_stamp(path):
    path = Path(path)
    try:
        s = path.stat()
        return (str(path.resolve()), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    except FileNotFoundError:
        return (str(path.absolute()), None)


def source_stamp(episode):
    from vive_correct.data.layout import dataset_root, episode_stem, is_indexed_episode

    episode = Path(episode)
    root, stem = dataset_root(episode), episode_stem(episode)
    return tuple(
        file_stamp(p)
        for folder in ("data", "videos", "calibrations", "annotation", "meta")
        for p in sorted((root / folder).rglob("*"))
        if p.is_file()
        and not p.name.endswith((".lock", ".restore"))
        and (
            not is_indexed_episode(episode)
            or folder in ("calibrations", "meta")
            or p.stem == stem
            or stem in p.parts
        )
    )


def byte_size(value):
    if hasattr(value, "nbytes"):
        return value.nbytes + sys.getsizeof(value)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(byte_size(k) + byte_size(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(map(byte_size, value))
    if is_dataclass(value):
        return sys.getsizeof(value) + sum(byte_size(getattr(value, f.name)) for f in fields(value))
    return sys.getsizeof(value)


class MemoryCache:
    def __init__(self, limit=512 * 1024 * 1024):
        self.limit = limit
        self.size = 0
        self.values = OrderedDict()
        self.lock = threading.Lock()
        self.pending = {}
        self.generation = 0

    def get(self, key, loader):
        with self.lock:
            if key in self.values:
                value, _ = self.values[key]
                self.values.move_to_end(key)
                hit = True
            else:
                hit = False
                pending_key = (self.generation, key)
                future = self.pending.get(pending_key)
                owner = future is None
                if owner:
                    future = self.pending[pending_key] = Future()
        if hit:
            return deepcopy(value)
        if not owner:
            return deepcopy(future.result())
        try:
            value = loader()
            stored = deepcopy(value)
            size = byte_size(stored) + byte_size(key)
            with self.lock:
                if size <= self.limit and pending_key[0] == self.generation:
                    while self.values and self.size + size > self.limit:
                        self.size -= self.values.popitem(last=False)[1][1]
                    self.values[key] = (stored, size)
                    self.size += size
            future.set_result(stored)
            return value
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.pending.pop(pending_key, None)

    def resize(self, limit):
        if limit <= 0:
            raise ValueError("缓存大小必须为正数")
        with self.lock:
            self.limit = limit
            while self.values and self.size > limit:
                self.size -= self.values.popitem(last=False)[1][1]

    def clear(self):
        with self.lock:
            self.values.clear()
            self.size = 0
            self.generation += 1


CACHE = MemoryCache()


def cached_file(kind, path, loader):
    stamp = file_stamp(path)
    value = CACHE.get((kind, stamp), loader)
    if file_stamp(path) != stamp:
        raise ValueError(f"读取期间输入发生变化：{path}")
    return value
