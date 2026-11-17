"""Cross-process reservations, including queued jobs; nested calls share ownership."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import threading

LOCAL = threading.local()


def reserve(folder):
    path = Path(folder).resolve() / ".review_task.lock"
    stream = path.open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise ValueError("这条数据正在处理；可继续审核其他数据") from error
    return stream


@contextmanager
def adopt(stream):
    owners = getattr(LOCAL, "owners", set())
    LOCAL.owners = owners
    owners.add(stream.name)
    try:
        yield
    finally:
        owners.remove(stream.name)
        stream.close()


@contextmanager
def operation_lock(folder):
    path = str(Path(folder).resolve() / ".review_task.lock")
    if path in getattr(LOCAL, "owners", set()):
        yield
    else:
        with adopt(reserve(folder)):
            yield
