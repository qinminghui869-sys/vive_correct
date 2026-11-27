"""Open completed recordings without running repair or export again."""

import argparse
import ctypes
import errno
import json
from pathlib import Path
import shutil
import subprocess
import sys


def file_watchers_exhausted():
    """Rerun's Linux file reader needs an inotify instance, even for saved RRDs."""
    if not sys.platform.startswith("linux"):
        return False
    import os

    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC)
    if descriptor < 0:
        return ctypes.get_errno() in (errno.EMFILE, errno.ENFILE, errno.ENOSPC)
    os.close(descriptor)
    return False


def recordings_at(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if path.is_file() and path.suffix == ".rrd":
        recordings = [path]
    else:
        root = path.parent if path.is_file() else path
        manifest = next(
            (p for p in (root / "rerun/manifest.json", root / "manifest.json") if p.is_file()), None
        )
        if manifest:
            entries = json.loads(manifest.read_text())["entries"]
            recordings = [Path(e["recording"]) for e in entries if e.get("status") == "complete"]
            recordings = [p if p.is_absolute() else manifest.parent / p for p in recordings]
        else:
            recordings = list(root.glob("*.rrd"))
    recordings = sorted(p.resolve() for p in recordings if ".partial." not in p.name)
    if not recordings or any(not p.is_file() or p.stat().st_size == 0 for p in recordings):
        raise ValueError(
            "No completed Rerun recording found; check rerun/manifest.json and render.log"
        )
    return recordings


def launch(recordings, *, log_path=None):
    executable = shutil.which("rerun")
    if not executable:
        raise ValueError("Install Rerun: python3 -m pip install -r ego_data/requirements.txt")
    recordings = [Path(p).resolve() for p in recordings]
    log_path = Path(log_path) if log_path else recordings[0].parent / "viewer.log"
    command = [executable, "--bind", "127.0.0.1", "--port", "0"]
    if sys.platform.startswith("linux"):
        command += ["--renderer", "vulkan"]
    if file_watchers_exhausted():
        print("Linux file watchers exhausted; loading RRD via a local HTTP server.", flush=True)
        command = [
            sys.executable,
            "-m",
            "vive_correct.export.viewer_http",
            json.dumps(command),
            *map(str, recordings),
        ]
    else:
        command += [str(p) for p in recordings]
    with log_path.open("a") as log:
        viewer = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            code = viewer.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print(
                f"Rerun started (PID {viewer.pid}); loading {len(recordings)} recording(s). Log: {log_path}",
                flush=True,
            )
            return
    raise ValueError(f"Rerun exited during startup (code {code}); see {log_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="Completed .rrd, output directory, or review_index.html"
    )
    args = parser.parse_args(argv)
    try:
        launch(recordings_at(args.path))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0
