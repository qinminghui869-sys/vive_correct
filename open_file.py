#!/usr/bin/env python3
"""Open a specified episode, Rerun recording, or media file.

    python3 open_file.py <path>
    python3 open_file.py <path> --dry-run
    python3 open_file.py <episode> --tool stickers
    python3 open_file.py <episode-or-file-inside-it> --tool wrists

An episode directory opens the wrist alignment tool. The built-in L3
recording keeps its saved outputs. Any other episode writes a separate
JSON and RRD under ego_data/outputs, and starts on a frame whose head
and both wrist poses match the depth timestamp. A .rrd opens in Rerun.
Images and videos open in the system viewer.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EGO = ROOT / "ego_data"
DEFAULT_DATASET = EGO / "L3-001-2026-10-27-13-47-37"
POSE_NAMES = ("head_pose", "left_wrist_pose", "right_wrist_pose")
DEPTH_KEY = "observation.images.depth_headband_left_rectified"
MEDIA = {".mp4", ".mkv", ".mov", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def episode_root(path: Path) -> Path | None:
    from vive_correct.data.layout import is_indexed_episode

    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if is_indexed_episode(candidate):
            return candidate
        data = candidate / "data/chunk-000/episode_000000"
        videos = candidate / "videos/chunk-000"
        siblings = data.parent.glob("episode_*") if data.parent.is_dir() else ()
        if data.is_dir() and videos.is_dir() and sum(1 for _ in siblings) <= 1:
            return candidate
    return None


def is_default_dataset(path: Path) -> bool:
    return path.resolve() == DEFAULT_DATASET.resolve()


def output_slug(episode: Path) -> str:
    parent = episode.parent.name
    if parent and parent not in {".", "ego_data", ROOT.name}:
        return f"{parent}_{episode.name}"
    return episode.name


def synced_initial_frame(episode: Path, prefer: int = 300, tolerance: float = 0.005) -> int:
    """First depth frame near `prefer` with head and both wrists within tolerance."""
    sys.path.insert(0, str(EGO))
    import numpy as np
    import pyarrow.parquet as pq
    from render_rerun import nearest, table

    from vive_correct.data.layout import data_directory

    data = data_directory(episode)
    times = pq.read_table(data / f"{DEPTH_KEY}.parquet")["timestamp"].to_numpy()
    streams = [table(data / f"observation.{name}.parquet") for name in POSE_NAMES]
    count = len(times)

    def ready(index: int) -> bool:
        stamp = float(times[index])
        return all(np.isfinite(nearest(stream, stamp, tolerance)).all() for stream in streams)

    order = []
    if 0 <= prefer < count:
        order.append(prefer)
    for delta in range(1, count):
        for index in (prefer - delta, prefer + delta):
            if 0 <= index < count:
                order.append(index)
    for index in order:
        if ready(index):
            return index
    raise SystemExit(f"{episode} 没有与深度时间戳对齐的头/腕姿态帧")


def command_for(
    path: Path, tool: str = "auto", initial_frame: int | None = None, extra: list[str] | None = None
) -> list[str]:
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")
    path = path.resolve()
    extra = list(extra or [])
    episode = episode_root(path)
    kind = tool
    if kind == "auto":
        if path.suffix.lower() == ".rrd":
            kind = "rerun"
        elif path.is_dir() and episode == path:
            kind = "wrists"
        elif path.suffix.lower() in MEDIA:
            kind = "view"
        elif episode is not None:
            raise SystemExit(
                f"{path} 在 episode 内。打开整个目录，或加上 --tool wrists / --tool stickers。\n目录：{episode}"
            )
        else:
            raise SystemExit(f"无法判断如何打开：{path}\n可打开 episode 目录、.rrd、图片或视频。")

    if kind == "rerun":
        if path.suffix.lower() != ".rrd":
            raise SystemExit(f"Rerun 需要 .rrd 文件：{path}")
        rerun = shutil.which("rerun")
        if rerun is None:
            raise SystemExit("未找到 rerun，请先安装并加入 PATH")
        return [rerun, str(path.resolve()), *extra]

    if kind == "view":
        opener = shutil.which("xdg-open")
        if opener is None:
            raise SystemExit("未找到 xdg-open，无法打开该媒体文件")
        return [opener, str(path.resolve()), *extra]

    if episode is None:
        raise SystemExit(f"不是 L3 episode，也没有包含在某个 episode 里：{path}")

    if kind == "stickers":
        return [
            sys.executable,
            str(ROOT / "vive_correct/estimate/sticker_pipeline.py"),
            "--episode",
            str(episode),
            *extra,
        ]

    if kind == "manus":
        return [sys.executable, str(ROOT / "open_manus.py"), str(episode), *extra]

    if kind != "wrists":
        raise SystemExit(f"未知打开方式：{kind}")

    command = [sys.executable, str(EGO / "align_l3_wrists.py"), "--dataset", str(episode)]
    if not is_default_dataset(episode):
        slug = output_slug(episode)
        command += [
            "--output",
            str(EGO / "outputs" / f"wrist_alignment_{slug}.json"),
            "--rerun-output",
            str(EGO / "outputs" / f"{slug}_wrists_aligned.rrd"),
        ]
        if initial_frame is None and "--initial-frame" not in extra:
            initial_frame = synced_initial_frame(episode)
    if initial_frame is not None:
        command += ["--initial-frame", str(initial_frame)]
    return command + extra


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="episode 目录、.rrd、图片或视频")
    parser.add_argument(
        "--tool", choices=("auto", "wrists", "manus", "stickers", "rerun", "view"), default="auto"
    )
    parser.add_argument(
        "--initial-frame", type=int, help="腕部工具的起始预览帧；其他 episode 默认选可同步帧"
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    args, extra = parser.parse_known_args(argv)
    command = command_for(args.path, args.tool, args.initial_frame, extra)
    print(shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
