#!/usr/bin/env python3
"""Recompute depth origins from saved reviews and write six source Parquets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq

from ego_data.vive_origin_alignment import build_alignment
from vive_correct.batch.report_io import digest
from vive_correct.data import source_data
from vive_correct.data.layout import data_directory, dataset_root, episode_stem, timestamp_file
from vive_correct.estimate.geometry import VIDEO_KEY
from vive_correct.review.axis_calibration import prepare_pose
from vive_correct.review.human_review import checked_pose
from vive_correct.review.parquet_correction import translate, verify_table
from vive_correct.review.review_io import atomic_json, export_code_signature, output_lock, read_json
from vive_correct.runtime.runtime_cache import source_stamp


def check_journal(episode, directory, hashes):
    """Read-only validation: preview must never trigger transaction recovery."""
    state = read_json(directory / "state.json")
    if read_json(directory / "transaction.json")["status"] not in ("committed", "rolled_back"):
        raise ValueError(f"存在未完成事务，请先恢复：{directory}")
    current = {}
    for name in source_data.FILES:
        record = state["files"][name]
        if (
            record["baseline_sha256"] != hashes[name]
            or digest(directory / "baseline" / name) != hashes[name]
        ):
            raise ValueError(f"原始基线缺失或损坏：{name}")
        current[name] = digest(data_directory(episode) / name)
        if current[name] != record["current_sha256"]:
            raise ValueError(f"源文件被外部修改：{name}")
    return current


def discover(dataset, process, selected):
    directories = sorted((dataset / "data/chunk-000").glob("episode_*"))
    if not directories:
        raise ValueError("数据集没有 episode")
    episodes = (
        directories
        if len(directories) > 1 or directories[0].name != "episode_000000"
        else [dataset]
    )
    wanted = set(selected or [])
    if wanted - {episode_stem(ep) for ep in episodes}:
        raise ValueError("--episode 必须是数据集中存在的 episode_NNNNNN")
    pointers = [(p, read_json(p)) for p in (process / "features").rglob("approval.json")]
    history = read_json(process / "rebuild_manifest.json")["entries"]
    jobs = []
    for ep in episodes:
        stem = episode_stem(ep)
        if wanted and stem not in wanted:
            continue
        # Old indexed paths may no longer exist; use their explicit episode name.
        matches = [
            (p, a)
            for p, a in pointers
            if (
                Path(a["episode"]).name
                if Path(a["episode"]).name.startswith("episode_")
                else "episode_000000"
            )
            == stem
        ]
        if len(matches) != 1:
            raise ValueError(f"审核记录缺失或不唯一：{stem}")
        pointer, approval = matches[0]
        rows = [r for r in history if r.get("episode") == approval["episode"] and r.get("in_place")]
        if len(rows) != 1 or not rows[0].get("baseline"):
            raise ValueError(f"缺少已写回数据的原始基线记录：{stem}")
        directory = process / "rebuild_state" / Path(rows[0]["baseline"][0]).parent.parent.name
        snapshot = process / "approved" / Path(approval["directory"]).name
        for old, sha in approval["artifacts"].items():
            if digest(snapshot / Path(old).name) != sha:
                raise ValueError(f"审核产物哈希不符：{old}")
        review = read_json(snapshot / "human_review.json")
        if digest(snapshot / "human_review.json") != approval["review_revision"]:
            raise ValueError("审核快照哈希不符")
        if any(
            review["selected"][s].get("status") != "accepted"
            or review["selected"][s].get("physical_side") != s
            for s in ("left", "right")
        ):
            raise ValueError("需要双手已通过的历史审核选择")
        for relative, stamp in review.get("binding", {}).get("input_signature", {}).items():
            path = dataset_root(ep) / relative
            # Historical signatures can contain baseline hashes instead of stat pairs.
            if isinstance(stamp, list) and len(stamp) == 2:
                stat = path.stat()
                if [str(stat.st_size), str(stat.st_mtime_ns)] != list(map(str, stamp)):
                    raise ValueError(f"历史审核输入已变化：{path}")
        current = check_journal(ep, directory, approval["baseline_hashes"])
        jobs.append(
            dict(
                episode=ep,
                directory=directory,
                snapshot=snapshot,
                approval=approval,
                review=review,
                current=current,
                pointer=pointer,
            )
        )
    return jobs


@contextmanager
def scoped_index(path, jobs):
    previous = os.environ.get("VIVE_REBUILD_INDEX")
    atomic_json(
        path, dict(version=1, episodes={str(j["episode"]): str(j["directory"]) for j in jobs})
    )
    os.environ["VIVE_REBUILD_INDEX"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VIVE_REBUILD_INDEX", None)
        else:
            os.environ["VIVE_REBUILD_INDEX"] = previous


def input_stamp(episode):
    # Metadata statistics change after each committed episode; geometry inputs do not.
    meta = dataset_root(episode) / "meta"
    return tuple(
        s
        for s in source_stamp(episode)
        if not Path(s[0]).is_relative_to(meta) or Path(s[0]).name == "info.json"
    )


def prepare(job, output):
    ep, snapshot = job["episode"], job["snapshot"]
    folder = output / episode_stem(ep)
    folder.mkdir()
    with source_data.source_guard(ep):
        stamp = input_stamp(ep)
        originals = source_data.original_files(ep)
        times = pq.read_table(timestamp_file(ep))["timestamp"].to_numpy()
        observations = {}
        for side in ("left", "right"):
            choice = job["review"]["selected"][side]
            pose, camera = checked_pose(ep, choice, side=side)
            prepare_pose(pose, camera, choice)
            frame = choice["frame_index"]
            observations.setdefault(
                frame, dict(frame_index=frame, timestamp_s=float(times[frame]), stickers={})
            )["stickers"][side] = pose
        report = dict(
            episode=str(ep),
            video_key=VIDEO_KEY,
            camera=dict(K=camera.K.tolist(), resolution=list(camera.resolution)),
            observations=list(observations.values()),
        )
        alignment = build_alignment(ep, report, snapshot / "camera_mount.json")
        alignment.update(
            source_review=str(snapshot / "human_review.json"),
            source_review_sha256=job["approval"]["review_revision"],
            production_approval_created=False,
        )
        files = {}
        for name, path in originals.items():
            table, counts = translate(pq.read_table(path), name, alignment)
            target = folder / name
            pq.write_table(table, target)
            verify_table(pq.read_table(target), table)
            files[name] = dict(path=str(target), sha256=digest(target), counts=counts)
        if stamp != input_stamp(ep):
            raise ValueError("计算期间输入发生变化")
        job["stamp"] = stamp
        return dict(
            episode=str(ep),
            status="prepared",
            files=files,
            origins=alignment["samples"],
            baseline_hashes=job["approval"]["baseline_hashes"],
            source_hashes=job["current"],
        )


def prepare_worker(job, staging):
    row = prepare(job, staging)
    return row, job["stamp"]


def run(dataset, process, *, episodes=None, workers=1):
    if workers < 1:
        raise ValueError("--jobs 必须大于 0")
    dataset, process = Path(dataset).resolve(), Path(process).resolve()
    if process.is_relative_to(dataset):
        raise ValueError("过程目录必须位于源数据集之外")
    with output_lock(process, name=".depth_origin.lock"):
        record = process / "depth_origin_runs" / f"{uuid.uuid4().hex}.json"
        manifest = dict(
            origin_method="depth_pixel",
            workers=workers,
            production_approval_created=False,
            code_signature=export_code_signature(),
            script_sha256=digest(Path(__file__)),
            status="preparing",
            entries=[],
        )

        def save():
            atomic_json(record, manifest, durable=True)

        save()
        try:
            jobs = discover(dataset, process, episodes)
            with TemporaryDirectory(prefix=".depth_origin-", dir=process) as staging:
                staging = Path(staging)
                with scoped_index(staging / "baseline_index.json", jobs):
                    if workers == 1:
                        for job in jobs:
                            print(f"计算 {episode_stem(job['episode'])}", flush=True)
                            manifest["entries"].append(prepare(job, staging))
                            save()
                    else:
                        with ProcessPoolExecutor(
                            max_workers=min(workers, len(jobs)),
                            mp_context=multiprocessing.get_context("spawn"),
                        ) as pool:
                            futures = [pool.submit(prepare_worker, job, staging) for job in jobs]
                            try:
                                for job, future in zip(jobs, futures, strict=True):
                                    row, stamp = future.result()
                                    job["stamp"] = stamp
                                    manifest["entries"].append(row)
                                    print(f"计算完成 {episode_stem(job['episode'])}", flush=True)
                                    save()
                            except BaseException:
                                for future in futures:
                                    future.cancel()
                                raise
                for job in jobs:
                    source_data.register_existing_journal(
                        job["episode"], job["directory"], job["approval"]["baseline_hashes"]
                    )
                for job, row in zip(jobs, manifest["entries"], strict=True):
                    row["status"] = "applying"
                    save()
                    if input_stamp(job["episode"]) != job["stamp"]:
                        raise ValueError("准备后输入发生变化，请重新运行")
                    row.update(
                        source_data.apply_files(
                            job["episode"],
                            row["files"],
                            row["baseline_hashes"],
                            record,
                            process,
                            expected_current=row["source_hashes"],
                        )
                    )
                    for name, item in row["files"].items():
                        item["path"] = str(data_directory(job["episode"]) / name)
                    save()
            manifest["status"] = "complete"
            save()
            return manifest
        except Exception as error:
            manifest.update(status="failed", error=str(error))
            for row in manifest["entries"]:
                if row["status"] == "applying":
                    row.update(status="failed", error=str(error))
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--process-out", required=True, type=Path)
    parser.add_argument("--episode", action="append", help="episode_NNNNNN; repeatable")
    parser.add_argument(
        "--jobs", type=int, default=1, help="并行计算进程数；写回按 episode 顺序提交"
    )
    args = parser.parse_args()
    try:
        result = run(args.dataset, args.process_out, episodes=args.episode, workers=args.jobs)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"失败：{error}\n")
    print(f"{result['status']}: {args.process_out / 'depth_origin_runs'}")


if __name__ == "__main__":
    main()
