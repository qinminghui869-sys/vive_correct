"""Export approved RRD or rebuild source files, without changing review decisions."""

import argparse
import uuid
from time import perf_counter
from vive_correct.task_locks import operation_lock
from pathlib import Path

from vive_correct.batch.report_io import digest
from vive_correct.human_review import (
    ReviewStore,
    output_directory,
)
from vive_correct.rerun_export import DEFAULT_MOUNT, export_recordings
from vive_correct.review_bundle import load_bundle, stage_files
from vive_correct.review_io import atomic_json, merge_rows, output_lock, read_json
from vive_correct.source_data import apply_files, baseline_read


def export_approved(store, index, calibration, progress=None):
    with operation_lock(store.path(index).parent):
        return _export_approved(store, index, calibration, progress)


def _export_approved(store, index, calibration, progress=None):
    started = perf_counter()
    with output_lock(store.path(index).parent), baseline_read(Path(store.load(index)["episode"])):
        bundle = load_bundle(store, index, calibration)
        prepared = stage_files(bundle)
        prepared_s = perf_counter() - started
        cache_path = Path(bundle["directory"]) / "rrd_manifest.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            recording = Path(cached["recording"])
            if (
                cached.get("corrected_files") == prepared["files"]
                and recording.is_file()
                and digest(recording) == cached["recording_sha256"]
                and recording.with_suffix(".json").is_file()
                and digest(recording.with_suffix(".json")) == cached["stats_sha256"]
            ):
                return dict(cached, reused=True)
        run = store.output / "repairs" / uuid.uuid4().hex
        entry = dict(
            index=index,
            episode=bundle["episode"],
            status="reviewed",
            hands=["left", "right"],
            report=bundle["report"],
            report_sha256=digest(Path(bundle["report"])),
            alignment=bundle["alignment"],
            alignment_sha256=digest(Path(bundle["alignment"])),
            reviewed_data=prepared["data_directory"],
            corrected_files=prepared["files"],
        )
        atomic_json(run / "batch/results/manifest.json", {"entries": [entry]})
        row = dict(index=index, episode=bundle["episode"], status="rendering", run=str(run))
        with output_lock(store.output, blocking=True):
            merge_rows(store.output / "repair_manifest.json", {row["episode"]}, [row])
            for relative in ("rerun/manifest.json", "batch/results/manifest.json"):
                merge_rows(store.output / relative, {row["episode"]}, [])
        try:
            if progress:
                progress("rendering")
            render_start = perf_counter()
            export_recordings(run, Path(bundle["calibration"]))
            row.update(read_json(run / "rerun/manifest.json")["entries"][0])
            if row["status"] != "complete":
                raise ValueError(row.get("error", "RRD 导出失败"))
            load_bundle(store, index, calibration)
            row.update(
                recording_sha256=digest(Path(row["recording"])),
                stats_sha256=digest(Path(row["recording"]).with_suffix(".json")),
                corrected_files=prepared["files"],
                timing=dict(
                    prepare_s=round(prepared_s, 3), render_s=round(perf_counter() - render_start, 3)
                ),
            )
            atomic_json(cache_path, row)
        except Exception as error:
            row.update(status="failed", error=str(error))
        with output_lock(store.output, blocking=True):
            merge_rows(store.output / "repair_manifest.json", {row["episode"]}, [row])
            merge_rows(
                store.output / "rerun/manifest.json",
                {row["episode"]},
                [row] if row["status"] == "complete" else [],
            )
            merge_rows(
                store.output / "batch/results/manifest.json",
                {row["episode"]},
                [entry] if row["status"] == "complete" else [],
            )
        atomic_json(run / "repair_manifest.json", {"entries": [row], "status": row["status"]})
        return row


def rebuild(output, calibration=DEFAULT_MOUNT, *, in_place=False, episode=None):
    store = ReviewStore(output)
    indices = [
        i
        for i in range(len(store.paths))
        if episode is None or Path(store.load(i)["episode"]).resolve() == Path(episode).resolve()
    ]
    if not indices:
        raise ValueError("指定数据不在审核列表中")
    rows = []
    for index in indices:
        row = dict(index=index, status="failed")
        try:
            with operation_lock(store.path(index).parent), output_lock(store.path(index).parent):
                source = Path(store.load(index)["episode"])
                row["episode"] = str(source)
                with baseline_read(source):
                    bundle = load_bundle(store, index, calibration)
                    prepared = stage_files(bundle)
                row.update(files=prepared["files"], status="prepared")
                if in_place:
                    row.update(
                        apply_files(
                            source,
                            prepared["files"],
                            bundle["baseline_hashes"],
                            bundle["directory"],
                        )
                    )
                row["in_place"] = in_place
        except Exception as error:
            row.update(status="failed", error=str(error))
        rows.append(row)
        with output_lock(store.output, blocking=True):
            merge_rows(store.output / "rebuild_manifest.json", {row.get("episode")}, [row])
    return {
        "status": "complete"
        if all(r["status"] in ("prepared", "complete") for r in rows)
        else "incomplete",
        "entries": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=output_directory)
    parser.add_argument("--initial-mount", type=Path, default=DEFAULT_MOUNT)
    parser.add_argument("--episode", type=Path)
    parser.add_argument(
        "--in-place", action="store_true", help="备份为 .parquet.bak 后覆盖六个源文件"
    )
    args = parser.parse_args(argv)
    try:
        result = rebuild(args.out, args.initial_mount, in_place=args.in_place, episode=args.episode)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"rebuild: {error}\n")
    for row in result["entries"]:
        print(f"{row.get('episode', row['index'])}: {row['status']} {row.get('error', '')}")
    return int(result["status"] != "complete")
