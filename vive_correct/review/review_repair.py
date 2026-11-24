"""Non-destructive repair CLI, including compatibility with pre-bundle wrist reviews."""

import argparse
import uuid
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import pyarrow.parquet as pq

from ego_data.vive_origin_alignment import build_alignment
from vive_correct.batch.report_io import digest
from vive_correct.review.human_review import (
    SIDES,
    ReviewStore,
    output_directory,
    verify_provenance,
)
from vive_correct.review.parquet_correction import translate, verify_table
from vive_correct.export.rerun_export import DEFAULT_MOUNT, export_recordings
from vive_correct.review.review_io import (
    atomic_json,
    export_code_signature,
    merge_rows,
    output_lock,
    read_json,
)
from vive_correct.data.layout import data_directory
from vive_correct.data.source_data import DATA, original_path
from vive_correct.runtime.task_locks import operation_lock
from vive_correct.review.review_exclusions import FILENAME, skipped


def write_corrected_wrists(episode, destination, alignment):
    """Preserve rows, Arrow nulls, schema and all fields except finite xyz positions."""
    files = {}
    for side in SIDES:
        relative = DATA / f"observation.{side}_wrist_pose.parquet"
        original = pq.read_table(
            original_path(data_directory(episode) / f"observation.{side}_wrist_pose.parquet")
        )
        table, counts = translate(original, relative.name, alignment)
        corrected_rows = counts["x"]
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        verify_table(pq.read_table(path), table)
        files[side] = dict(
            path=str(path),
            sha256=digest(path),
            rows=len(table),
            corrected_rows=corrected_rows,
            missing_rows=len(table) - corrected_rows,
        )
    return files


def artifact_signature(paths):
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in map(Path, paths)}


def cached_result(store, index, mount, code_signature):
    """Reuse only a complete result whose decisions, inputs, code and artifacts still match."""
    try:
        episode = store.load(index)["episode"]
        summary = read_json(store.output / "repair_manifest.json")
        row = next(r for r in summary["entries"] if r.get("episode") == episode)
        if row["status"] != "complete" or row.get("export_code_signature") != code_signature:
            return None
        entry = next(
            r
            for r in read_json(store.output / "batch/results/manifest.json")["entries"]
            if r["episode"] == episode
        )
        if read_json(entry["alignment"])["camera_calibration"] != mount:
            return None
        verify_provenance(read_json(entry["report"]))
        signatures = row["artifact_signature"]
        if artifact_signature(signatures) != signatures:
            return None
        return dict(row, reused=True), entry
    except (OSError, ValueError, KeyError, StopIteration):
        return None


def repair_dataset(
    output,
    calibration=DEFAULT_MOUNT,
    *,
    indices=None,
    progress=None,
    expected_revision=None,
    export_jobs=1,
):
    if export_jobs < 1:
        raise ValueError("--export-jobs 必须为正整数")
    store = ReviewStore(output)
    targets = list(range(len(store.paths))) if indices is None else list(dict.fromkeys(indices))
    for index in targets:
        store.path(index)
    if (store.output / FILENAME).exists() or any(
        store.path(i).with_name("approval.json").exists()
        or (
            store.path(i).with_name("human_review.json").exists()
            and read_json(store.path(i).with_name("human_review.json")).get("schema_version") == 3
        )
        for i in targets
    ):
        from vive_correct.review.review_actions import export_approved

        def export_one(i):
            try:
                state = store.load(i)
                if state["exclusion"]:
                    return skipped(state, i)
                if state["status"] != "ready":
                    return dict(
                        index=i,
                        episode=state["episode"],
                        status="pending",
                        reason=state["pending_reason"],
                    )
                return export_approved(store, i, calibration, progress)
            except Exception as error:
                return dict(
                    index=i,
                    episode=str(store.path(i).parent),
                    status="failed",
                    error=str(error),
                )

        with ThreadPoolExecutor(max_workers=export_jobs) as pool:
            rows = list(pool.map(export_one, targets))
        return dict(
            status="complete"
            if all(r["status"] in ("complete", "excluded") for r in rows)
            else "incomplete",
            entries=rows,
        )
    mount = read_json(Path(calibration).resolve())
    code_signature = export_code_signature()
    started = perf_counter()
    # Long work never holds the shared manifest lock. Browser jobs select exactly one episode.
    with ExitStack() as locks:
        for index in sorted(targets):
            locks.enter_context(operation_lock(store.path(index).parent))
            locks.enter_context(output_lock(store.path(index).parent))
        if expected_revision is not None and any(
            store.load(i)["revision"] != expected_revision for i in targets
        ):
            raise ValueError("审核已被修改，请重新提交导出")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run = store.output / "repairs" / run_id
        run.mkdir(parents=True)
        mount_path = run / "camera_mount.json"
        atomic_json(mount_path, mount)
        summary = dict(run=str(run), status="running", entries=[])
        entries, rendering_rows = [], []
        cache = {i: cached_result(store, i, mount, code_signature) for i in targets}
        rebuilding = {store.load(i)["episode"] for i in targets if cache[i] is None}
        with output_lock(store.output, blocking=True):
            for relative in ("batch/results/manifest.json", "rerun/manifest.json"):
                merge_rows(store.output / relative, rebuilding, [])
            merge_rows(
                store.output / "repair_manifest.json",
                rebuilding,
                [dict(episode=e, status="running", run=str(run)) for e in rebuilding],
            )
        for index in targets:
            row = dict(index=index, status="failed", run=str(run))
            try:
                state = store.load(index)
                row["episode"] = state["episode"]
                if state["status"] != "ready":
                    row.update(status="pending", reason=state["pending_reason"])
                elif cache[index] is not None:
                    row, entry = cache[index]
                    entries.append(entry)
                    rendering_rows.append(row)
                else:
                    if progress:
                        progress("preparing")
                    report = store.observation_report(index)
                    # Historical position-only exports never imply an axis review.
                    for observation in report["observations"]:
                        for pose in observation["stickers"].values():
                            pose.pop("axis_angles_deg", None)
                            pose.pop("axis_convention", None)
                    episode = Path(report["episode"])
                    folder = run / f"{index:04d}_{episode.name}"
                    report_path = folder / "observations.json"
                    atomic_json(report_path, report)
                    alignment = build_alignment(episode, report_path, mount_path, refine_depth=True)
                    alignment_path = folder / "wrist_alignment.json"
                    atomic_json(alignment_path, alignment)
                    files = write_corrected_wrists(episode, folder / "corrected", alignment)
                    verify_provenance(report)
                    entry = dict(
                        index=index,
                        episode=str(episode),
                        report=str(report_path),
                        report_sha256=digest(report_path),
                        status="reviewed",
                        hands=list(SIDES),
                        alignment=str(alignment_path),
                        alignment_sha256=digest(alignment_path),
                        corrected_files=files,
                    )
                    entries.append(entry)
                    row.update(
                        status="prepared",
                        report=str(report_path),
                        alignment=str(alignment_path),
                        corrected_files=files,
                    )
            except (ValueError, OSError, KeyError, TypeError, IndexError, cv2.error) as error:
                row["error"] = str(error)
            summary["entries"].append(row)
        to_render = [e for e in entries if e["episode"] in rebuilding]
        atomic_json(run / "batch/results/manifest.json", {"entries": to_render})
        with output_lock(store.output, blocking=True):
            merge_rows(
                store.output / "repair_manifest.json",
                rebuilding,
                [r for r in summary["entries"] if r["episode"] in rebuilding],
            )
        if to_render:
            try:
                if progress:
                    progress("rendering")
                export_recordings(run, mount_path)
                rendering = read_json(run / "rerun/manifest.json")
                by_episode = {r["episode"]: r for r in rendering["entries"]}
                for row in summary["entries"]:
                    if row["status"] == "prepared":
                        row.update(by_episode[row["episode"]])
                        if row["status"] == "complete":
                            entry = next(e for e in entries if e["episode"] == row["episode"])
                            try:
                                verify_provenance(read_json(entry["report"]))
                                paths = [
                                    entry["report"],
                                    entry["alignment"],
                                    row["alignment"],
                                    row["recording"],
                                    str(Path(row["recording"]).with_suffix(".json")),
                                ]
                                paths += [f["path"] for f in entry["corrected_files"].values()]
                                row.update(
                                    export_code_signature=code_signature,
                                    artifact_signature=artifact_signature(paths),
                                )
                            except (ValueError, OSError, KeyError) as error:
                                row.update(status="failed", error=str(error))
                        rendering_rows.append(row)
            except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
                for row in summary["entries"]:
                    if row["status"] in ("prepared", "complete") and row["episode"] in rebuilding:
                        row.update(status="failed", error=str(error))
        complete = {r["episode"] for r in summary["entries"] if r["status"] == "complete"}
        summary["status"] = "complete" if len(complete) == len(targets) else "incomplete"
        summary["elapsed_s"] = round(perf_counter() - started, 3)
        affected = {r["episode"] for r in summary["entries"]}
        completed_entries = [e for e in entries if e["episode"] in complete]
        completed_renderings = [r for r in rendering_rows if r["episode"] in complete]
        with output_lock(store.output, blocking=True):
            merge_rows(
                store.output / "batch/results/manifest.json",
                affected,
                completed_entries,
            )
            merge_rows(
                store.output / "rerun/manifest.json",
                affected,
                completed_renderings,
            )
            merge_rows(store.output / "repair_manifest.json", affected, summary["entries"])
        atomic_json(run / "batch/results/manifest.json", {"entries": completed_entries})
        atomic_json(run / "rerun/manifest.json", {"entries": completed_renderings})
        atomic_json(run / "repair_manifest.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=output_directory, required=True)
    parser.add_argument("--initial-mount", type=Path, default=DEFAULT_MOUNT)
    parser.add_argument("--export-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        summary = repair_dataset(args.out, args.initial_mount, export_jobs=args.export_jobs)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"repair: {error}\n")
    for row in summary["entries"]:
        print(
            f"{row.get('episode', row['index'])}: {row['status']} {row.get('error', row.get('reason', ''))}"
        )
    print(f"结果：{args.out.resolve() / 'repair_manifest.json'}")
    return int(summary["status"] != "complete")
