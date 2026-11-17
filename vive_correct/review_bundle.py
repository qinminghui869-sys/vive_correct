"""Approved evidence and six-file correction, independent of RRD rendering."""

import uuid
from time import perf_counter
from vive_correct.task_locks import operation_lock
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from ego_data.vive_origin_alignment import build_alignment
from vive_correct.batch.report_io import digest
from vive_correct.human_review import (
    decode_frame,
    verify_provenance,
)
from vive_correct.parquet_correction import translate, verify_table
from vive_correct.review_io import atomic_json, export_code_signature, output_lock, read_json
from vive_correct.source_data import DATA, FILES, baseline_read, original_files
from vive_correct.axis_calibration import enrich_pose


def load_bundle(store, index, calibration):
    pointer = store.path(index).with_name("approval.json")
    if not pointer.exists():
        raise ValueError("尚未保存完整重建依据，请点击「通过审核」")
    bundle = read_json(pointer)
    state = store.load(index)
    if state["status"] != "ready" or state["revision"] != bundle["review_revision"]:
        raise ValueError("审核选择已变化，请重新点击「通过审核」")
    if read_json(calibration) != read_json(bundle["calibration"]):
        raise ValueError("标定已变化，请重新点击「通过审核」")
    if bundle.get("code_signature") != export_code_signature():
        raise ValueError("计算代码已变化，请重新点击「通过审核」生成依据")
    for path, signature in bundle["artifacts"].items():
        if digest(Path(path)) != signature:
            raise ValueError("审核依据文件已变化，请重新通过审核")
    verify_provenance(read_json(bundle["report"]))
    return bundle


def review_state(store, index, calibration):
    state = store.load(index)
    try:
        bundle = load_bundle(store, index, calibration)
        state.update(approved=True, approval=bundle["directory"])
    except (ValueError, OSError, KeyError, TypeError) as error:
        state.update(approved=False, approval_error=str(error))
        if state["status"] == "ready":
            state.update(status="pending", pending_reason=str(error))
    return state


def approve(store, index, payload, calibration):
    with operation_lock(store.path(index).parent):
        return _approve(store, index, payload, calibration)


def _approve(store, index, payload, calibration):
    try:
        state = store.save(index, payload)
    except ValueError:
        # Keep geometrically invalid edits as a draft, never as a successful approval.
        draft = dict(
            payload,
            selected={
                s: dict(c, status="pending", physical_side="unknown")
                for s, c in payload.get("selected", {}).items()
            },
        )
        try:
            store.save(index, draft)
        except (ValueError, OSError, KeyError, TypeError):
            pass
        raise
    if state["status"] != "ready":
        raise ValueError(state["pending_reason"])
    with output_lock(store.path(index).parent), baseline_read(Path(state["episode"])):
        if store.load(index)["revision"] != state["revision"]:
            raise ValueError("审核记录在保存后发生变化，请重新加载")
        try:
            return load_bundle(store, index, calibration)
        except (ValueError, OSError, KeyError):
            pass
        folder = store.output / "approved" / f"{index:04d}_{uuid.uuid4().hex[:12]}"
        folder.mkdir(parents=True)
        report = store.observation_report(index)
        episode = Path(report["episode"])
        report_path, mount_path = folder / "observations.json", folder / "camera_mount.json"
        report["reviewer"] = state["reviewer"]
        snapshot = folder / "human_review.json"
        atomic_json(snapshot, read_json(store.path(index).with_name("human_review.json")))
        atomic_json(report_path, report)
        atomic_json(mount_path, read_json(calibration))
        alignment = build_alignment(episode, report_path, mount_path, refine_depth=True)
        for observation in report["observations"]:
            for side, pose in observation["stickers"].items():
                sample = next(s for s in alignment["samples"] if s["side"] == side)
                enrich_pose(pose, sample, report["camera"]["K"])
        atomic_json(report_path, report)
        alignment["source_report_sha256"] = digest(report_path)
        alignment_path = folder / "wrist_alignment.json"
        atomic_json(alignment_path, alignment)
        originals = original_files(episode)
        baseline = {}
        for name, path in originals.items():
            table = pq.read_table(path)
            translate(table, name, alignment)  # Validate all six schemas before approval.
            baseline[name] = digest(path)
        images = {}
        for side, choice in state["selected"].items():
            frame = choice["frame_index"]
            image, _ = decode_frame(episode, frame)
            clean = folder / f"{side}_frame_{frame}.png"
            overlay = folder / f"{side}_frame_{frame}_quad.png"
            if not cv2.imwrite(str(clean), image):
                raise ValueError("特征原帧保存失败")
            points = np.rint(choice["quad"]).astype(np.int32)
            cv2.polylines(image, [points], True, (100, 240, 30), 2)
            for i, point in enumerate(points):
                cv2.putText(
                    image, f"q{i}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 240, 30), 2
                )
            pose = next(
                o["stickers"][side] for o in report["observations"] if side in o["stickers"]
            )
            projection = pose["axis_projection"]
            if projection["origin"] is not None:
                origin = tuple(np.rint(projection["origin"]).astype(int))
                for label, endpoint, color in zip(
                    "XYZ", projection["endpoints"], [(0, 0, 255), (0, 200, 0), (255, 100, 0)]
                ):
                    if endpoint is None:
                        continue
                    end = tuple(np.rint(np.clip(endpoint, -1e7, 1e7)).astype(int))
                    visible, start, end = cv2.clipLine(
                        (0, 0, image.shape[1], image.shape[0]), origin, end
                    )
                    if visible:
                        cv2.arrowedLine(image, start, end, color, 2, tipLength=0.15)
                        cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if not cv2.imwrite(str(overlay), image):
                raise ValueError("特征叠图保存失败")
            images[side] = dict(
                frame_index=frame, clean=str(clean), overlay=str(overlay), quad=choice["quad"]
            )
        artifacts = [report_path, mount_path, alignment_path, snapshot] + [
            Path(v[k]) for v in images.values() for k in ("clean", "overlay")
        ]
        bundle = dict(
            version=1,
            directory=str(folder),
            episode=str(episode),
            review_revision=state["revision"],
            reviewer=state["reviewer"],
            human_review_snapshot=str(snapshot),
            report=str(report_path),
            alignment=str(alignment_path),
            calibration=str(mount_path),
            baseline_hashes=baseline,
            images=images,
            code_signature=export_code_signature(),
            artifacts={str(p): digest(p) for p in artifacts},
        )
        verify_provenance(report)
        if read_json(calibration) != read_json(mount_path):
            raise ValueError("审核期间标定发生变化，请重试")
        atomic_json(folder / "bundle.json", bundle)
        atomic_json(store.path(index).with_name("approval.json"), bundle)
        return bundle


def stage_files(bundle):
    """Build once; both in-place rebuild and rendering consume these exact Parquets."""
    started = perf_counter()
    folder = Path(bundle["directory"])
    cache = folder / "corrected_manifest.json"
    originals = original_files(Path(bundle["episode"]))
    if {n: digest(p) for n, p in originals.items()} != bundle["baseline_hashes"]:
        raise ValueError("原始文件已变化，请重新审核")
    if cache.exists():
        saved = read_json(cache)
        if all(
            Path(row["path"]).is_file() and digest(Path(row["path"])) == row["sha256"]
            for row in saved["files"].values()
        ) and set(saved["files"]) == set(FILES):
            return saved
    alignment = read_json(bundle["alignment"])
    destination = folder / ("corrected_" + uuid.uuid4().hex[:8]) / DATA
    destination.mkdir(parents=True)
    files = {}
    for name in FILES:
        table = pq.read_table(originals[name])
        corrected, counts = translate(table, name, alignment)
        path = destination / name
        pq.write_table(corrected, path)
        verify_table(pq.read_table(path), corrected)
        files[name] = dict(
            path=str(path), sha256=digest(path), rows=len(table), corrected_groups=counts
        )
    verify_provenance(read_json(bundle["report"]))
    result = dict(
        status="complete",
        data_directory=str(destination),
        files=files,
        baseline_hashes=bundle["baseline_hashes"],
        elapsed_s=round(perf_counter() - started, 3),
    )
    atomic_json(cache, result)
    return result
