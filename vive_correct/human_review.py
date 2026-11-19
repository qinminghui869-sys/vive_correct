"""Local human decisions, native video frames and revalidated label geometry."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from vive_correct.batch.report_io import digest
from vive_correct.dataset import TIMESTAMP_SUFFIX, VIDEO_SUFFIX, input_signature
from vive_correct.estimate.geometry import (
    VIDEO_KEY,
    Camera,
    ReprojectionError,
    estimate_pose,
    order_quad,
)
from vive_correct.review_io import atomic_json, output_lock, read_json

from vive_correct.axis_calibration import CONVENTION, axis_angles, prepare_pose
from vive_correct.runtime_cache import cached_file
from vive_correct.review_catalog import refresh_catalog
from vive_correct.review_exclusions import FILENAME, exclusion
from vive_correct.task_locks import operation_lock

SIDES = ("left", "right")


def output_directory(value):
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "--out 为空：请重新设置 vive_out，或直接填写特征输出目录的绝对路径"
        )
    return Path(value)


def source_signature(episode):
    """Hash the six baseline Parquets; use size/mtime for other input files."""
    from vive_correct.source_data import DATA, FILES, original_files

    episode = Path(episode)
    originals = original_files(episode)
    result = {}
    for folder in ("data", "videos", "calibrations", "annotation"):
        for p in sorted((episode / folder).rglob("*")):
            if not p.is_file() or p.name.endswith((".bak", ".restore")):
                continue
            relative = p.relative_to(episode)
            if relative.parent == DATA and p.name in FILES:
                result[str(relative)] = {"sha256": digest(originals[p.name])}
            else:
                stat = p.stat()
                result[str(relative)] = [stat.st_size, stat.st_mtime_ns]
    return result


def decode_frame(episode, index):
    video = Path(episode) / VIDEO_SUFFIX
    return cached_file(("frame", index), video, lambda: _decode_frame(episode, index))


def _decode_frame(episode, index):
    if type(index) is not int or index < 0:
        raise ValueError("帧号必须为从 0 开始的非负整数")
    video = Path(episode) / VIDEO_SUFFIX
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("校正左目视频缺失或无法解码")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if index >= count:
            raise ValueError(f"帧号超出范围：0–{count - 1}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.5:
            raise ValueError(f"无法准确解码第 {index} 帧")
        return frame, count
    finally:
        capture.release()


def checked_pose(episode, choice, *, side=None):
    image, _ = decode_frame(episode, choice.get("frame_index"))
    camera_path = Path(episode) / "calibrations/rectify_params.yaml"
    camera = cached_file("camera", camera_path, lambda: Camera.load(camera_path))
    height, width = image.shape[:2]
    if camera.resolution != (width, height):
        raise ValueError("视频分辨率与相机标定不一致")
    quad = np.asarray(choice.get("quad"), dtype=float)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("请标注四个有效角点")
    if np.any(quad < 0) or np.any(quad >= [width, height]):
        raise ValueError("标签四角超出原图范围")
    ordered = order_quad(quad)
    try:
        pose = estimate_pose(ordered, camera, allow_origin_only=True)
    except ReprojectionError as error:
        hand = {"left": "左手", "right": "右手"}.get(side, "标签")
        raise ValueError(
            f"{hand}第 {choice['frame_index']} 帧：重投影误差 {error.rms_px:.2f} px，"
            f"超过 {error.limit_px:g} px 阈值。请展开四角调整，放大后将角点对齐白色标签的四角；"
            "仍失败时重选更清晰、遮挡更少的帧。"
        ) from error
    homogeneous = np.column_stack((ordered, np.ones(4)))
    center = np.cross(
        np.cross(homogeneous[0], homogeneous[2]), np.cross(homogeneous[1], homogeneous[3])
    )
    pose.update(
        label_center_px=(center[:2] / center[2]).tolist(),
        quality_status="verified",
        identity_review={"source": "human"},
    )
    return pose, camera


class ReviewStore:
    def __init__(self, output, *, allow_empty=False):
        self.output = Path(output).resolve()
        self.paths, self.catalog_errors = refresh_catalog(self.output)
        if not self.paths and not allow_empty:
            raise ValueError(f"在 {self.output} 未找到特征帧。请确认 --out 指向已有特征输出根目录")

    def refresh(self):
        self.paths, self.catalog_errors = refresh_catalog(self.output)

    def path(self, index):
        if type(index) is not int or not 0 <= index < len(self.paths):
            raise ValueError("未知数据编号")
        return self.paths[index]

    def binding(self, index):
        path = self.path(index)
        episode = Path(read_json(path)["episode"]).resolve()
        # Nanosecond mtimes exceed JavaScript's safe integer range; round trips must be exact.
        signature = {
            name: [str(v) for v in value] if value is not None else None
            for name, value in input_signature(episode).items()
        }
        return dict(features_sha256=digest(path), input_signature=signature)

    def candidates(self, index):
        path = self.path(index)
        report = read_json(path)
        items = [
            dict(value, candidate_id=f"feature:{side}", image_side=side)
            for side, value in report.get("features", {}).items()
        ]
        pool = path.with_name("candidate_pool.json")
        if pool.is_file():
            items += [
                dict(value, candidate_id=f"pool:{i}") for i, value in enumerate(read_json(pool))
            ]
        return items

    def load(self, index):
        path = self.path(index)
        report = read_json(path)
        review_path = path.with_name("human_review.json")
        saved = read_json(review_path) if review_path.exists() else {}
        binding = self.binding(index)
        stale = saved.get("schema_version") != 3 or saved.get("binding") != binding
        selected = saved.get("selected", {}) if saved.get("schema_version") in (2, 3) else {}
        legacy = {
            r.get("physical_side") if r.get("physical_side") in SIDES else r.get("image_side"): r
            for r in saved.get("observations", [])
        }
        choices = {}
        for side in SIDES:
            candidate = report.get("features", {}).get(side, {})
            old = legacy.get(side, {})
            choice = dict(
                selected.get(side)
                or dict(
                    frame_index=old.get("frame_index", candidate.get("frame_index", 0)),
                    quad=old.get("corrected_quad_px") or candidate.get("quad", []),
                    candidate_id=f"feature:{side}" if candidate else None,
                    source="automatic" if candidate else "manual",
                    notes=old.get("notes", ""),
                    status="pending",
                    physical_side="unknown",
                )
            )
            choice.setdefault("axis_angles_deg", [0, 0, 0])
            choice.setdefault("axis_convention", CONVENTION)
            if stale:
                choice.update(status="pending", physical_side="unknown")
            choices[side] = choice
        ready = all(
            choices[s].get("status") == "accepted" and choices[s].get("physical_side") == s
            for s in SIDES
        )
        pending = [
            "左手" if s == "left" else "右手"
            for s in SIDES
            if choices[s].get("status") != "accepted" or choices[s].get("physical_side") != s
        ]
        if ready:
            reason = ""
        elif saved.get("schema_version") in (2, 3) and stale:
            reason = "输入或特征报告已变化，请重新确认双手后导出"
        else:
            reason = "、".join(pending) + "尚未确认通过；请检查后点击「通过审核」（暂存不代表通过）"
        rejected = exclusion(self.output / FILENAME, report["episode"])
        if rejected:
            reason = "整条数据审核不通过，已排除后续导出和重建"
            if rejected.get("reason"):
                reason += "：" + rejected["reason"]
        return dict(
            id=index,
            episode=report["episode"],
            binding=binding,
            revision=digest(review_path) if review_path.exists() else None,
            needs_reconfirmation=stale,
            selected=choices,
            reviewer=saved.get("reviewer") or "",
            status="excluded" if rejected else "ready" if ready else "pending",
            exclusion=rejected,
            pending_reason=reason,
            error=report.get("error"),
            missing_hands=report.get("missing_hands", []),
        )

    def save(self, index, payload):
        with operation_lock(self.path(index).parent), output_lock(self.path(index).parent):
            current = self.load(index)
            if current["exclusion"]:
                raise ValueError(current["pending_reason"])
            if payload.get("revision") != current["revision"]:
                raise ValueError("审核已被修改，请重新加载后操作")
            if payload.get("binding") != current["binding"]:
                raise ValueError("输入或特征报告已变化，请重新加载并审核")
            reviewer = str(payload.get("reviewer", "")).strip()
            selected = payload.get("selected")
            if not isinstance(selected, dict) or set(selected) != set(SIDES):
                raise ValueError("审核必须包含左右手记录，可暂存未完成的一侧")
            if (
                not current["needs_reconfirmation"]
                and selected == current["selected"]
                and reviewer == current["reviewer"]
            ):
                return current
            candidates = {c["candidate_id"]: c for c in self.candidates(index)}
            result = {}
            for side in SIDES:
                item = selected[side]
                angles = axis_angles(item)
                status = item.get("status")
                if status not in ("pending", "accepted", "rejected"):
                    raise ValueError("未知审核状态")
                quad = np.asarray(item.get("quad", []), dtype=float)
                if (quad.ndim != 2 or quad.shape[1:] != (2,) or len(quad) > 4) and quad.size:
                    raise ValueError("角点格式应为至多四组 [x, y]")
                if not np.isfinite(quad).all():
                    raise ValueError("角点必须为有限数值")
                frame = item.get("frame_index")
                if type(frame) is not int or frame < 0:
                    raise ValueError("帧号必须为非负整数")
                if status == "accepted":
                    if not reviewer or item.get("physical_side") != side:
                        raise ValueError("通过前请填写审核人并确认物理左右手")
                    checked_pose(Path(current["episode"]), item, side=side)
                candidate = candidates.get(item.get("candidate_id"))
                automatic = (
                    candidate is not None
                    and candidate["frame_index"] == frame
                    and np.array_equal(np.asarray(candidate["quad"]), quad)
                )
                result[side] = dict(
                    frame_index=frame,
                    axis_angles_deg=angles.tolist(),
                    axis_convention=CONVENTION,
                    quad=quad.tolist(),
                    status=status,
                    physical_side=side if status == "accepted" else "unknown",
                    source="automatic" if automatic else "manual",
                    candidate_id=item.get("candidate_id") if candidate else None,
                    observation_id=candidate.get("observation_id") if candidate else None,
                    notes=str(item.get("notes", "")),
                    reviewer=reviewer,
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                )
            self.check_distinct(result)
            value = dict(
                schema_version=3,
                episode=current["episode"],
                binding=current["binding"],
                reviewer=reviewer,
                selected=result,
            )
            if self.binding(index) != current["binding"]:
                raise ValueError("保存期间输入发生变化，请重新加载")
            # Reconfirming unchanged decisions must not invalidate a completed export.
            for side in SIDES:
                previous = current["selected"][side]
                if all(previous.get(k) == v for k, v in result[side].items() if k != "reviewed_at"):
                    result[side]["reviewed_at"] = previous["reviewed_at"]
            if (
                not current["needs_reconfirmation"]
                and result == current["selected"]
                and reviewer == current["reviewer"]
            ):
                return current
            with output_lock(self.output, blocking=True):
                atomic_json(self.path(index).with_name("human_review.json"), value)
                self.invalidate(current["episode"])
            return self.load(index)

    def reject_episode(self, index, payload):
        """Exclude an episode without requiring valid frames, geometry or source poses."""
        with operation_lock(self.path(index).parent), output_lock(self.path(index).parent):
            current = self.load(index)
            if payload.get("revision") != current["revision"]:
                raise ValueError("审核已被修改，请重新加载后操作")
            if payload.get("binding") != current["binding"]:
                raise ValueError("输入或特征报告已变化，请重新加载后操作")
            reviewer = str(payload.get("reviewer", "")).strip()
            if not reviewer:
                raise ValueError("不通过前请填写审核人")
            if current["exclusion"]:
                return current
            episode = str(Path(current["episode"]).resolve())
            record = dict(
                episode=episode,
                video=str(Path(episode) / VIDEO_SUFFIX),
                features_file=str(self.path(index)),
                reviewer=reviewer,
                reason=str(payload.get("reason", "")).strip(),
                rejected_at=datetime.now(timezone.utc).isoformat(),
                review_revision=current["revision"],
            )
            with output_lock(self.output, blocking=True):
                path = self.output / FILENAME
                document = read_json(path) if path.exists() else dict(schema_version=1, entries=[])
                document["entries"] = [
                    row for row in document["entries"] if row["episode"] != episode
                ] + [record]
                document["entries"].sort(key=lambda row: row["episode"])
                # Publish the exclusion first: downstream checks remain fail-closed
                # even if invalidating an old display manifest subsequently fails.
                atomic_json(path, document, durable=True)
                self.invalidate(current["episode"])
            return self.load(index)

    @staticmethod
    def check_distinct(selected):
        left, right = (selected[s] for s in SIDES)
        if (
            left["status"] == right["status"] == "accepted"
            and left["frame_index"] == right["frame_index"]
            and np.allclose(order_quad(left["quad"]), order_quad(right["quad"]))
        ):
            raise ValueError("同一个标签不能同时标为左右手")

    def invalidate(self, episode):
        for relative in (
            "batch/results/manifest.json",
            "rerun/manifest.json",
            "repair_manifest.json",
        ):
            path = self.output / relative
            if path.exists():
                doc = read_json(path)
                for row in doc.get("entries", []):
                    if row.get("episode") == episode:
                        row["status"] = "stale"
                if relative == "batch/results/manifest.json":
                    doc["entries"] = [r for r in doc["entries"] if r.get("status") != "stale"]
                elif relative == "repair_manifest.json" and any(
                    r.get("status") == "stale" for r in doc.get("entries", [])
                ):
                    doc["status"] = "incomplete"
                atomic_json(path, doc)

    def observation_report(self, index):
        review = self.load(index)
        if review["status"] != "ready":
            raise ValueError("双手尚未通过审核，或审核记录已过期")
        episode = Path(review["episode"]).resolve()
        selected = review["selected"]
        self.check_distinct(selected)
        times = pq.read_table(episode / TIMESTAMP_SUFFIX)["timestamp"].to_numpy()
        observations = {}
        for side in SIDES:
            choice = selected[side]
            if not choice.get("reviewer"):
                raise ValueError("审核记录缺少审核人，请重新确认")
            pose, camera = checked_pose(episode, choice, side=side)
            prepare_pose(pose, camera, choice)
            frame = choice["frame_index"]
            if frame >= len(times) or not np.isfinite(times[frame]):
                raise ValueError("参考帧缺少有效时间戳")
            observation = observations.setdefault(
                frame, dict(frame_index=frame, timestamp_s=float(times[frame]), stickers={})
            )
            observation["stickers"][side] = pose
        review_path = self.path(index).with_name("human_review.json")
        return dict(
            episode=str(episode),
            video_key=VIDEO_KEY,
            origin="label_center",
            identity="human_reviewed",
            reviewer_type="human",
            camera=dict(K=camera.K.tolist(), resolution=list(camera.resolution)),
            observations=[observations[k] for k in sorted(observations)],
            human_review=str(review_path),
            human_review_sha256=digest(review_path),
            exclusions_file=str(self.output / FILENAME),
            review_binding=review["binding"],
            features_file=str(self.path(index)),
            input_signature=source_signature(episode),
        )


def verify_provenance(report):
    """Used again by standalone RRD export; old decisions cannot silently be replayed."""
    if report.get("exclusions_file") and exclusion(report["exclusions_file"], report["episode"]):
        raise ValueError("整条数据审核不通过，已排除后续导出和重建")
    if "human_review_sha256" not in report:
        return
    if (
        digest(Path(report["human_review"])) != report["human_review_sha256"]
        or digest(Path(report["features_file"])) != report["review_binding"]["features_sha256"]
        or source_signature(Path(report["episode"])) != report["input_signature"]
    ):
        raise ValueError("审核或源数据已变化，请重新审核／运行 repair")
