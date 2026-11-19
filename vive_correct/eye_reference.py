"""Reference-eye anchoring of tracked Vive poses and existing hand shapes."""

from pathlib import Path
import json
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from ego_data.tracking_sync import resample_tracking
from ego_data.vive_origin_alignment import pose_stream
from ego_data.wrist_alignment import pose_matrix
from vive_correct.axis_calibration import local_rotation, project_axes
from vive_correct.batch.hand_overlay import JOINTS
from vive_correct.camera_input import EYE_KEY, timestamp_suffix, verify
from vive_correct.hand_repair import NODES, OFFSET
from vive_correct.source_data import DATA, FILES, original_files
from vive_correct.runtime_cache import cached_file

POSE_COLUMNS = ("x", "y", "z", "roll", "pitch", "yaw")
POINT_COLUMNS = [f"{j}_{a}" for j in JOINTS for a in "xyz"]
ORIGIN_ONLY_MODEL = "eye_origin_standing_axes"


def missing_inputs(episode):
    return [
        str(p)
        for p in [
            *(DATA / n for n in FILES),
            DATA / "observation.hand_skeleton.parquet",
            timestamp_suffix({"eye": True}),
        ]
        if not (Path(episode) / p).is_file()
    ]


def values(table, columns):
    for c in columns:
        if c not in table.column_names or not pa.types.is_floating(table.schema.field(c).type):
            raise ValueError(f"缺少浮点字段 {c}")
    return np.column_stack([table[c].to_numpy() for c in columns]).astype(float)


def sampled_pose(episode, side, times):
    return resample_tracking(
        pose_stream(Path(episode) / DATA / f"observation.{side}_wrist_pose.parquet"),
        times,
        pose=True,
    )[1]


def map_state_fields(episode, table):
    """Verify recorded device renaming against actual source pose samples."""
    journal = Path(episode) / ".vive_rewrite_journal.json"
    mapping = json.loads(journal.read_text()).get("mapping", {}) if journal.exists() else {}
    times = table["timestamp"].to_numpy()
    sample = np.linspace(0, len(table) - 1, min(len(table), 256), dtype=int)
    result = {}
    for side, key in (("left", "L"), ("right", "R")):
        named = f"{side}_wrist"
        if all(f"{named}_{c}" in table.column_names for c in POSE_COLUMNS):
            candidates = [named]
        else:
            candidates = [f"device{i}" for i in range(3) if f"device{i}_x" in table.column_names]
        ts, pose = pose_stream(Path(episode) / DATA / f"observation.{side}_wrist_pose.parquet")
        order = np.argsort(ts)
        ts, pose = ts[order], pose[order]
        matches = []
        for prefix in candidates:
            rows = values(table, [f"{prefix}_{c}" for c in POSE_COLUMNS])[sample]
            valid = np.isfinite(rows).all(1) & np.isfinite(times[sample])
            hits = np.zeros(len(rows), bool)
            right = np.searchsorted(ts, times[sample])
            for delta in range(-4, 5):
                idx = np.clip(right + delta, 0, max(0, len(ts) - 1))
                if not len(ts):
                    continue
                equal = np.isclose(pose[idx], rows, rtol=1e-5, atol=1e-5).all(1)
                hits |= equal & (abs(ts[idx] - times[sample]) <= 0.05)
            if valid.sum() and hits[valid].mean() >= 0.95:
                matches.append(prefix)
        if len(matches) != 1:
            raise ValueError(f"{side}: action/state 设备字段映射不明确：{matches}")
        prefix = matches[0]
        recorded = mapping.get(key)
        if recorded and prefix.startswith("device") and recorded != f"observation.{prefix}_pose":
            raise ValueError(f"{side}: 重命名记录与数值映射不一致")
        result[side] = prefix
    if result["left"] == result["right"]:
        raise ValueError("左右手不能映射到同一设备")
    return result


def validate_hands(episode):
    originals = original_files(episode)
    skeleton = pq.read_table(Path(episode) / DATA / "observation.hand_skeleton.parquet")
    sk = values(skeleton, [f"node{i:02d}_{a}" for i in range(50) for a in "xyz"])
    evidence = {}
    for side in ("left", "right"):
        hand = pq.read_table(originals[f"observation.hand_{side}_points.parquet"])
        times = hand["timestamp"].to_numpy()
        pose = sampled_pose(episode, side, times)
        points = values(hand, POINT_COLUMNS).reshape(-1, 21, 3)
        valid = np.isfinite(pose).all(1) & np.isfinite(points[:, 0]).all(1)
        if not valid.any():
            raise ValueError(f"{side}: 没有可同步的腕部手点")
        R = Rotation.from_euler("xyz", pose[valid, 3:], degrees=True).as_matrix()
        local = np.einsum("nji,nkj->nki", R, points[valid] - pose[valid, None, :3])
        if np.nanmax(np.linalg.norm(local[:, 0] - OFFSET, axis=1)) > 0.001:
            raise ValueError(f"{side}: 现有腕部安装偏移不符合 [0,-.085,-.058] m")
        sampled = resample_tracking((skeleton["timestamp"].to_numpy(), sk), times[valid])[
            1
        ].reshape(-1, 50, 3)
        matches = []
        for block in (0, 25):
            expected = sampled[:, NODES + block][:, :, [1, 2, 0]] + OFFSET
            error = np.linalg.norm(local - expected, axis=2)
            finite = np.isfinite(error)
            if finite.any() and error[finite].max() <= 0.001:
                matches.append(block)
        if len(matches) != 1:
            raise ValueError(f"{side}: 手点与原骨架安装关系不一致或无法区分")
        evidence[side] = dict(
            source_block=matches[0], valid_rows=int(valid.sum()), physical_side_verified=False
        )
    if evidence["left"]["source_block"] == evidence["right"]["source_block"]:
        raise ValueError("左右手 point 使用了同一组骨架")
    return evidence


def reference_sample(episode, side, choice, pose, K):
    times = pq.read_table(Path(episode) / timestamp_suffix({"eye": True}))["timestamp"].to_numpy()
    frame = choice["frame_index"]
    if frame >= len(times) or not np.isfinite(times[frame]):
        raise ValueError("参考帧缺少有效眼动时间戳")
    source = sampled_pose(episode, side, [times[frame]])[0]
    if not np.isfinite(source).all():
        raise ValueError("参考帧没有 50 ms 内的有效 Vive 位姿")
    # Manual intrinsic XYZ angles map the label basis to the reviewed device basis.
    from vive_correct.estimate.geometry import Camera, local_basis

    label = np.asarray(pose["rotation_camera_label"])
    camera = Camera(np.asarray(K), (1, 1), "review")
    base = label @ local_basis(
        label, np.asarray(pose["origin_camera_m"]), camera, require_direction_limits=False
    )
    corrected = base @ local_rotation(choice)
    anchor = np.eye(4)
    anchor[:3, :3], anchor[:3, 3] = corrected, pose["origin_camera_m"]
    alignment = anchor @ np.linalg.inv(pose_matrix(source, "xyz", True))
    return dict(
        side=side,
        source_rgb_frame=frame,
        timestamp_s=float(times[frame]),
        origin_camera_m=pose["origin_camera_m"],
        wrist_pose_original=source.tolist(),
        rotation_camera_source=base.tolist(),
        rotation_camera_corrected=corrected.tolist(),
        rotation_local=local_rotation(choice).tolist(),
        T_reference_source=alignment.tolist(),
        reference_pose=anchor.tolist(),
        depth_refinement={"status": "pnp_only"},
    )


# Orientation constraints have the metric weight of a 10 cm axis.
SHARED_AXIS_WEIGHT_M = 0.10
SHARED_POSITION_LIMIT_M = 0.03
SHARED_ROTATION_LIMIT_DEG = 15.0


def fit_shared_reference(samples, *, require_bilateral=False):
    """Fit one SE(3) to reviewed device anchors, preserving all source relationships."""
    if not samples or len({s["timestamp_s"] for s in samples}) != 1:
        raise ValueError("共同外参需要同一时刻的有效位姿")
    if require_bilateral and {s["side"] for s in samples} != {"left", "right"}:
        raise ValueError("共同外参需要左右两侧标签审核")
    source = np.asarray([pose_matrix(s["wrist_pose_original"]) for s in samples])
    target = np.asarray([s["reference_pose"] for s in samples])
    x, y = source[:, :3, 3], target[:, :3, 3]
    covariance = (x - x.mean(0)).T @ (y - y.mean(0))
    covariance += SHARED_AXIS_WEIGHT_M**2 * sum(
        a[:3, :3] @ b[:3, :3].T for a, b in zip(source, target)
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = y.mean(0) - rotation @ x.mean(0)
    residuals = {}
    for sample, original, observed in zip(samples, source, target):
        predicted = matrix @ original
        residuals[sample["side"]] = dict(
            position_m=float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])),
            rotation_deg=float(
                np.rad2deg(Rotation.from_matrix(predicted[:3, :3].T @ observed[:3, :3]).magnitude())
            ),
        )
        sample.update(
            reviewed_reference_pose=observed.tolist(),
            reference_pose=predicted.tolist(),
            origin_camera_m=predicted[:3, 3].tolist(),
            rotation_camera_corrected=predicted[:3, :3].tolist(),
            T_reference_source=matrix.tolist(),
            shared_extrinsic=True,
        )
    consistent = all(
        r["position_m"] <= SHARED_POSITION_LIMIT_M
        and r["rotation_deg"] <= SHARED_ROTATION_LIMIT_DEG
        for r in residuals.values()
    )
    quality = dict(
        method="joint_rigid_position_orientation_fit",
        sides=[s["side"] for s in samples],
        provisional=len(samples) != 2,
        consistent=consistent,
        residuals=residuals,
        axis_weight_m=SHARED_AXIS_WEIGHT_M,
        position_limit_m=SHARED_POSITION_LIMIT_M,
        rotation_limit_deg=SHARED_ROTATION_LIMIT_DEG,
        mounting_status="label centre approximates device origin; reviewed label-to-device rotation",
        source_pose_decoding="xyz extrinsic degrees; acquisition conversion not independently verified",
    )
    if require_bilateral and not consistent:
        details = "; ".join(
            f"{side}: {r['position_m'] * 1000:.1f} mm / {r['rotation_deg']:.1f} deg"
            for side, r in residuals.items()
        )
        raise ValueError("共同外参不一致，请核对四角、标签安装方向和源位姿编码：" + details)
    return matrix, quality


def shared_preview(episode, side, selected, context):
    from vive_correct.human_review import checked_pose
    from vive_correct.axis_calibration import prepare_pose, enrich_pose, corner_correction

    if side not in ("left", "right") or side not in selected:
        raise ValueError("未知物理手侧")
    if len(selected[side].get("quad", [])) != 4:
        raise ValueError("请标注四个有效角点")
    if len({c["frame_index"] for c in selected.values()}) != 1:
        raise ValueError("共同外参要求左右手同帧")
    samples, poses = [], {}
    for hand, choice in selected.items():
        if hand not in ("left", "right"):
            raise ValueError("未知物理手侧")
        if len(choice.get("quad", [])) != 4:
            continue
        pose, camera = checked_pose(Path(episode), choice, side=hand, context=context)
        prepare_pose(pose, camera, choice)
        poses[hand] = pose
        samples.append(reference_sample(episode, hand, choice, pose, camera.K))
    camera_source, quality = fit_shared_reference(samples)
    rebase_reference(samples, camera_source)
    sample = next(s for s in samples if s["side"] == side)
    result = enrich_pose(poses[side], sample, camera.K)
    result["shared_fit"] = quality
    try:
        result["corner_correction"] = corner_correction(result, camera)
        result["corner_correction_error"] = None
    except ValueError as error:
        result["corner_correction"] = None
        result["corner_correction_error"] = str(error)
    return enrich_hand_preview(result, episode, side, selected[side]["frame_index"], camera.K)


def rebase_reference(samples, camera_source):
    """Move the origin to the reference eye camera without rotating source world axes.

    camera_source = camera_reference @ reference_source. Only reference_source
    is applied to output data; camera_reference is used for image projection.
    """
    camera_source = np.asarray(camera_source)
    reference_source = np.eye(4)
    reference_source[:3, 3] = camera_source[:3, :3].T @ camera_source[:3, 3]
    camera_reference = np.eye(4)
    camera_reference[:3, :3] = camera_source[:3, :3]
    for sample in samples:
        sample.update(
            camera_pose=sample["reference_pose"],
            reference_pose=(reference_source @ pose_matrix(sample["wrist_pose_original"])).tolist(),
            T_reference_source=reference_source.tolist(),
            T_camera_source=camera_source.tolist(),
            T_camera_reference=camera_reference.tolist(),
            orientation_policy="preserve_source",
        )
    return reference_source, camera_reference


def build_alignment(episode, report, context, *, validate=True):
    verify(context)
    observations = report["observations"]
    if len(observations) != 1 or set(observations[0]["stickers"]) != {"left", "right"}:
        raise ValueError("眼动固定参考系要求同一参考帧中审核左右手")
    obs = observations[0]
    samples = [
        reference_sample(
            episode, side, dict(pose, frame_index=obs["frame_index"]), pose, report["camera"]["K"]
        )
        for side, pose in obs["stickers"].items()
    ]
    if any(abs(s["timestamp_s"] - obs["timestamp_s"]) > 1e-6 for s in samples):
        raise ValueError("参考帧时间戳不匹配")
    camera_source, quality = fit_shared_reference(samples, require_bilateral=True)
    shared, camera_reference = rebase_reference(samples, camera_source)
    fields, evidence = {}, {}
    if validate:
        evidence = validate_hands(episode)
        originals = original_files(episode)
        fields = {
            name: map_state_fields(episode, pq.read_table(originals[name]))
            for name in ("action.parquet", "observation.state.parquet")
        }
    return dict(
        version=5,
        model="eye_reference",
        extrinsic_model=ORIGIN_ONLY_MODEL,
        T_reference_source=shared.tolist(),
        T_camera_source=camera_source.tolist(),
        T_camera_reference=camera_reference.tolist(),
        camera_origin_source_m=(-shared[:3, 3]).tolist(),
        orientation_policy="preserve_source",
        shared_fit=quality,
        coordinate_mode="eye-reference",
        episode=str(Path(episode).resolve()),
        camera_calibration=context,
        video_key=EYE_KEY,
        world_axes="RUB",
        world_frame=ORIGIN_ONLY_MODEL,
        reference_frame=obs["frame_index"],
        reference_timestamp_s=obs["timestamp_s"],
        pose_euler_sequence="xyz",
        pose_degrees=True,
        offset_local_m=OFFSET.tolist(),
        state_fields=fields,
        hand_validation=evidence,
        samples=samples,
        wrists={
            s["side"]: dict(model="eye_reference", T_reference_source=s["T_reference_source"])
            for s in samples
        },
        status="Label centre is a Vive-origin approximation; mounting and physical axes not independently calibrated",
    )


def transform_poses(pose, matrix, *, preserve_rotation=False):
    result = np.full_like(pose, np.nan, dtype=float)
    A = np.asarray(matrix)
    if preserve_rotation:
        if not np.array_equal(A[:3, :3], np.eye(3)):
            raise ValueError("保留源轴向模式只允许平移")
        result[:, 3:] = pose[:, 3:]
    valid = np.isfinite(pose).all(1)
    if valid.any():
        result[valid, :3] = pose[valid, :3] @ A[:3, :3].T + A[:3, 3]
        if preserve_rotation:
            return result
        R = A[:3, :3] @ Rotation.from_euler("xyz", pose[valid, 3:], degrees=True).as_matrix()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result[valid, 3:] = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return result


def correct_table(table, name, alignment):
    counts = {}
    origin_only = alignment.get("extrinsic_model") == ORIGIN_ONLY_MODEL

    def write(columns, result):
        nonlocal table
        for i, col in enumerate(columns):
            field = table.schema.field(col)
            table = table.set_column(
                table.schema.get_field_index(col),
                field,
                pa.array(result[:, i], type=field.type, mask=table[col].is_null().to_numpy()),
            )

    for side in ("left", "right"):
        matrix = alignment.get(
            "T_reference_source", alignment["wrists"][side]["T_reference_source"]
        )
        if name == f"observation.hand_{side}_points.parquet":
            points = values(table, POINT_COLUMNS).reshape(-1, 21, 3)
            pose = sampled_pose(alignment["episode"], side, table["timestamp"].to_numpy())
            corrected = transform_poses(pose, matrix, preserve_rotation=origin_only)
            result = np.full_like(points, np.nan)
            valid = np.isfinite(pose).all(1) & np.isfinite(points[:, 0]).all(1)
            if valid.any() and origin_only:
                # Existing world points already contain the mounting offset.
                result[valid] = points[valid] + np.asarray(matrix)[:3, 3]
            elif valid.any():
                old_R = Rotation.from_euler("xyz", pose[valid, 3:], degrees=True).as_matrix()
                new_R = Rotation.from_euler("xyz", corrected[valid, 3:], degrees=True).as_matrix()
                local = np.einsum("nji,nkj->nki", old_R, points[valid] - points[valid, :1])
                result[valid] = corrected[valid, None, :3] + np.einsum(
                    "nij,nkj->nki", new_R, local + alignment["offset_local_m"]
                )
            # A partially missing joint cannot mix old-world and reference-world components.
            result[~np.isfinite(result).all(2)] = np.nan
            write(POINT_COLUMNS, result.reshape(-1, 63))
            counts[side] = int(np.isfinite(result).all(2).sum())
            counts[side + "_skipped"] = int((~np.isfinite(result).all(2)).sum())
        elif name == f"observation.{side}_wrist_pose.parquet" or name in alignment["state_fields"]:
            prefix = (
                ""
                if name.endswith("_wrist_pose.parquet")
                else alignment["state_fields"][name][side] + "_"
            )
            cols = [prefix + c for c in POSE_COLUMNS]
            result = transform_poses(values(table, cols), matrix, preserve_rotation=origin_only)
            # Keep source angle arrays (including nulls and non-canonical Euler values) untouched.
            write(cols[:3] if origin_only else cols, result)
            counts[side] = int(np.isfinite(result).all(1).sum())
            counts[side + "_skipped"] = len(result) - counts[side]
    metadata = dict(table.schema.metadata or {})
    if name in alignment["state_fields"]:
        metadata[b"wrist_coordinate_frame"] = alignment["world_frame"].encode()
        metadata[b"wrist_field_mapping"] = json.dumps(alignment["state_fields"][name]).encode()
        metadata[b"wrist_position_unit"] = b"meter"
    else:
        metadata[b"coordinate_frame"] = alignment["world_frame"].encode()
        metadata[b"unit"] = b"meter"
    if alignment.get("extrinsic_model"):
        metadata[b"extrinsic_model"] = alignment["extrinsic_model"].encode()
        metadata[b"source_coordinate_frame"] = b"OpenVR TrackingUniverseStanding world"
    if origin_only:
        metadata[b"orientation_policy"] = b"preserve_source"
        metadata[b"world_axes"] = b"RUB"
    metadata.update(
        reference_frame=str(alignment["reference_frame"]).encode(),
        invalid_value=b"NaN: unavailable pose or joint; no source-world fallback",
        extrinsic_status=b"label-origin and mounting approximation; not independently calibrated",
    )
    return table.replace_schema_metadata(metadata), counts


def enrich_hand_preview(pose, episode, side, frame, K):
    from vive_correct.batch.hand_overlay import HandStream
    from vive_correct.source_data import original_path

    path = original_path(Path(episode) / DATA / f"observation.hand_{side}_points.parquet")
    stream = cached_file("eye_hand_preview", path, lambda: HandStream(path))
    ts = pose["calibration_sample"]["timestamp_s"]
    try:
        points, _ = stream.at(ts, tolerance=0.05)
        # Validate against the hand sample's own synchronized source pose.
        idx = int(np.argmin(abs(stream.times - ts)))
        old = sampled_pose(episode, side, [stream.times[idx]])[0]
        if not np.isfinite(old).all() or not np.isfinite(points[0]).all():
            raise ValueError("手点没有可同步的 Vive 位姿")
        sample = pose["calibration_sample"]
        if sample.get("orientation_policy") == "preserve_source":
            # Project the same existing points that the exporter translates.
            camera_source = np.asarray(sample["T_camera_source"])
            result = points @ camera_source[:3, :3].T + camera_source[:3, 3]
        else:
            R = np.asarray(sample["rotation_camera_corrected"])
            local = (points - points[0]) @ Rotation.from_euler(
                "xyz", old[3:], degrees=True
            ).as_matrix()
            result = (local + OFFSET) @ R.T + pose["origin_camera_m"]
        projection = []
        for p in result:
            projection.append(project_axes(p, np.eye(3), K)["origin"])
        pose["hand_projection"] = projection
        pose["wrist_camera_m"] = result[0].tolist()
    except ValueError as error:
        pose["hand_preview_error"] = str(error)
    return pose


def draw_hand_preview(image, pose, side):
    import cv2
    from vive_correct.batch.hand_overlay import CHAINS

    points = pose.get("hand_projection", [])
    if not points:
        return
    color = (255, 210, 30) if side == "left" else (220, 90, 240)
    for chain in CHAINS:
        for a, b in zip(chain, chain[1:]):
            if points[a] is None or points[b] is None:
                continue
            a, b = [
                tuple(np.rint(np.clip(p, -1e7, 1e7)).astype(int)) for p in (points[a], points[b])
            ]
            visible, a, b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), a, b)
            if visible:
                cv2.line(image, a, b, color, 2)
    for point in points:
        if point is not None and 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1)
