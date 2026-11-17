"""Shared position-only correction for wrist, state/action and existing hand points."""

import numpy as np
import pyarrow as pa
from pathlib import Path
from scipy.spatial.transform import Rotation

from vive_correct.batch.hand_overlay import JOINTS
from vive_correct.source_data import FILES


def position_groups(name: str):
    """Return physical side and xyz fields for one supported source filename."""
    if name not in FILES:
        raise ValueError(f"不支持的重建文件：{name}")
    if name.endswith("_wrist_pose.parquet"):
        side = "left" if "left" in name else "right"
        return [(side, list("xyz"))]
    if name in ("action.parquet", "observation.state.parquet"):
        return [(s, [f"{s}_wrist_{a}" for a in "xyz"]) for s in ("left", "right")]
    side = "left" if "left" in name else "right"
    return [(side, [f"{joint}_{a}" for a in "xyz"]) for joint in JOINTS]


def translate(table: pa.Table, name: str, alignment: dict) -> tuple[pa.Table, dict]:
    """Translate complete finite xyz groups; leave rotations, nulls and metadata intact."""
    if any(w["model"] == "wrist_local_rigid" for w in alignment["wrists"].values()):
        return rigid_correct(table, name, alignment)
    counts = {}
    for side, columns in position_groups(name):
        for column in columns:
            if column not in table.column_names or not pa.types.is_floating(
                table.schema.field(column).type
            ):
                raise ValueError(f"{name}: 缺少浮点坐标字段 {column}")
        xyz = np.column_stack([table[c].to_numpy() for c in columns]).astype(float)
        valid = np.isfinite(xyz).all(axis=1)
        correction = alignment["wrists"][side]
        offset = np.asarray(correction["offset_m"], dtype=float)
        if correction["model"] != "world" or offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("仅支持有限的固定世界平移")
        xyz[valid] += offset
        counts[columns[0].removesuffix("_x")] = int(valid.sum())
        for i, column in enumerate(columns):
            field = table.schema.field(column)
            values = pa.array(xyz[:, i], type=field.type, mask=table[column].is_null().to_numpy())
            table = table.set_column(table.schema.get_field_index(column), field, values)
    return table, counts


def rigid_parameters(correction):
    offset = np.asarray(correction["offset_local_m"], dtype=float)
    rotation = np.asarray(correction["rotation_local"], dtype=float)
    if (
        correction["model"] != "wrist_local_rigid"
        or offset.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(offset).all()
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
    ):
        raise ValueError("无效的本地刚性校准")
    return offset, rotation


def rigid_correct(table, name, alignment):
    """Transform source poses and world landmarks once, preserving incomplete groups."""
    from ego_data.tracking_sync import resample_tracking
    from ego_data.vive_origin_alignment import pose_stream
    from vive_correct.source_data import DATA

    sequence, degrees = alignment["pose_euler_sequence"], alignment["pose_degrees"]
    counts, poses = {}, {}
    hand = name.startswith("observation.hand_")

    def read(columns):
        for column in columns:
            if column not in table.column_names or not pa.types.is_floating(
                table.schema.field(column).type
            ):
                raise ValueError(f"{name}: 缺少浮点坐标字段 {column}")
        return np.column_stack([table[c].to_numpy() for c in columns]).astype(float)

    def write(columns, values):
        nonlocal table
        for i, column in enumerate(columns):
            field = table.schema.field(column)
            table = table.set_column(
                table.schema.get_field_index(column),
                field,
                pa.array(values[:, i], type=field.type, mask=table[column].is_null().to_numpy()),
            )

    for side, columns in position_groups(name):
        xyz = read(columns)
        offset, correction = rigid_parameters(alignment["wrists"][side])
        if hand:
            if side not in poses:
                if "timestamp" not in table.column_names:
                    raise ValueError(f"{name}: 缺少 timestamp")
                stream = pose_stream(
                    Path(alignment["episode"]) / DATA / f"observation.{side}_wrist_pose.parquet"
                )
                poses[side] = resample_tracking(
                    stream,
                    table["timestamp"].to_numpy(),
                    pose=True,
                    sequence=sequence,
                    degrees=degrees,
                )[1]
            pose = poses[side]
        else:
            prefix = "" if name.endswith("_wrist_pose.parquet") else f"{side}_wrist_"
            angle_columns = [prefix + a for a in ("roll", "pitch", "yaw")]
            angles = read(angle_columns)
            pose = np.column_stack([xyz, angles])
        valid_pose = np.isfinite(pose).all(axis=1)
        valid = valid_pose & np.isfinite(xyz).all(axis=1)
        rotation = (
            Rotation.from_euler(sequence, pose[valid, 3:], degrees=degrees).as_matrix()
            if valid.any()
            else np.empty((0, 3, 3))
        )
        translated = pose[valid, :3] + np.einsum("nij,j->ni", rotation, offset)
        if hand:
            local = np.einsum("nji,nj->ni", rotation, xyz[valid] - pose[valid, :3])
            xyz[valid] = translated + np.einsum("nij,nj->ni", rotation @ correction, local)
        else:
            xyz[valid] = translated
            if valid.any() and not np.allclose(correction, np.eye(3), atol=1e-12):
                angles[valid] = Rotation.from_matrix(rotation @ correction).as_euler(
                    sequence, degrees=degrees
                )
                write(angle_columns, angles)
        write(columns, xyz)
        key = columns[0].removesuffix("_x")
        counts[key] = int(valid.sum())
        counts[key + "_skipped"] = int((~valid).sum())
    return table, counts


def verify_table(actual: pa.Table, expected: pa.Table) -> None:
    """Validate a written Parquet, including schema metadata and non-target fields."""
    if len(actual) != len(expected) or not actual.schema.equals(
        expected.schema, check_metadata=True
    ):
        raise ValueError("重建文件行数或 schema 不一致")
    for name in actual.column_names:
        a, b = actual[name], expected[name]
        if not a.is_null().equals(b.is_null()):
            raise ValueError(f"{name}: 空值掩码变化")
        if pa.types.is_floating(a.type):
            if not np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True):
                raise ValueError(f"{name}: 重建数值校验失败")
        elif not a.equals(b):
            raise ValueError(f"{name}: 非目标字段发生变化")
