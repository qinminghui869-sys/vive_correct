"""Shared position-only correction for wrist, state/action and existing hand points."""

import numpy as np
import pyarrow as pa

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
