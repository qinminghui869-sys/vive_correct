"""Rebuild L3 hand points using skeleton sides verified by ergonomics channels."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from ego_data.tracking_sync import resample_tracking
from vive_correct.batch.hand_overlay import JOINTS
from vive_correct.batch.report_io import digest

NODES = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24])
OFFSET = np.array([0.0, -0.085, -0.058])


def bend(points):
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.arccos(np.clip(cosine, -1, 1))


def identify_sides(skeleton, ergonomics):
    evidence, mapping = [], {}
    for block in (0, 25):
        scores = {side: [] for side in ("left", "right")}
        for finger, nodes in [("Middle", [11, 12, 13]), ("Ring", [16, 17, 18])]:
            angle = bend(skeleton[:, np.array(nodes) + block])
            for side in scores:
                values = np.asarray(ergonomics[f"{side}_{finger}PIPStretch"])
                valid = np.isfinite(angle) & np.isfinite(values)
                correlation = float("nan")
                if (
                    valid.sum() >= 30
                    and np.std(angle[valid]) > 0.005
                    and np.std(values[valid]) > 1e-6
                ):
                    correlation = float(np.corrcoef(angle[valid], values[valid])[0, 1])
                scores[side].append(correlation)
        means = {side: float(np.mean(values)) for side, values in scores.items()}
        if not all(np.isfinite(list(means.values()))):
            raise ValueError("Insufficient finger motion to identify skeleton sides")
        side = max(means, key=means.get)
        other = "right" if side == "left" else "left"
        if means[side] < 0.95 or means[side] - means[other] < 0.2 or side in mapping:
            raise ValueError(f"Ambiguous skeleton side for block {block}: {means}")
        mapping[side] = block
        evidence.append(dict(block=block, side=side, correlations=scores))
    return mapping, evidence


def world_points(skeleton, pose, block):
    """Retain the existing L3 mounting hypothesis; only skeleton side changes."""
    result = np.full((len(pose), 21, 3), np.nan)
    local = skeleton[:, NODES + block][:, :, [1, 2, 0]] + OFFSET
    valid = np.isfinite(pose).all(1) & np.isfinite(local).all((1, 2))
    rotation = Rotation.from_euler("xyz", pose[valid, 3:], degrees=True).as_matrix()
    result[valid] = np.einsum("nij,nkj->nki", rotation, local[valid]) + pose[valid, None, :3]
    return result


def repair(episode, output):
    episode, output = Path(episode).resolve(), Path(output).resolve()
    if output == episode or episode in output.parents or output in episode.parents:
        raise ValueError("Repair output must be separate from source data")
    data = episode / "data/chunk-000/episode_000000"
    source = data / "observation.hand_skeleton.parquet"
    raw = pq.read_table(source)
    times = raw["timestamp"].to_numpy()
    skeleton = np.column_stack(
        [raw[f"node{i:02d}_{a}"].to_numpy() for i in range(50) for a in "xyz"]
    ).reshape(-1, 50, 3)
    ergonomic_path = data / "observation.hand_pose.parquet"
    ergonomic = pq.read_table(ergonomic_path)
    if not np.array_equal(times, ergonomic["timestamp"].to_numpy()):
        raise ValueError("Skeleton and ergonomics timestamps differ")
    mapping, evidence = identify_sides(skeleton, ergonomic)
    report = dict(
        episode=str(episode),
        mapping=mapping,
        evidence=evidence,
        mounting="Existing L3 hypothesis: tracker xyz = MANUS yzx + [0,-.085,-.058] m; pose xyz degrees; not calibrated",
        source_sha256={source.name: digest(source), ergonomic_path.name: digest(ergonomic_path)},
        files={},
    )
    tables = {}
    for side, block in mapping.items():
        pose_path = data / f"observation.{side}_wrist_pose.parquet"
        table = pq.read_table(pose_path)
        poses = np.column_stack(
            [table[n].to_numpy() for n in ("x", "y", "z", "roll", "pitch", "yaw")]
        )
        pose = resample_tracking((table["timestamp"].to_numpy(), poses), times, pose=True)[1]
        old_path = data / f"observation.hand_{side}_points.parquet"
        old = pq.read_table(old_path)
        if not np.array_equal(times, old["timestamp"].to_numpy()):
            raise ValueError("Stored hand and skeleton timestamps differ")
        columns = [f"{j}_{a}" for j in JOINTS for a in "xyz"]
        stored = np.column_stack([old[n].to_numpy() for n in columns]).reshape(-1, 21, 3)
        # Accept either previous grouping, but reject different mounting conventions.
        errors = []
        for previous in (0, 25):
            predicted = world_points(skeleton, pose, previous)
            valid = np.isfinite(predicted).all((1, 2)) & np.isfinite(stored).all((1, 2))
            errors.append(
                float(np.sqrt(np.mean((predicted[valid] - stored[valid]) ** 2)))
                if valid.any()
                else float("inf")
            )
        if min(errors) > 1e-4:
            raise ValueError(f"{side}: source mounting differs from supported L3 hypothesis")
        points = world_points(skeleton, pose, block).reshape(-1, 63).astype("float32")
        fields = {"timestamp": raw["timestamp"], "episode_index": raw["episode_index"]}
        fields.update({name: pa.array(points[:, i]) for i, name in enumerate(columns)})
        tables[side] = pa.table(fields).replace_schema_metadata(
            {
                b"coordinate_frame": b"OpenVR TrackingUniverseStanding world",
                b"unit": b"meter",
                b"skeleton_side": side.encode(),
                b"raw_block_start": str(block).encode(),
                b"extrinsic_status": b"mounting hypothesis; not independently calibrated",
            }
        )
        report["source_sha256"].update(
            {pose_path.name: digest(pose_path), old_path.name: digest(old_path)}
        )
        report["files"][side] = dict(
            source_block=block,
            previous_block=int(np.argmin(errors)) * 25,
            valid_rows=int(np.isfinite(points).all(1).sum()),
        )
    output.mkdir(parents=True, exist_ok=True)
    for side, table in tables.items():
        path = output / f"observation.hand_{side}_points.parquet"
        pq.write_table(table, path)
        report["files"][side].update(path=str(path), sha256=digest(path))
    (output / "repair.json").write_text(json.dumps(report, indent=2) + "\n")
    return output
