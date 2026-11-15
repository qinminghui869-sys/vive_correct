"""Bounded interpolation for positions and xyz Euler tracking streams."""

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_tracking(stream):
    """Sort finite timestamps; preserve conflicting duplicates as missing-data barriers."""
    times, values = (np.asarray(v, dtype=float) for v in stream)
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values):
        raise ValueError("Expected timestamps (N,) and values (N,D)")
    finite = np.isfinite(times)
    report = dict(
        input_rows=len(times),
        invalid_timestamps=int((~finite).sum()),
        reordered=bool(np.any(np.diff(times[finite]) < 0)),
        identical_duplicates=0,
        conflicting_timestamps=0,
    )
    order = np.argsort(times[finite], kind="stable")
    times, values = times[finite][order], values[finite][order]
    unique, starts, counts = np.unique(times, return_index=True, return_counts=True)
    result = values[starts].copy()
    for i in np.flatnonzero(counts > 1):
        group = values[starts[i] : starts[i] + counts[i]]
        if all(np.array_equal(group[0], row, equal_nan=True) for row in group[1:]):
            report["identical_duplicates"] += int(counts[i] - 1)
        else:
            result[i] = np.nan
            report["conflicting_timestamps"] += 1
    report["output_rows"] = len(unique)
    return (unique, result), report


def resample_tracking(stream, targets, *, pose=False, sequence="xyz", degrees=True, max_gap=0.05):
    (times, values), _ = normalize_tracking(stream)
    targets = np.asarray(targets, dtype=float)
    valid = np.isfinite(values).all(axis=1)
    result = np.full((len(targets), values.shape[1]), np.nan)
    if not len(times):
        return targets, result
    if targets.ndim != 1:
        raise ValueError("Target timestamps must be one-dimensional")
    right = np.searchsorted(times, targets)
    inside = np.isfinite(targets) & (targets >= times[0]) & (targets <= times[-1])
    clipped = np.minimum(right, len(times) - 1)
    exact = inside & (times[clipped] == targets)
    matched = exact & valid[clipped]
    result[matched] = values[clipped[matched]]
    indices = np.flatnonzero(inside & ~exact & (right > 0) & (right < len(times)))
    r = right[indices]
    left = r - 1
    usable = valid[left] & valid[r] & (times[r] - times[left] <= max_gap)
    indices, left, r = indices[usable], left[usable], r[usable]
    weight = (targets[indices] - times[left]) / (times[r] - times[left])
    result[indices] = (1 - weight[:, None]) * values[left] + weight[:, None] * values[r]
    if pose and len(indices):
        first = Rotation.from_euler(sequence, values[left, 3:], degrees=degrees)
        last = Rotation.from_euler(sequence, values[r, 3:], degrees=degrees)
        # Relative rotation vectors take the shortest arc, equivalent to pairwise SLERP.
        relative = (first.inv() * last).as_rotvec()
        rotations = first * Rotation.from_rotvec(relative * weight[:, None])
        result[indices, 3:] = rotations.as_euler(sequence, degrees=degrees)
    return targets, result


def sample_nearest(stream, timestamp, tolerance):
    times, values = stream
    if not len(times) or not np.isfinite(timestamp):
        return np.full(values.shape[1], np.nan), float("nan")
    i = np.searchsorted(times, timestamp)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
    j = min(candidates, key=lambda j: abs(times[j] - timestamp))
    delta = float(times[j] - timestamp)
    return (values[j] if abs(delta) <= tolerance else np.full(values.shape[1], np.nan)), delta
