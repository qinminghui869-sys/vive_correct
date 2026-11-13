"""Bounded interpolation for positions and xyz Euler tracking streams."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample_tracking(stream, targets, *, pose=False, sequence="xyz", degrees=True, max_gap=0.05):
    times, values = (np.asarray(v, dtype=float) for v in stream)
    targets = np.asarray(targets, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) < 0) or not np.isfinite(times).all():
        raise ValueError("Tracking timestamps must be finite and sorted")
    unique, counts = np.unique(times, return_counts=True)
    valid = np.isfinite(values).all(axis=1) & ~np.isin(times, unique[counts > 1])
    result = np.full((len(targets), values.shape[1]), np.nan)
    for i, t in enumerate(targets):
        if not np.isfinite(t) or not times[0] <= t <= times[-1]:
            continue
        right = int(np.searchsorted(times, t))
        if times[right] == t:
            if valid[right]:
                result[i] = values[right]
            continue
        left = right - 1
        if not valid[left : right + 1].all() or times[right] - times[left] > max_gap:
            continue
        weight = (t - times[left]) / (times[right] - times[left])
        result[i] = (1 - weight) * values[left] + weight * values[right]
        if pose:
            rotations = Rotation.from_euler(sequence, values[[left, right], 3:], degrees=degrees)
            result[i, 3:] = Slerp(times[[left, right]], rotations)([t]).as_euler(
                sequence, degrees=degrees
            )[0]
    return targets, result
