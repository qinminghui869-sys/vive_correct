"""Local deterministic point-cloud thinning, without moving hand or depth geometry."""

import numpy as np
from scipy.spatial import cKDTree


def local_hand_cloud_mask(cloud, hand_points, pixel_ids, radius=0.04, minimum_keep=0.2):
    cloud = np.asarray(cloud)
    hands = np.asarray(hand_points).reshape(-1, 3)
    hands = hands[np.isfinite(hands).all(axis=1)]
    if radius <= 0 or not len(hands):
        return np.ones(len(cloud), dtype=bool)
    if not 0 <= minimum_keep <= 1:
        raise ValueError("minimum_keep must be between zero and one")
    distances = cKDTree(hands).query(cloud)[0]
    keep_probability = minimum_keep + (1 - minimum_keep) * np.clip(distances / radius, 0, 1) ** 2
    # Fixed pixel hash: reproducible without altering the untouched far-field cloud.
    ids = np.asarray(pixel_ids, dtype=np.uint64)
    hashed = (ids * np.uint64(1664525) + np.uint64(1013904223)) & np.uint64(0xFFFFFFFF)
    hashed = ((hashed ^ (hashed >> np.uint64(16))) * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    threshold = hashed.astype(np.float64) / 4294967296
    return (distances >= radius) | (threshold < keep_probability)
