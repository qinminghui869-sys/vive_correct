"""Resolve legacy episode folders and LeRobot datasets that hold many episodes.

Legacy episode::

    <episode>/videos/chunk-000/<key>/episode_000000.mp4
    <episode>/data/chunk-000/episode_000000/
    <episode>/calibrations/rectify_params.yaml

KS3 / LeRobot dataset::

    <dataset>/videos/chunk-000/<key>/episode_000000.mp4
    <dataset>/videos/chunk-000/<key>/episode_000001.mp4
    <dataset>/data/chunk-000/episode_000000/
    <dataset>/data/chunk-000/episode_000001/
    <dataset>/calibrations/rectify_params.yaml

A multi-episode dataset is addressed by the data directory of one episode.
A single episode_000000 dataset stays addressed by the dataset root, which is
the layout used by the existing tests and older recordings.
"""

from pathlib import Path

from vive_correct.estimate.geometry import VIDEO_KEY

EPISODE_FILE = "episode_000000"
VIDEO_SUFFIX = Path("videos/chunk-000") / VIDEO_KEY / f"{EPISODE_FILE}.mp4"
TIMESTAMP_SUFFIX = Path("data/chunk-000") / EPISODE_FILE / f"{VIDEO_KEY}.parquet"
CALIBRATION_SUFFIX = Path("calibrations/rectify_params.yaml")
_SKIP = {"videos", "data", "annotation", "audios", "meta", "calibrations", "outputs"}


def _is_data_episode(path):
    path = Path(path)
    return (
        path.name.startswith("episode_")
        and path.parent.name == "chunk-000"
        and path.parent.parent.name == "data"
    )


def is_indexed_episode(episode):
    """True when this path is one episode inside a multi-episode dataset."""
    episode = Path(episode)
    if not _is_data_episode(episode):
        return False
    if episode.name != EPISODE_FILE:
        return True
    return sum(1 for _ in episode.parent.glob("episode_*")) > 1


def dataset_root(episode):
    episode = Path(episode)
    if is_indexed_episode(episode):
        return episode.parents[2]
    return episode


def episode_stem(episode):
    episode = Path(episode)
    if is_indexed_episode(episode):
        return episode.name
    return EPISODE_FILE


def data_directory(episode):
    episode = Path(episode)
    if is_indexed_episode(episode):
        return episode
    return episode / "data/chunk-000" / EPISODE_FILE


def video_file(episode, key=VIDEO_KEY):
    root = dataset_root(episode)
    return root / "videos/chunk-000" / key / f"{episode_stem(episode)}.mp4"


def timestamp_file(episode, key=VIDEO_KEY):
    return data_directory(episode) / f"{key}.parquet"


def calibration_file(episode):
    root = dataset_root(episode)
    per_episode = root / "calibrations/chunk-000" / episode_stem(episode) / "rectify_params.yaml"
    return per_episode if per_episode.is_file() else root / CALIBRATION_SUFFIX


def annotation_dir(episode, key):
    return dataset_root(episode) / "annotation/chunk-000" / key / episode_stem(episode)


def video_dir(root):
    path = Path(root) / "videos/chunk-000" / VIDEO_KEY
    return path if path.is_dir() else None


def iter_dataset_dirs(dataset, depth=5):
    """Yield a dataset root and nested episode folders, without entering media trees."""
    dataset = Path(dataset)
    yield dataset
    if depth <= 0 or not dataset.is_dir():
        return
    try:
        children = list(dataset.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_dir() and child.name not in _SKIP and not child.name.startswith("."):
            yield from iter_dataset_dirs(child, depth - 1)
