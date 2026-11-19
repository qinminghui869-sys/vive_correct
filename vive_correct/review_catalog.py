"""Append-only episode identifiers shared by live review and offline commands."""

from pathlib import Path

from vive_correct.review_io import atomic_json, output_lock, read_json


def validate_path(output, path):
    if not path.resolve().is_relative_to(output):
        raise ValueError("特征文件必须位于输出目录内")
    report = read_json(path)
    episode = Path(report["episode"]).resolve()
    if episode == output or episode in output.parents or output in episode.parents:
        raise ValueError("审核输出目录必须与源数据分离")
    return report


def published(output, path):
    report = validate_path(output, path)
    if report.get("status") == "skipped":
        return False
    assets = list(report.get("feature_images", {}).values())
    if report.get("reference_frame_image"):
        assets.append(report["reference_frame_image"])
    for item in report.get("features", {}).values():
        assets += [item[k] for k in ("frame_image", "clean_frame_image") if k in item]
    if report.get("publication_version"):
        assets += [str(path.with_name("candidate_pool.json"))]
    return all(Path(p).is_file() for p in assets)


def refresh_catalog(output):
    output = Path(output).resolve()
    # Never turn a dataset directory into an empty review workspace.
    if (output / "data/chunk-000").exists() or (output / "videos/chunk-000").exists():
        raise ValueError("审核输出目录必须与源数据分离")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output, name=".review_catalog.lock", blocking=True):
        registry = output / "review_catalog.json"
        names = read_json(registry)["paths"] if registry.exists() else []
        candidates = sorted((output / "features").rglob("features.json"))
        if not candidates:
            summary = output / "batch/delivery/features_summary.json"
            if summary.is_file():
                candidates = [
                    Path(r["human_review_file"]).resolve().parent / "features.json"
                    for r in read_json(summary)["episodes"]
                ]
        errors = []
        for path in candidates:
            if not path.resolve().is_relative_to(output):
                raise ValueError("特征文件必须位于输出目录内")
            name = path.relative_to(output).as_posix()
            if name in names:
                continue
            try:
                if published(output, path):
                    names.append(name)
            except (OSError, ValueError, KeyError) as error:
                errors.append(dict(path=str(path), error=str(error)))
        # Keep missing entries as tombstones so existing IDs never move.
        paths = [output / n for n in names]
        for path in paths:
            if not path.resolve().is_relative_to(output):
                raise ValueError("特征文件必须位于输出目录内")
        if not registry.exists() or read_json(registry)["paths"] != names:
            atomic_json(registry, dict(schema_version=1, paths=names), durable=True)
        return paths, errors
