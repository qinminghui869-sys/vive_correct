"""Episode discovery, input signatures and resumable run plans."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from vive_correct.batch.report_io import digest
from vive_correct.config import run_settings
from vive_correct.estimate.geometry import VIDEO_KEY

VIDEO_SUFFIX = Path("videos/chunk-000") / VIDEO_KEY / "episode_000000.mp4"
TIMESTAMP_SUFFIX = Path("data/chunk-000/episode_000000") / f"{VIDEO_KEY}.parquet"


def discover(dataset):
    """Include incomplete episodes so missing videos are reported, not hidden."""
    episodes = {
        p.parents[3]
        for p in dataset.rglob(f"{VIDEO_KEY}/episode_000000.mp4")
        if p.relative_to(p.parents[3]) == VIDEO_SUFFIX
    }
    episodes.update(p.parent.parent for p in dataset.rglob("calibrations/rectify_params.yaml"))
    return sorted(episodes)


def input_signature(episode):
    files = (VIDEO_SUFFIX, TIMESTAMP_SUFFIX, Path("calibrations/rectify_params.yaml"))
    result = {}
    for relative in files:
        path = episode / relative
        if path.is_file():
            stat = path.stat()
            result[str(relative)] = [stat.st_size, stat.st_mtime_ns]
        else:
            result[str(relative)] = None
    return result


def select_episodes(dataset, episodes_file=None):
    episodes = discover(dataset)
    if episodes_file is None:
        return episodes
    names = json.loads(Path(episodes_file).read_text())
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(
            "--episodes-file must contain a nonempty JSON list of relative episode paths"
        )
    if len(names) != len(set(names)):
        raise ValueError("Duplicate episodes in --episodes-file")
    available = {p.relative_to(dataset).as_posix(): p for p in episodes}
    if set(names) - available.keys():
        raise ValueError("Unknown episodes: " + ", ".join(sorted(set(names) - available.keys())))
    return [available[name] for name in names]


def source_hashes():
    package = Path(__file__).resolve().parent
    files = sorted(package.rglob("*.py")) + sorted((package / "config").glob("*.yaml"))
    return {str(p.relative_to(package)): digest(p) for p in files}


def make_run_plan(args, dataset, selection, initial_mount, model_config=None):
    """Credential-free audit record of effective settings, inputs and source version."""
    episodes = select_episodes(dataset, args.episodes_file)
    if not episodes:
        raise ValueError(f"No episodes found under {dataset}")
    parameters = {key: getattr(args, key) for key in run_settings(args.profile)}
    parameters.update(
        model_review=args.qwen_review == "on",
        fixed_candidates=args.fixed_candidates,
        baseline_only=args.baseline_only,
        auto_rescan=not args.fixed_candidates and not args.baseline_only,
    )
    if model_config:
        model = model_config.model
    elif args.baseline_only:
        model = "baseline-replay"
    elif args.qwen_review == "off":
        model = "cv-only"
    else:
        model = args.model or (args.codex_model if args.transport == "codex" else None)
    if args.qwen_review == "off" and (args.profile == "cv" or not args.fixed_candidates):
        from vive_correct.estimate.cv_config import CVConfig

        parameters["cv_detector"] = CVConfig.load(getattr(args, "cv_config", None)).report()
        parameters["auto_rescan"] = False
        parameters["scan_mode"] = "sequential_identity_tracking"
    references = {}
    for key, path in (("initial_mount", initial_mount), ("axis_baseline", args.axis_baseline)):
        if path is not None:
            references[key] = {"path": str(path.resolve()), "sha256": digest(path)}
    return {
        "schema_version": 1,
        "profile": args.profile,
        "dataset": str(dataset),
        "selection": asdict(selection),
        "parameters": parameters,
        "model": model,
        "model_resolution": "resolved" if model is not None else "API config resolved at execution",
        # Endpoint identity affects cache validity; never persist credentials or URL secrets.
        "endpoint_sha256": hashlib.sha256(model_config.base_url.encode()).hexdigest()
        if model_config
        else None,
        "references": references,
        "source_hashes": source_hashes(),
        "episodes": [
            {
                "relative_episode": p.relative_to(dataset).as_posix(),
                "input_signature": input_signature(p),
            }
            for p in episodes
        ],
        "scope": "label_center; model review is not human verification or measured accuracy",
    }


def check_run_plan(output, plan):
    path = output / "run_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "Run inputs, model, code or review settings changed; use a new --out directory"
            )
    elif (output / "automatic_state.json").exists() or (
        output / "batch/run_metadata.json"
    ).exists():
        raise ValueError("Existing review has no frozen run plan; use a new --out directory")
    state_path = output / "batch_state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        settings = previous.get("settings", {})
        if any(settings.get(key) != plan[key] for key in ("dataset", "selection", "source_hashes")):
            raise ValueError(
                "Existing estimates use different settings or code; use a new --out directory"
            )
        current = {e["relative_episode"]: e["input_signature"] for e in plan["episodes"]}
        selected = settings.get("selected_episodes")
        if selected is not None and selected != list(current):
            raise ValueError("Existing estimate selection changed; use a new --out directory")
        for entry in previous.get("entries", []):
            if current.get(entry["relative_episode"]) != entry.get("input_signature"):
                raise ValueError("Existing estimate inputs changed; use a new --out directory")
