"""Run dataset estimation, Qwen review and delivery, or refresh an existing gallery."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from vive_correct.batch.report_io import digest
from vive_correct.batch.bilateral_search import scan_manifest
from vive_correct.batch.workflow import BatchConfig, run as run_batch
from vive_correct.config import batch_settings
from vive_correct.estimate.find_stable_frame import SelectionConfig
from vive_correct.estimate.geometry import VIDEO_KEY
from vive_correct.estimate.sticker_pipeline import run as estimate
from vive_correct.rerun_export import DEFAULT_MOUNT, check_dependencies, export_recordings

DEFAULT_DATASET = Path("samples")
VIDEO_SUFFIX = Path("videos/chunk-000") / VIDEO_KEY / "episode_000000.mp4"
TIMESTAMP_SUFFIX = Path("data/chunk-000/episode_000000") / f"{VIDEO_KEY}.parquet"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


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


def reusable(entry, signature):
    if not entry or entry.get("input_signature") != signature:
        return False
    if entry.get("status") not in ("unreviewed", "no_usable_label"):
        return False
    report_path = Path(entry["report"])
    if not report_path.is_file() or digest(report_path) != entry.get("report_sha256"):
        return False
    report = json.loads(report_path.read_text())
    return all(
        Path(observation[key]).is_file()
        for observation in report["observations"]
        for key in ("image", "overlay")
    )


def render_preview(output, entries):
    cards = []
    for entry in entries:
        details = html.escape(entry.get("error", entry["status"]))
        pictures = []
        if entry.get("report"):
            report = json.loads(Path(entry["report"]).read_text())
            for observation in report["observations"]:
                relative = Path(observation["overlay"]).relative_to(output).as_posix()
                pictures.append(
                    f'<p>帧 {observation["frame_index"]}</p><img loading="lazy" src="{html.escape(relative)}">'
                )
        cards.append(
            f"<article><h2>{entry['index']:02d} · {html.escape(entry['relative_episode'])}</h2>"
            f"<p>{details}</p>{''.join(pictures)}</article>"
        )
    (output / "review_index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>待复核估计结果</title>'
        "<style>body{max-width:1200px;margin:32px auto;font:16px/1.6 system-ui}img{max-width:100%}"
        "article{margin:24px 0;padding:16px;background:#f4f6f8}</style>"
        "<h1>待复核估计结果</h1><p>自动检测未确认标签身份、物理左右手及轴向；此页不是验收结果。</p>"
        + "".join(cards)
        + "</html>"
    )


def _init_worker():
    cv2.setNumThreads(1)


def _estimate_one(episode, folder, config):
    """Run one episode's estimate; returns entry fields (module-level so it pickles)."""
    try:
        report = estimate(episode, folder, config=config)
        report_path = folder / "sticker_pose_report_new.json"
        hints = sorted({side for o in report["observations"] for side in o["stickers"]})
        return dict(
            status="unreviewed" if hints else "no_usable_label",
            report=str(report_path),
            report_sha256=digest(report_path),
            mode=report["mode"],
            detected_side_hints=hints,
            hands=[],
            missing_hands=["left", "right"],
        )
    except (OSError, ValueError, cv2.error) as error:
        return dict(status="failed", error=str(error))


def estimate_dataset(dataset, output, config, jobs=1):
    episodes = discover(dataset)
    if not episodes:
        raise ValueError(f"No episodes found under {dataset}")
    package = Path(__file__).resolve().parent
    code_files = sorted(package.rglob("*.py")) + sorted((package / "config").glob("*.yaml"))
    settings = {
        "dataset": str(dataset),
        "selection": asdict(config),
        "source_hashes": {str(p.relative_to(package)): digest(p) for p in code_files},
    }
    state_path = output / "batch_state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous and previous.get("settings") != settings:
        raise ValueError("Dataset, code or settings changed; use a new --out directory")
    cached = {entry["episode"]: entry for entry in previous.get("entries", [])}
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    done, pending = {}, []
    output.mkdir(parents=True, exist_ok=True)
    for index, episode in enumerate(episodes, 1):
        signature = input_signature(episode)
        entry = {
            "index": index,
            "episode": str(episode),
            "relative_episode": episode.relative_to(dataset).as_posix(),
            "input_signature": signature,
        }
        missing = [name for name, value in signature.items() if value is None]
        if missing:
            entry.update(status="missing_input", error="Missing: " + ", ".join(missing))
        elif reusable(cached.get(str(episode)), signature):
            entry = dict(cached[str(episode)], index=index)
            print(f"[{index}/{len(episodes)}] cached {entry['relative_episode']}", flush=True)
        else:
            pending.append((entry, episode, output / "estimates" / episode.relative_to(dataset)))
            continue
        done[index] = entry

    def record(entry, fields):
        entry.update(fields)
        done[entry["index"]] = entry
        write_json(state_path, {"settings": settings, "entries": [done[i] for i in sorted(done)]})
        print(
            f"[{entry['index']}/{len(episodes)}] {entry['status']} {entry['relative_episode']}",
            flush=True,
        )

    if jobs == 1 or len(pending) <= 1:
        for entry, episode, folder in pending:
            print(
                f"[{entry['index']}/{len(episodes)}] estimate {entry['relative_episode']}",
                flush=True,
            )
            record(entry, _estimate_one(episode, folder, config))
    else:
        with ProcessPoolExecutor(min(jobs, len(pending)), initializer=_init_worker) as pool:
            futures = {
                pool.submit(_estimate_one, episode, folder, config): entry
                for entry, episode, folder in pending
            }
            try:
                for future in as_completed(futures):
                    record(futures[future], future.result())
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    entries = [done[i] for i in sorted(done)]
    write_json(state_path, {"settings": settings, "entries": entries})
    # No automatic detections become accepted physical wrists. Scan both halves.
    manifest = {
        "entries": [e for e in entries if e["status"] in ("unreviewed", "no_usable_label")],
        "identity": "unreviewed; hands intentionally empty",
        "note": "For candidate scanning only; not an accepted reconstruction baseline.",
    }
    write_json(output / "scan_manifest.json", manifest)
    render_preview(output, entries)
    counts = {
        status: sum(e["status"] == status for e in entries)
        for status in ("unreviewed", "no_usable_label", "missing_input", "failed")
    }
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    print(f"Review: {output / 'review_index.html'}", flush=True)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--out", type=Path, required=True, help="Work directory; resume unchanged estimates"
    )
    parser.add_argument(
        "--stage", choices=("all", "estimate", "scan", "finish", "gallery", "rerun"), default="all"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Export full Rerun recordings after repair"
    )
    parser.add_argument(
        "--open-rerun", action="store_true", help="Export and open Rerun recordings"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Replay a bilateral --axis-baseline without model requests",
    )
    defaults = SelectionConfig()
    parser.add_argument(
        "--stride", type=int, default=defaults.stride, help="Estimate sampling stride"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=defaults.window,
        help="Stability window for estimates and automatic review (default: 11)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit initial estimates only; scan still covers the full video",
    )
    parser.add_argument("--scan-stride", type=int, default=batch_settings()["scan"]["stride"])
    parser.add_argument(
        "--jobs",
        type=int,
        default=12,
        help="CPU workers (default: 12); model requests have a separate limit",
    )
    parser.add_argument(
        "--reviewed-baseline", type=Path, help="Full, reviewed baseline manifest for finish"
    )
    parser.add_argument("--review", type=Path, help="Reviewed missing-wrist choices for finish")
    parser.add_argument(
        "--initial-mount", type=Path, help="Initial head-camera transform for all or finish"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Qwen connection JSON; defaults to local config or example config",
    )
    parser.add_argument("--model", help="Override configured Qwen model name")
    parser.add_argument("--model-jobs", type=int, default=3, help="Concurrent model requests")
    parser.add_argument("--coarse-stride", type=int, default=15)
    parser.add_argument("--first-pass", type=int, default=8)
    parser.add_argument(
        "--axis-baseline",
        type=Path,
        help="Hash-verified observation baseline; preserve existing poses and search missing hands",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=24,
        help="Maximum candidates sent for model review per episode; geometry failures excluded",
    )
    parser.add_argument("--review-batch-size", type=int, default=1)
    parser.add_argument("--model-timeout", type=int, default=180)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--review-confidence", type=float, default=0.8)
    args = parser.parse_args(argv)
    dataset, output = args.dataset.resolve(), args.out.resolve()
    try:
        if args.baseline_only and (args.stage != "all" or not args.axis_baseline):
            raise ValueError("--baseline-only requires --stage all and --axis-baseline")
        render = args.rerun or args.open_rerun or args.stage == "rerun"
        initial_mount = (args.initial_mount or DEFAULT_MOUNT).resolve()
        if render:
            if args.stage not in ("all", "finish", "rerun"):
                raise ValueError("--rerun/--open-rerun require stage all, finish or rerun")
            check_dependencies(args.open_rerun)
            if not initial_mount.is_file():
                raise ValueError(f"Initial mounting file missing: {initial_mount}")
        if args.stage == "rerun":
            return export_recordings(output, initial_mount, open_viewer=args.open_rerun)
        if args.stage == "gallery":
            from vive_correct.batch.bilateral_gallery import build

            build(output / "batch")
            print(f"Delivery: {output / 'batch/delivery/index.html'}", flush=True)
            return 0
        if output == dataset or dataset in output.parents or output in dataset.parents:
            raise ValueError("--out must be separate from the dataset directory")
        if args.scan_stride < 1:
            raise ValueError("--scan-stride must be positive")
        cv2.setNumThreads(1)
        if args.stage == "finish":
            if not all((args.reviewed_baseline, args.review, args.initial_mount)):
                raise ValueError(
                    "finish requires --reviewed-baseline, --review and --initial-mount"
                )
            baseline = args.reviewed_baseline.resolve()
            baseline_data = json.loads(baseline.read_text())
            if baseline_data.get("identity", "").startswith("unreviewed"):
                raise ValueError("Automatic scan manifest cannot be used as a reviewed baseline")
            required = {"index", "episode", "report", "report_sha256", "status", "hands"}
            if not baseline_data.get("entries") or any(
                required - entry.keys() for entry in baseline_data["entries"]
            ):
                raise ValueError(
                    "Reviewed baseline entries require index, episode, report, report_sha256, status and hands"
                )
            batch_output = output / "batch"
            if batch_output in baseline.parents:
                raise ValueError("Reviewed baseline must be outside the batch output directory")
            run_batch(
                BatchConfig(
                    baseline, batch_output, args.initial_mount.resolve(), args.review.resolve()
                )
            )
            return (
                export_recordings(output, initial_mount, open_viewer=args.open_rerun)
                if render
                else 0
            )
        if not dataset.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset}")
        config = SelectionConfig(stride=args.stride, window=args.window, max_frames=args.max_frames)
        if args.jobs < 1:
            raise ValueError("--jobs must be positive")
        model_config = None
        if args.stage == "all":
            from vive_correct.estimate.openai_vision import (
                OpenAIConfig,
                load_config,
                _post_chat,
                extract_json,
            )
            from vive_correct.batch.automatic import run_automatic

            package = Path(__file__).resolve().parent
            config_path = args.model_config or (
                package / "openai.local.json"
                if (package / "openai.local.json").exists()
                else package / "openai.example.json"
            )
            if args.baseline_only:
                model_config = OpenAIConfig("baseline-only-no-key", "", "baseline-replay", "none")
            else:
                model_config = load_config(model=args.model, config_path=config_path)
                if model_config is None or "qwen" not in model_config.model.lower():
                    raise ValueError(
                        "all requires a Qwen model and API key; configure --model-config or environment variables"
                    )
            if (
                min(
                    args.review_limit,
                    args.review_batch_size,
                    args.model_timeout,
                    args.model_retries,
                    args.model_jobs,
                    args.first_pass,
                    args.coarse_stride,
                )
                < 1
            ):
                raise ValueError("Review limits, timeout and retries must be positive")
            if args.first_pass > args.review_limit:
                raise ValueError("--first-pass must not exceed --review-limit")
            if args.axis_baseline:
                from vive_correct.batch.selection_policy import load_constraints

                constraints = load_constraints(args.axis_baseline, retain_observations=True)
                if args.baseline_only:
                    for episode in discover(dataset):
                        sides = {
                            c["physical_side"] for c in constraints.get(str(episode.resolve()), [])
                        }
                        if sides != {"left", "right"}:
                            raise ValueError(
                                f"Bilateral baseline missing for {episode}; omit --baseline-only to search with Qwen"
                            )
            if not 0 <= args.review_confidence <= 1:
                raise ValueError("--review-confidence must be between 0 and 1")
            if not initial_mount.is_file():
                raise FileNotFoundError(f"Initial mounting file missing: {initial_mount}")
            # Fail before expensive video scanning when credentials/endpoint are unavailable.
            if not args.baseline_only:
                try:
                    probe = _post_chat(
                        model_config,
                        {
                            "model": model_config.model,
                            "max_tokens": 1024,
                            "enable_thinking": False,
                            "messages": [
                                {"role": "user", "content": 'Return only JSON: {"ok": true}'}
                            ],
                        },
                        timeout=args.model_timeout,
                    )
                    if extract_json(probe).get("ok") is not True:
                        raise ValueError("Unexpected model preflight response")
                except Exception as error:
                    raise ValueError(
                        "Qwen preflight failed: "
                        + str(error).replace(model_config.api_key, "[REDACTED]")
                    ) from error
            print(f"Full pipeline: model={model_config.model}, jobs={args.jobs}", flush=True)
        entries = estimate_dataset(dataset, output, config, args.jobs)
        if args.stage in ("scan", "all"):
            scan_manifest(
                output / "scan_manifest.json",
                output / "search",
                args.coarse_stride if args.stage == "all" else args.scan_stride,
                jobs=args.jobs,
            )
        if args.stage == "all":
            result = run_automatic(
                output,
                model_config,
                initial_mount,
                jobs=args.jobs,
                limit=0 if args.baseline_only else args.review_limit,
                batch_size=args.review_batch_size,
                timeout=args.model_timeout,
                retries=args.model_retries,
                confidence=args.review_confidence,
                window=args.window,
                model_jobs=args.model_jobs,
                axis_baseline=args.axis_baseline,
                first_pass=args.first_pass,
                fine_stride=args.scan_stride,
            )
            failed = int(any(e["status"] in ("failed", "missing_input") for e in entries))
            if render:
                result = max(
                    result, export_recordings(output, initial_mount, open_viewer=args.open_rerun)
                )
            return max(result, failed)
        return 1 if any(e["status"] == "failed" for e in entries) else 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
