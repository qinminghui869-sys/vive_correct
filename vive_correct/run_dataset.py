"""Run label-center recovery with Astra defaults, or refresh an existing gallery."""

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

from vive_correct.batch.report_io import digest, write_json
from vive_correct.batch.bilateral_search import scan_manifest
from vive_correct.batch.workflow import BatchConfig, run as run_batch
from vive_correct.config import run_settings
from vive_correct.dataset import (
    TIMESTAMP_SUFFIX as TIMESTAMP_SUFFIX,
    VIDEO_SUFFIX as VIDEO_SUFFIX,
    discover as discover,
    input_signature,
    select_episodes,
    source_hashes,
    make_run_plan,
    check_run_plan,
)
from vive_correct.estimate.find_stable_frame import SelectionConfig
from vive_correct.estimate.sticker_pipeline import run as estimate
from vive_correct.rerun_export import DEFAULT_MOUNT, check_dependencies, export_recordings

DEFAULT_DATASET = Path("samples")


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


def _estimate_one(episode, folder, config, detector_config=None):
    """Run one episode's estimate; returns entry fields (module-level so it pickles)."""
    try:
        kwargs = (
            {} if detector_config is None else dict(cv_only=True, detector_config=detector_config)
        )
        report = estimate(episode, folder, config=config, **kwargs)
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


def estimate_dataset(dataset, output, config, jobs=1, episodes_file=None, detector_config=None):
    episodes = select_episodes(dataset, episodes_file)
    if not episodes:
        raise ValueError(f"No episodes found under {dataset}")
    settings = {
        "dataset": str(dataset),
        "selection": asdict(config),
        "source_hashes": source_hashes(),
    }
    if episodes_file is not None:
        settings["selected_episodes"] = [p.relative_to(dataset).as_posix() for p in episodes]
    if detector_config is not None:
        settings["cv_detector"] = detector_config.report()
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
            record(entry, _estimate_one(episode, folder, config, detector_config))
    else:
        with ProcessPoolExecutor(min(jobs, len(pending)), initializer=_init_worker) as pool:
            futures = {
                pool.submit(_estimate_one, episode, folder, config, detector_config): entry
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
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", choices=("astra", "legacy", "cv"), default="astra")
    profile_args, _ = profile_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        preset = run_settings(profile_args.profile)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    parser.add_argument(
        "--profile",
        choices=("astra", "legacy", "cv"),
        default="astra",
        help="cv: fast offline feature selection; astra (default): model review; legacy: API defaults",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the effective plan without model requests or output writes (all/features)",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--episodes-file", type=Path, help="JSON list of episode paths relative to dataset"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Work directory; resume unchanged estimates"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "estimate", "scan", "features", "finish", "gallery", "rerun"),
        default=None,
        help="Default: features for cv, all for model profiles; all includes CV identity tracking",
    )
    parser.add_argument(
        "--feature-search",
        choices=("fast", "full"),
        default="fast",
        help="fast: early stop and bounded scanning; full: original coarse-to-dense recovery",
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
        default=None,
        help="CV identity initialization frames (default: 5); model stability window (default: 11)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit CV sequential frames or model-mode initial estimates",
    )
    parser.add_argument("--scan-stride", type=int)
    parser.add_argument(
        "--jobs",
        type=int,
        help="CPU workers; model requests have a separate limit",
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
        help="API connection JSON; used only with --transport api",
    )
    parser.add_argument("--model", help="Override the exact recognition model name")
    parser.add_argument("--transport", choices=("api", "codex"))
    parser.add_argument("--codex-model", help="Exact model for Codex image review")
    parser.add_argument(
        "--tracker-guided-frames",
        type=int,
        default=None,
        help="Additional evenly spaced Tracker-localized recovery frames (0 disables)",
    )
    parser.add_argument("--model-jobs", type=int, help="Concurrent model requests")
    parser.add_argument("--coarse-stride", type=int)
    parser.add_argument("--first-pass", type=int)
    parser.add_argument(
        "--axis-baseline",
        type=Path,
        help="Hash-verified observation baseline; preserve existing poses and search missing hands",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        help="Maximum candidates sent for model review per episode; geometry failures excluded",
    )
    parser.add_argument("--review-batch-size", type=int)
    parser.add_argument("--model-timeout", type=int)
    parser.add_argument("--model-retries", type=int)
    parser.add_argument("--review-confidence", type=float)
    parser.add_argument(
        "--model-review",
        "--qwen-review",
        dest="qwen_review",
        choices=("on", "off"),
        default="on",
        help="Disable model calls with off; initializes side identity under the uncrossed assumption",
    )
    parser.add_argument(
        "--fixed-candidates",
        action="store_true",
        help="Use only the coarse pool and full candidate budget, without adaptive search/early stop",
    )
    parser.set_defaults(**{k: v for k, v in preset.items() if k != "tracker_guided_frames"})
    parser.add_argument("--cv-config", type=Path, help="Partial overrides for config/cv.yaml")
    parser.add_argument(
        "--coordinate-mode",
        choices=("headband", "eye-reference"),
        default="headband",
        help="headband: rectified left camera, source tracking world (default); "
        "eye-reference: eye video, reference-camera origin with Standing axes (CV only)",
    )
    parser.add_argument(
        "--eye-calibration", type=Path, help="Neon intrinsic JSON for eye-reference mode"
    )
    args = parser.parse_args(argv)
    if args.stage is None:
        args.stage = "features" if args.profile == "cv" else "all"
    if args.coordinate_mode == "eye-reference" and (
        args.profile != "cv" or args.stage not in ("features", "gallery", "rerun")
    ):
        parser.error("eye-reference requires --profile cv and --stage features/gallery/rerun")
    if args.eye_calibration and args.coordinate_mode != "eye-reference":
        parser.error("--eye-calibration requires --coordinate-mode eye-reference")
    if args.profile == "cv":
        args.qwen_review = "off"
        args.tracker_guided_frames = 0
    if args.tracker_guided_frames is None:
        args.tracker_guided_frames = (
            preset["tracker_guided_frames"]
            if args.qwen_review == "on" and not args.fixed_candidates and not args.baseline_only
            else 0
        )
    if args.window is None:
        if args.qwen_review == "off" and (args.profile == "cv" or not args.fixed_candidates):
            from vive_correct.estimate.cv_config import CVConfig

            try:
                args.window = CVConfig.load(args.cv_config).tracking_window
            except (ValueError, OSError) as error:
                parser.error(str(error))
        else:
            args.window = defaults.window
    dataset, output = args.dataset.resolve(), args.out.resolve()
    try:
        if args.dry_run and args.stage not in ("all", "features"):
            raise ValueError("--dry-run requires --stage all or features")
        if args.baseline_only and (args.stage != "all" or not args.axis_baseline):
            raise ValueError("--baseline-only requires --stage all and --axis-baseline")
        render = args.rerun or args.open_rerun or args.stage == "rerun"
        initial_mount = (args.initial_mount or DEFAULT_MOUNT).resolve()
        if render and not args.dry_run:
            if args.stage not in ("all", "finish", "rerun"):
                raise ValueError("--rerun/--open-rerun require stage all, finish or rerun")
            check_dependencies(args.open_rerun)
            if not initial_mount.is_file():
                raise ValueError(f"Initial mounting file missing: {initial_mount}")
        if args.stage == "rerun":
            return export_recordings(output, initial_mount, open_viewer=args.open_rerun)
        if args.stage == "gallery":
            features = output / "batch/delivery/features_summary.json"
            if features.exists():
                from vive_correct.cv_features import render_features

                saved = json.loads(features.read_text())
                render_features(output, saved["episodes"], saved["detector_config"])
                return 0
            cv_summary = output / "batch/delivery/summary.json"
            if cv_summary.exists() and "detector_config" in json.loads(cv_summary.read_text()):
                from vive_correct.cv_dataset import render_cv_delivery

                summary = json.loads(cv_summary.read_text())
                entries = json.loads((output / "batch_state.json").read_text())["entries"]
                return render_cv_delivery(output, entries, summary["detector_config"])
            from vive_correct.batch.bilateral_gallery import build

            build(output / "batch")
            print(f"Delivery: {output / 'batch/delivery/index.html'}", flush=True)
            return 0
        if output == dataset or dataset in output.parents or output in dataset.parents:
            raise ValueError("--out must be separate from the dataset directory")
        if args.scan_stride < 1:
            raise ValueError("--scan-stride must be positive")
        if args.tracker_guided_frames < 0:
            raise ValueError("--tracker-guided-frames must be nonnegative")
        if args.tracker_guided_frames and (
            args.fixed_candidates or args.qwen_review == "off" or args.baseline_only
        ):
            raise ValueError("Tracker-guided recovery requires model review and adaptive search")
        if args.episodes_file:
            if args.coordinate_mode == "eye-reference":
                from vive_correct.camera_input import from_args

                select_episodes(dataset, args.episodes_file, from_args(args))
            else:
                select_episodes(dataset, args.episodes_file)
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
        if args.stage == "features":
            if args.qwen_review != "off":
                raise ValueError("--stage features requires --profile cv or --model-review off")
            from vive_correct.cv_features import run_feature_dataset
            from vive_correct.estimate.cv_config import CVConfig

            return run_feature_dataset(args, dataset, output, CVConfig.load(args.cv_config))
        if args.stage == "all" and args.qwen_review != "off":
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
            if not 0 <= args.review_confidence <= 1:
                raise ValueError("--review-confidence must be between 0 and 1")
        if args.stage == "all" and args.qwen_review == "off":
            if args.coarse_stride < 1:
                raise ValueError("--coarse-stride must be positive")
            if args.axis_baseline or args.baseline_only:
                raise ValueError("--model-review off cannot inherit reviewed baselines")
        if args.cv_config is not None and (
            args.qwen_review != "off" or (args.fixed_candidates and args.profile != "cv")
        ):
            raise ValueError("--cv-config requires the sequential CV profile")
        if args.dry_run:
            print(
                json.dumps(
                    make_run_plan(args, dataset, config, initial_mount),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if (
            args.qwen_review == "off"
            and args.stage in ("all", "estimate", "scan")
            and (args.profile == "cv" or not args.fixed_candidates)
        ):
            if render:
                raise ValueError(
                    "Tracked CV outputs label observations; Rerun repair needs independently verified identities"
                )
            from vive_correct.cv_dataset import run_cv_dataset
            from vive_correct.estimate.cv_config import CVConfig

            return run_cv_dataset(
                args, dataset, output, config, initial_mount, CVConfig.load(args.cv_config)
            )
        if args.cv_config is not None:
            raise ValueError("--cv-config requires --profile cv or --model-review off")
        model_config = None
        transport = None
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
            if args.qwen_review == "off":
                model_config = OpenAIConfig("cv-only-no-key", "", "cv-only", "none")
            elif args.baseline_only:
                model_config = OpenAIConfig("baseline-only-no-key", "", "baseline-replay", "none")
            elif args.transport == "codex":
                from vive_correct.batch.codex_transport import CodexTransport

                model_name = args.model or args.codex_model
                model_config = OpenAIConfig(
                    "codex-account-no-api-key", "codex://chatgpt-account", model_name, "codex_login"
                )
                transport = CodexTransport(
                    output / "codex_requests", model=model_name, timeout=args.model_timeout
                )
            else:
                model_config = load_config(model=args.model, config_path=config_path)
                if model_config is None or "qwen" not in model_config.model.lower():
                    raise ValueError(
                        "all requires a Qwen model and API key; configure --model-config or environment variables"
                    )
            if args.axis_baseline:
                from vive_correct.batch.selection_policy import load_constraints

                constraints = load_constraints(args.axis_baseline, retain_observations=True)
                if args.baseline_only:
                    for episode in select_episodes(dataset, args.episodes_file):
                        sides = {
                            c["physical_side"] for c in constraints.get(str(episode.resolve()), [])
                        }
                        if sides != {"left", "right"}:
                            raise ValueError(
                                f"Bilateral baseline missing for {episode}; omit --baseline-only to search with the model"
                            )
            if not initial_mount.is_file():
                raise FileNotFoundError(f"Initial mounting file missing: {initial_mount}")
            plan = make_run_plan(args, dataset, config, initial_mount, model_config)
            check_run_plan(output, plan)
            # Fail before expensive video scanning when credentials/endpoint are unavailable.
            if not args.baseline_only and args.qwen_review == "on":
                try:
                    probe_fn = transport or (
                        lambda c, b: _post_chat(c, b, timeout=args.model_timeout)
                    )
                    probe = probe_fn(
                        model_config,
                        {
                            "model": model_config.model,
                            "max_tokens": 1024,
                            "enable_thinking": False,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": 'Return only JSON: {"ok": true}'}
                                    ],
                                }
                            ],
                        },
                    )
                    if extract_json(probe).get("ok") is not True:
                        raise ValueError("Unexpected model preflight response")
                except Exception as error:
                    raise ValueError(
                        "Model preflight failed: "
                        + str(error).replace(model_config.api_key, "[REDACTED]")
                    ) from error
            print(f"Full pipeline: model={model_config.model}, jobs={args.jobs}", flush=True)
            write_json(output / "run_plan.json", plan)
        entries = estimate_dataset(
            dataset, output, config, args.jobs, episodes_file=args.episodes_file
        )
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
                qwen_review=args.qwen_review == "on",
                fixed_candidates=args.fixed_candidates,
                transport=transport,
                guided_frames=args.tracker_guided_frames,
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
