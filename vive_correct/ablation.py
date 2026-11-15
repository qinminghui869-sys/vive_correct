"""Paired model on/off experiment using an explicit, frozen CV proposal pool."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from ego_data.vive_origin_alignment import build_alignment

from vive_correct.batch.automatic import run_automatic
from vive_correct.batch.report_io import digest, write_json
from vive_correct.estimate.openai_vision import OpenAIConfig, load_config
from vive_correct.dataset import discover, input_signature
from vive_correct.rerun_export import DEFAULT_MOUNT


def prepare(source, output, dataset):
    state = json.loads((source / "batch_state.json").read_text())
    old = {str(Path(e["episode"]).resolve()): e for e in state["entries"]}
    entries, inventory = [], []
    for index, episode in enumerate(discover(dataset), 1):
        episode = episode.resolve()
        signature = input_signature(episode)
        missing = [name for name, value in signature.items() if value is None]
        row = dict(index=index, episode=str(episode), missing_inputs=missing)
        inventory.append(row)
        if missing:
            row["status"] = "missing_input"
            continue
        previous = old.get(str(episode))
        if previous is None or previous["input_signature"] != signature:
            raise ValueError(f"Frozen pool has no matching input signature: {episode}")
        folder = source / "search" / f"{previous['index']:02d}_{episode.name}"
        pool = folder / "candidates.json"
        record = json.loads((folder / "scan_complete.json").read_text())
        if digest(pool) != record["candidate_pool_sha256"]:
            raise ValueError(f"Frozen pool hash mismatch: {pool}")
        if Path(json.loads(pool.read_text())["episode"]).resolve() != episode:
            raise ValueError(f"Frozen pool episode mismatch: {pool}")
        row.update(status="ready", pool_sha256=digest(pool), source_pool=str(pool))
        for mode in ("off", "on"):
            work = output / mode
            target = work / "search" / f"{index:02d}_{episode.name}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("candidates.json", "scan_complete.json"):
                shutil.copy2(folder / name, target / name)
        entries.append(dict(previous, index=index))
    for mode in ("off", "on"):
        write_json(output / mode / "scan_manifest.json", {"entries": entries})
    write_json(output / "inventory.json", {"entries": inventory})
    return inventory


def summarize(output):
    inventory = json.loads((output / "inventory.json").read_text())["entries"]
    modes, rows = {}, {e["index"]: dict(e) for e in inventory}
    for mode in ("off", "on"):
        manifest = output / mode / "batch/results/manifest.json"
        if not manifest.exists():
            continue
        entries = json.loads(manifest.read_text())["entries"]
        identity_path = output / mode / "model_identity.json"
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
        rms, stats = (
            [],
            dict(
                model=identity.get("model"),
                transport=identity.get("transport"),
                total_samples=len(inventory),
                missing_input=sum(bool(row["missing_inputs"]) for row in inventory),
                processed=len(entries),
                bilateral=0,
                single_hand=0,
                no_observation=0,
                errors=0,
                selected_hands=0,
                stable_hands=0,
                model_batches=0,
                localization_requests=0,
                evaluated_candidates=0,
                alignment_ready=0,
                service_rejected_batches=0,
                service_rejected_candidates=0,
                incomplete_review_samples=0,
            ),
        )
        for entry in entries:
            report = json.loads(Path(entry["report"]).read_text())
            hands = {}
            for obs in report["observations"]:
                for side, pose in obs["stickers"].items():
                    hands[side] = dict(
                        frame=obs["frame_index"],
                        timestamp_s=obs["timestamp_s"],
                        origin_camera_m=pose["origin_camera_m"],
                        rms_px=pose["rms_px"],
                        stable=pose.get("stability", {}).get("passed", False),
                        candidate_id=pose.get("source_candidate_id"),
                        image=obs["overlay"],
                    )
                    rms.append(pose["rms_px"])
                    stats["stable_hands"] += bool(hands[side]["stable"])
            stats["selected_hands"] += len(hands)
            stats[{2: "bilateral", 1: "single_hand", 0: "no_observation"}[len(hands)]] += 1
            stats["errors"] += entry["status"] == "review_error"
            if mode == "on" and not stats["model"]:
                stats["model"] = report.get("model")
            review = report.get("model_review", {})
            rejected = [
                w for w in review.get("warnings", []) if w.get("kind") == "content_rejected"
            ]
            stats["service_rejected_batches"] += len(rejected)
            stats["service_rejected_candidates"] += sum(
                len(w.get("candidate_ids", [])) for w in rejected
            )
            review_complete = (
                not review.get("errors") and not rejected and entry["status"] != "review_error"
            )
            stats["incomplete_review_samples"] += not review_complete
            for key in ("localization_requests", "evaluated_candidates"):
                stats[key] += review.get(key, 0)
            stats["model_batches"] += review.get("requested_batches", 0)
            rows[entry["index"]][mode] = dict(
                status=entry["status"],
                hands=hands,
                errors=review.get("errors", []),
                service_rejections=rejected,
                review_complete=review_complete,
                report=entry["report"],
                evaluated_candidates=review.get("evaluated_candidates", 0),
            )
            review_path = (
                output / mode / "qwen_reviews" / f"{entry['index']:02d}" / "review_summary.json"
            )
            if review_path.exists():
                detail = json.loads(review_path.read_text())
                ids = {d["candidate_id"] for d in detail["decisions"] if "is_label" in d}
                for error in detail.get("errors", []) + detail.get("warnings", []):
                    ids.update(error.get("candidate_ids", []))
                rows[entry["index"]][mode]["evaluated_candidate_ids"] = sorted(ids)
            if len(hands) == 2:
                try:
                    alignment = build_alignment(
                        Path(entry["episode"]), Path(entry["report"]), DEFAULT_MOUNT
                    )
                    write_json(
                        output / mode / "alignments" / f"{entry['index']:02d}.json", alignment
                    )
                    rows[entry["index"]][mode]["world_offsets_m"] = {
                        side: value["offset_m"] for side, value in alignment["wrists"].items()
                    }
                    stats["alignment_ready"] += 1
                except (ValueError, OSError, KeyError) as error:
                    rows[entry["index"]][mode]["alignment_error"] = str(error)
        stats["rms_median_px"] = float(np.median(rms)) if rms else None
        modes[mode] = stats
    for row in rows.values():
        if "on" not in row or "off" not in row:
            continue
        row["comparison"] = {}
        row["equal_candidate_count"] = (
            row["off"]["evaluated_candidates"] == row["on"]["evaluated_candidates"]
        )
        row["equal_candidate_ids"] = (
            row["off"]["evaluated_candidate_ids"] == row["on"]["evaluated_candidate_ids"]
            if "evaluated_candidate_ids" in row["off"] and "evaluated_candidate_ids" in row["on"]
            else None
        )
        off_offsets = row["off"].get("world_offsets_m", {})
        on_offsets = row["on"].get("world_offsets_m", {})
        row["world_correction_difference_m"] = {
            side: float(np.linalg.norm(np.asarray(on_offsets[side]) - off_offsets[side]))
            for side in off_offsets.keys() & on_offsets.keys()
        }
        for side in ("left", "right"):
            a, b = row["off"]["hands"].get(side), row["on"]["hands"].get(side)
            if a and b:
                row["comparison"][side] = dict(
                    same_frame=a["frame"] == b["frame"],
                    same_candidate=a["candidate_id"] == b["candidate_id"],
                    same_frame_origin_delta_m=float(
                        np.linalg.norm(np.array(a["origin_camera_m"]) - b["origin_camera_m"])
                    )
                    if a["frame"] == b["frame"]
                    else None,
                )
    result = dict(
        summary=modes,
        entries=list(rows.values()),
        interpretation="Coverage and geometry diagnostics, not accuracy; no independent ground truth. Position differences only compared at the same frame.",
    )
    paired = [
        r
        for r in rows.values()
        if "on" in r and "off" in r and r["on"]["review_complete"] and r["off"]["review_complete"]
    ]
    if set(modes) == {"off", "on"}:
        result["complete_review_subset"] = dict(
            sample_count=len(paired),
            indices=[r["index"] for r in paired],
            summary={
                mode: dict(
                    bilateral=sum(len(r[mode]["hands"]) == 2 for r in paired),
                    single_hand=sum(len(r[mode]["hands"]) == 1 for r in paired),
                    no_observation=sum(not r[mode]["hands"] for r in paired),
                )
                for mode in ("off", "on")
            },
        )
    write_json(output / "comparison.json", result)
    from vive_correct.ablation_report import render

    render(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("samples"))
    parser.add_argument("--source-work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("vive_correct/openai.local.json"))
    parser.add_argument("--transport", choices=("api", "codex"), default="api")
    parser.add_argument("--codex-model", default="gpt-6.1-sol")
    parser.add_argument("--review-limit", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--model-jobs", type=int, default=3)
    parser.add_argument("--mode", choices=("both", "off", "on", "summarize"), default="both")
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if args.mode == "summarize":
        print(json.dumps(summarize(output)["summary"], indent=2))
        return 0
    if min(args.review_limit, args.jobs, args.model_jobs) < 1:
        parser.error("limits and worker counts must be positive")
    if output.exists() and (output / "experiment.json").exists():
        parser.error(
            "Use a new experiment directory; summarize is read-only apart from comparison.json"
        )
    cv2.setNumThreads(1)
    inventory = prepare(args.source_work.resolve(), output, args.dataset.resolve())
    write_json(
        output / "experiment.json",
        dict(
            source_work=str(args.source_work.resolve()),
            dataset=str(args.dataset.resolve()),
            candidate_source="historical frozen OpenCV scan; current geometry refitted in both arms",
            review_limit=args.review_limit,
            window=11,
            confidence=0.8,
            batch_size=4,
            adaptive_search=False,
            inherited_baselines=False,
            missing_samples=sum(bool(e["missing_inputs"]) for e in inventory),
            source_hashes={str(p): digest(p) for p in Path("vive_correct").rglob("*.py")},
        ),
    )
    transport = None
    if args.transport == "codex":
        from vive_correct.batch.codex_transport import CodexTransport

        transport = CodexTransport(output / "codex_requests", model=args.codex_model)
    errors = 0
    for mode in ("off", "on") if args.mode == "both" else (args.mode,):
        config = (
            (
                OpenAIConfig(
                    "codex-account-no-api-key",
                    "codex://chatgpt-account",
                    args.codex_model,
                    "codex_login",
                )
                if args.transport == "codex"
                else load_config(config_path=args.model_config, environ={}, dotenv_paths=[])
            )
            if mode == "on"
            else OpenAIConfig("cv-only-no-key", "", "cv-only", "none")
        )
        if config is None:
            raise ValueError("Model configuration missing")
        write_json(
            output / mode / "model_identity.json",
            dict(
                model=config.model,
                endpoint=config.base_url,
                transport=args.transport if mode == "on" else "none",
                unsupported_api_controls=[
                    "temperature",
                    "max_tokens",
                    "enable_thinking",
                    "response_format",
                ]
                if mode == "on" and args.transport == "codex"
                else [],
            ),
        )
        start = time.monotonic()
        errors = max(
            errors,
            run_automatic(
                output / mode,
                config,
                DEFAULT_MOUNT,
                transport=transport if mode == "on" else None,
                jobs=args.jobs,
                model_jobs=args.model_jobs,
                limit=args.review_limit,
                batch_size=4,
                first_pass=min(8, args.review_limit),
                qwen_review=mode == "on",
                fixed_candidates=True,
            ),
        )
        write_json(output / mode / "timing.json", dict(elapsed_seconds=time.monotonic() - start))
        summarize(output)
    print(json.dumps(summarize(output)["summary"], indent=2))
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
