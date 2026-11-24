"""Offline batch entry point: no model transport is imported or constructed."""

import html
import json
from os.path import relpath
from pathlib import Path

from vive_correct.data.dataset import make_run_plan, check_run_plan
from vive_correct.batch.report_io import write_json


def run_cv_dataset(args, dataset, output, selection, initial_mount, detector_config):
    from vive_correct.run_dataset import estimate_dataset

    plan = make_run_plan(args, dataset, selection, initial_mount)
    check_run_plan(output, plan)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_plan.json", plan)
    entries = estimate_dataset(
        dataset,
        output,
        selection,
        args.jobs,
        episodes_file=args.episodes_file,
        detector_config=detector_config,
    )
    render_cv_delivery(output, entries, detector_config.report())
    status = int(any(e["status"] in ("failed", "missing_input") for e in entries))
    if args.stage == "all":
        from vive_correct.cv.cv_features import run_feature_dataset

        status = max(status, run_feature_dataset(args, dataset, output, detector_config))
    return status


def render_cv_delivery(output, entries, detector_report):
    delivery = output / "batch/delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    cards, summaries = [], []
    for entry in entries:
        summary = dict(
            episode=entry["episode"],
            status=entry["status"],
            human_verified=False,
            hands={},
            missing_hands=["left", "right"],
        )
        cards.append("<h2>" + html.escape(entry["relative_episode"]) + "</h2>")
        if entry.get("report"):
            report = json.loads(Path(entry["report"]).read_text())
            for observation in report["observations"]:
                for side, pose in observation["stickers"].items():
                    summary["hands"][side] = dict(
                        frame_index=observation["frame_index"],
                        pose=pose,
                        image=observation["image"],
                        overlay=observation["overlay"],
                    )
                    src = html.escape(relpath(observation["overlay"], delivery), quote=True)
                    diagnostics = pose["cv_diagnostics"]
                    cards.append(
                        f"<p>{side}: frame {observation['frame_index']}; "
                        f"white={diagnostics['white_fraction']:.3f}; "
                        f"score={diagnostics['score']:.3f}</p>"
                        f'<img loading="lazy" width="800" src="{src}">'
                    )
            summary.update(
                missing_hands=sorted({"left", "right"} - summary["hands"].keys()),
                side_assignment=report["side_assignment"],
                tracking_events=report["tracking_events"],
                rejected=report["rejected"],
            )
            summary["status"] = "cv_tracked_unverified" if summary["hands"] else "no_usable_label"
        else:
            summary["error"] = entry.get("error")
        cards.append("<p>Missing: " + html.escape(", ".join(summary["missing_hands"])) + "</p>")
        summaries.append(summary)
    write_json(
        delivery / "summary.json",
        dict(model=None, human_verified=False, detector_config=detector_report, episodes=summaries),
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>CV label observations</title>'
        "<h1>Pure CV label observations</h1><p>Side identity assumes the first reliable "
        "bilateral window is uncrossed. No human verification. Origin: label centre.</p>"
        + '<p><a href="index.html">特征图 / 交付首页</a></p>'
        + "\n".join(cards)
    )
    (delivery / "tracked.html").write_text(page, encoding="utf-8")
    # Keep the feature gallery on resume; the tracked report has its own stable URL.
    if not (delivery / "features_summary.json").exists():
        (delivery / "index.html").write_text(page, encoding="utf-8")
    print(f"Delivery: {delivery / 'index.html'}", flush=True)
    return 0
