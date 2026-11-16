"""Export reviewed bilateral origins through the existing Rerun renderer."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from vive_correct.batch.report_io import digest
from vive_correct.viewer import launch
from vive_correct.hand_repair import prepare_hands

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOUNT = ROOT / "calibrations/left_camera_to_head_vive.json"
DEPTH_KEY = "observation.images.depth_headband_left_rectified"


def check_dependencies(open_viewer=False):
    for module in ("av", "rerun"):
        if importlib.util.find_spec(module) is None:
            raise ValueError(
                "Rerun export requires: python3 -m pip install -r ego_data/requirements.txt"
            )
    if open_viewer and shutil.which("rerun") is None:
        raise ValueError("Rerun viewer executable is not on PATH")


def export_recordings(work, calibration, *, open_viewer=False):
    import pyarrow.parquet as pq
    from ego_data.vive_origin_alignment import build_alignment

    check_dependencies(open_viewer)
    manifest_path = work / "batch/results/manifest.json"
    entries = json.loads(manifest_path.read_text())["entries"]
    if not entries:
        raise ValueError("No reviewed episodes to export")
    destination = work / "rerun"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"entries": []}
    index = destination / "manifest.json"
    index.write_text(json.dumps(summary, indent=2) + "\n")
    for entry in entries:
        episode = Path(entry["episode"]).resolve()
        folder = destination / f"{entry['index']:02d}_{episode.name}"
        folder.mkdir(parents=True, exist_ok=True)
        row = {"episode": str(episode), "status": "failed"}
        try:
            report = Path(entry["report"])
            if not report.is_absolute():
                report = manifest_path.parent / report
            if digest(report) != entry["report_sha256"]:
                raise ValueError("Reviewed report hash mismatch")
            alignment = build_alignment(episode, report, calibration, refine_depth=True)
            alignment_path = folder / "wrist_alignment.json"
            alignment_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2) + "\n")
            hands, hand_status = prepare_hands(episode, folder / "hands")
            row["hands"] = hand_status
            hand_status_path = folder / "hand_status.json"
            hand_status_path.write_text(
                json.dumps(hand_status, ensure_ascii=False, indent=2) + "\n"
            )
            time_path = episode / "data/chunk-000/episode_000000" / f"{DEPTH_KEY}.parquet"
            if not time_path.is_file():
                from vive_correct.estimate.geometry import VIDEO_KEY

                time_path = episode / "data/chunk-000/episode_000000" / f"{VIDEO_KEY}.parquet"
            times = pq.read_table(time_path)["timestamp"]
            if not len(times):
                raise ValueError("Export timeline is empty")
            # Publish a recording only after the renderer finishes successfully.
            pending = folder / "recording.partial.rrd"
            log = folder / "render.log"
            command = [
                sys.executable,
                str(ROOT / "ego_data/render_rerun.py"),
                "--dataset",
                str(episode),
                "--start",
                "0",
                "--end",
                str(len(times) - 1),
                "--calibration",
                str(calibration),
                "--wrist-calibration",
                str(alignment_path),
                "--pose-euler-sequence",
                alignment["pose_euler_sequence"],
                "--interpolate-tracking",
                "--skip-missing-head",
                "--show-uncorrected-wrists",
                "--show-uncorrected-hands",
                "--hide-axes",
                "--application-id",
                episode.name + "_vive_correct",
                "--output",
                str(pending),
            ]
            command += ["--hand-status", str(hand_status_path)]
            if hands:
                command += ["--hand-points-dir", str(hands)]
            if not alignment["pose_degrees"]:
                command.append("--pose-radians")
            print(f"Render: {episode.name} (0–{len(times) - 1}); log: {log}", flush=True)
            with log.open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            stats = json.loads(pending.with_suffix(".json").read_text())
            if stats["frames"] < 1 or stats["frames"] + len(
                stats["skipped_missing_head_frames"]
            ) != len(times):
                raise ValueError("Incomplete Rerun frame coverage")
            recording = folder / "recording.rrd"
            pending.replace(recording)
            pending.with_suffix(".json").replace(recording.with_suffix(".json"))
            row.update(
                status="complete",
                recording=str(recording),
                display_mode=stats.get("mode", "registered_depth"),
                vive_correction_displayed=stats.get("mode") != "rgb_manus_only",
                frames=stats["frames"],
                skipped_frames=stats["skipped_missing_head_frames"],
                alignment=str(alignment_path),
            )
        except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
            row["error"] = str(error)
        summary["entries"].append(row)
        index.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"Rerun {row['status']}: {row.get('recording', row.get('error'))}", flush=True)
    recordings = [row["recording"] for row in summary["entries"] if row["status"] == "complete"]
    if open_viewer and recordings:
        launch(recordings, log_path=destination / "viewer.log")
    return int(len(recordings) != len(entries))
