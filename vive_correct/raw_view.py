"""Export original Parquet hand points alongside the registered depth cloud."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from vive_correct.rerun_export import ROOT, DEFAULT_MOUNT, DEPTH_KEY, check_dependencies
from vive_correct.viewer import launch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial-mount", type=Path, default=DEFAULT_MOUNT)
    parser.add_argument("--frames", type=int, nargs="+", help="Export selected zero-based frames")
    parser.add_argument(
        "--rgb-only", action="store_true", help="Show RGB and original MANUS even when depth exists"
    )
    parser.add_argument(
        "--manus",
        action="store_true",
        help="Include original 50 MANUS nodes in local-coordinate views",
    )
    parser.add_argument("--open-rerun", action="store_true")
    parser.add_argument(
        "--show-labels", action="store_true", help="Show joint numbers; may overlap in dense views"
    )
    parser.add_argument(
        "--repair-hands",
        action="store_true",
        help="Rebuild skeleton sides after ergonomics verification; keep source files",
    )
    args = parser.parse_args(argv)
    episode, output = args.dataset.resolve(), args.out.resolve()
    if output == episode or episode in output.parents or output in episode.parents:
        parser.error("Output must be separate from the dataset")
    check_dependencies(args.open_rerun)
    timestamp_path = episode / "data/chunk-000/episode_000000" / f"{DEPTH_KEY}.parquet"
    from ego_data.manus_view import depth_available, rgb_source

    if args.rgb_only or not depth_available(episode):
        timestamp_path, _ = rgb_source(episode)
    count = pq.read_metadata(timestamp_path).num_rows
    if count < 1 or (args.frames and any(i < 0 or i >= count for i in args.frames)):
        parser.error("Requested frames must lie within the depth stream")
    mount = json.loads(args.initial_mount.read_text())
    output.mkdir(parents=True, exist_ok=True)
    hands = None
    if args.repair_hands:
        from vive_correct.hand_repair import prepare_hands

        hands, hand_status = prepare_hands(episode, output / "hands")
        (output / "hand_status.json").write_text(json.dumps(hand_status, indent=2) + "\n")
        if hands is None:
            print(f"Unverified source preview: {hand_status['reason']}", flush=True)
    mode = "repaired" if hands else "unverified" if args.repair_hands else "raw"
    pending = output / f"{mode}.partial.rrd"
    command = [
        sys.executable,
        str(ROOT / "ego_data/render_rerun.py"),
        "--dataset",
        str(episode),
        "--calibration",
        str(args.initial_mount.resolve()),
        "--start",
        "0",
        "--end",
        str(count - 1),
        "--raw-hands",
        "--interpolate-tracking",
        "--skip-missing-head",
        "--hide-axes",
        "--pose-euler-sequence",
        mount["head_euler_sequence"],
        "--application-id",
        episode.name + "_" + mode + "_parquet",
        "--output",
        str(pending),
    ]
    if args.rgb_only:
        command.append("--rgb-only")
    if args.repair_hands:
        command += ["--hand-status", str(output / "hand_status.json")]
    if args.manus:
        command.append("--manus-nodes")
    if hands:
        command += ["--hand-points-dir", str(hands)]
    if not mount["head_euler_degrees"]:
        command.append("--pose-radians")
    if args.show_labels:
        command.append("--show-labels")
    if args.frames:
        command += ["--frames", *map(str, args.frames)]
    print(f"Raw export: {episode.name}; log: {output / 'render.log'}", flush=True)
    with (output / "render.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    recording = output / f"{mode}.rrd"
    pending.replace(recording)
    pending.with_suffix(".json").replace(output / f"{mode}.json")
    print(f"Recording: {recording}", flush=True)
    if args.open_rerun:
        launch([recording])
    return 0
