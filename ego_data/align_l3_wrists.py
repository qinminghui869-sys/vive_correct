"""Launch wrist alignment with the saved A mount, frame 300 and full L3 export."""

import sys
from pathlib import Path
from align_wrists import main

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    main(
        [
            "--calibration",
            str(ROOT.parent / "calibrations/left_camera_to_head_vive.json"),
            "--output",
            str(ROOT / "outputs/wrist_alignment_A.json"),
            "--start",
            "0",
            "--end",
            "-1",
            "--initial-frame",
            "300",
            "--world-axes",
            "LUF",
            "--world-only",
            "--render-settings",
            str(ROOT / "l3_render_settings.json"),
            "--rerun-output",
            str(ROOT / "outputs/l3_A_wrists_aligned_all.rrd"),
        ]
        + sys.argv[1:]
    )
