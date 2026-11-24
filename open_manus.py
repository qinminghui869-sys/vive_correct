#!/usr/bin/env python3
"""Open an episode's original MANUS nodes, RGB and available depth in Rerun."""

import argparse
from pathlib import Path
from vive_correct.export.raw_view import main as raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args, extra = parser.parse_known_args()
    episode = args.episode.resolve()
    output = (
        Path(__file__).resolve().parent / "outputs/manus" / f"{episode.parent.name}_{episode.name}"
    )
    return raw(
        [
            "--dataset",
            str(episode),
            "--out",
            str(output),
            "--manus",
            "--repair-hands",
            "--show-labels",
            "--open-rerun",
            *extra,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
