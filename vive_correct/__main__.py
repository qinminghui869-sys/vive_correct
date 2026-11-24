"""Entry points for the full pipeline, single-frame estimates and reviewed batches."""

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "run": "run_dataset",
    "view": "export.viewer",
    "raw": "export.raw_view",
    "estimate": "estimate.sticker_pipeline",
    "batch": "batch.workflow",
    "review": "review.review_server",
    "repair": "review.review_repair",
    "rebuild": "review.review_actions",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python3 -m vive_correct")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # Keep imports lazy: CLI help must not initialize renderers or model clients.
    command = import_module(f"vive_correct.{COMMANDS[args.command]}")
    return command.main(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
