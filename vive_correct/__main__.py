"""Entry points for the full pipeline, single-frame estimates and reviewed batches."""

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python3 -m vive_correct")
    parser.add_argument("command", choices=("run", "view", "raw", "estimate", "batch"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "raw":
        from vive_correct.raw_view import main as raw

        return raw(args.arguments)
    if args.command == "view":
        from vive_correct.viewer import main as view

        return view(args.arguments)
    if args.command == "run":
        from vive_correct.run_dataset import main as run

        return run(args.arguments)
    if args.command == "estimate":
        from vive_correct.estimate.sticker_pipeline import main as estimate

        return estimate(args.arguments)

    from vive_correct.batch.workflow import main as batch

    return batch(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
