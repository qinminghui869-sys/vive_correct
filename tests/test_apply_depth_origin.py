"""Exercise saved-review migration and explicit depth-origin source writes."""

import errno
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.apply_depth_origin import run
from test_human_review import make_fixture
from vive_correct.batch.report_io import digest
from vive_correct.data.source_data import DATA, FILES, original_files
from vive_correct.review.human_review import ReviewStore
from vive_correct.review.review_actions import rebuild
from vive_correct.review.review_bundle import approve
from vive_correct.review.review_io import atomic_json, read_json


class DepthApplyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="深度 + ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, VIVE_REBUILD_INDEX=str(self.root / "index.json"))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.episode, self.process, mount = make_fixture(self.root)
        store = ReviewStore(self.process)
        state = store.load(0)
        for side, choice in state["selected"].items():
            choice.update(status="accepted", physical_side=side)
        state["reviewer"] = "fixture"
        approve(store, 0, state, mount)
        result = rebuild(self.process, mount, in_place=True)
        self.assertEqual(result["status"], "complete")
        self.before = self.hashes()
        self.baselines = {n: digest(p) for n, p in original_files(self.episode).items()}
        # Relocate historical absolute references while retaining immutable artifacts.
        pointer = next((self.process / "features").rglob("approval.json"))
        approval = read_json(pointer)
        for key in ("directory", "human_review_snapshot", "report", "alignment", "calibration"):
            approval[key] = "/old-machine/" + Path(approval[key]).name
        approval["artifacts"] = {
            "/old-machine/" + Path(p).name: sha for p, sha in approval["artifacts"].items()
        }
        atomic_json(pointer, approval)
        (self.root / "index.json").unlink()

    def hashes(self):
        return {n: digest(self.episode / DATA / n) for n in FILES}

    def execute(self):
        return run(self.episode, self.process)

    def test_write_repeat_and_preserved_fields(self):
        result = self.execute()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(list(self.process.glob(".depth_origin-*")))
        self.assertEqual(
            self.hashes(), {n: row["sha256"] for n, row in result["entries"][0]["files"].items()}
        )
        originals = original_files(self.episode)
        self.assertEqual({n: digest(p) for n, p in originals.items()}, self.baselines)
        for name in FILES:
            before = pq.read_table(originals[name])
            after = pq.read_table(self.episode / DATA / name)
            self.assertTrue(before.schema.equals(after.schema, check_metadata=True))
            preserved = [
                c
                for c in before.column_names
                if c in ("timestamp", "episode_index")
                or (
                    name in ("action.parquet", "observation.state.parquet")
                    and not c.startswith(("left_wrist_", "right_wrist_"))
                )
            ]
            for column in preserved:
                self.assertTrue(before[column].equals(after[column]), (name, column))
        first = self.hashes()
        self.execute()
        self.assertEqual(first, self.hashes())

    def test_failure_rolls_back_and_records_error(self):
        with patch(
            "vive_correct.review.dataset_meta.refresh_dataset_meta",
            side_effect=OSError("injected failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected failure"):
                self.execute()
        self.assertEqual(self.hashes(), self.before)
        manifest = read_json(next((self.process / "depth_origin_runs").glob("*.json")))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["entries"][0]["status"], "failed")
        self.assertEqual(self.execute()["status"], "complete")

    def test_external_change_rejected(self):
        source = self.episode / DATA / FILES[0]
        with source.open("ab") as stream:
            stream.write(b"external change")
        with self.assertRaisesRegex(ValueError, "外部修改"):
            self.execute()
        self.assertFalse((self.root / "index.json").exists())

    def test_missing_baseline_rejected(self):
        baseline = next((self.process / "rebuild_state").glob("*/baseline"))
        (baseline / FILES[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute()
        self.assertEqual(self.hashes(), self.before)
        self.assertFalse((self.root / "index.json").exists())

    def test_invalid_depth_aborts_before_write(self):
        depth_times = (
            self.episode / DATA / "observation.images.depth_headband_left_rectified.parquet"
        )
        table = pq.read_table(depth_times)
        values = pa.array([100.0] * len(table), type=table["timestamp"].type)
        table = table.set_column(
            table.schema.get_field_index("timestamp"), table.schema.field("timestamp"), values
        )
        pq.write_table(table, depth_times)
        with self.assertRaisesRegex(ValueError, "5 ms"):
            self.execute()
        self.assertEqual(self.hashes(), self.before)
        self.assertFalse((self.root / "index.json").exists())

    def test_cross_filesystem_publication(self):
        replace = os.replace

        def cross_mount(source, destination):
            if Path(source).parent.name == "new":
                raise OSError(errno.EXDEV, "cross-device link")
            return replace(source, destination)

        with patch("vive_correct.data.source_data.os.replace", side_effect=cross_mount):
            result = self.execute()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(list((self.episode / DATA).glob("*.restore")))
        self.assertEqual(
            self.hashes(), {n: r["sha256"] for n, r in result["entries"][0]["files"].items()}
        )

    def test_invalid_episode_rejected_without_source_changes(self):
        with self.assertRaisesRegex(ValueError, "--episode"):
            run(
                self.episode,
                self.process,
                episodes=["episode_999999"],
            )
        self.assertEqual(self.hashes(), self.before)

    def test_process_worker_matches_serial_and_cleans_staging(self):
        serial = self.execute()
        parallel = run(self.episode, self.process, workers=2)
        self.assertEqual(parallel["status"], "complete")
        self.assertEqual(
            {n: row["sha256"] for n, row in serial["entries"][0]["files"].items()},
            {n: row["sha256"] for n, row in parallel["entries"][0]["files"].items()},
        )
        self.assertFalse(list(self.process.glob(".depth_origin-*")))

    def test_invalid_worker_count_does_not_write(self):
        with self.assertRaisesRegex(ValueError, "--jobs"):
            run(self.episode, self.process, workers=0)
        self.assertEqual(self.hashes(), self.before)
