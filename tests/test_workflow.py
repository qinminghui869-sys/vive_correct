"""Configuration and dispatch tests never touch the real batch outputs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from vive_correct.__main__ import main
from vive_correct.batch.bilateral_gallery import build
from vive_correct.batch.report_io import digest
from vive_correct.batch.workflow import BatchConfig, run, search


class WorkflowTests(unittest.TestCase):
    def test_paths_are_relative_to_config_and_defaults_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "batch.yaml"
            path.write_text(
                "baseline: baseline/manifest.json\noutput: output\ninitial_mount: mount.json\n"
            )
            config = BatchConfig.load(path)
            self.assertEqual(config.baseline, root / "baseline/manifest.json")
            self.assertEqual(config.review, root / "output/reviews.json")
            self.assertEqual(config.stride, 5)
            for invalid in ("stride: 0", "stride: true", "strdie: 15"):
                path.write_text(
                    "baseline: baseline/manifest.json\noutput: output\ninitial_mount: mount.json\n"
                    + invalid
                )
                with self.assertRaises(ValueError):
                    BatchConfig.load(path)

    def test_output_cannot_overwrite_baseline_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.yaml"
            path.write_text(
                "baseline: baseline/manifest.json\noutput: baseline\ninitial_mount: mount.json\n"
            )
            with self.assertRaisesRegex(ValueError, "separate"):
                BatchConfig.load(path)

    def test_missing_review_stops_before_any_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            baseline.write_text("{}")
            config = BatchConfig(baseline, root / "out", root / "mount.json", root / "missing.json")
            with patch("vive_correct.batch.workflow.reconstruct") as reconstruct:
                with self.assertRaises(FileNotFoundError):
                    run(config)
                reconstruct.assert_not_called()
                self.assertFalse(config.output.exists())

    def test_default_run_preserves_stage_order_and_report_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("baseline.json", "review.json", "mount.json"):
                (root / name).write_text("{}")
            config = BatchConfig(
                root / "baseline.json", root / "out", root / "mount.json", root / "review.json"
            )
            calls = []

            def reconstruct(*args):
                calls.append("reconstruct")
                config.manifest.parent.mkdir(parents=True)
                report = config.manifest.parent / "report.json"
                report.write_text("{}")
                config.manifest.write_text(
                    json.dumps(
                        {
                            "entries": [
                                {"index": 1, "report": str(report), "report_sha256": digest(report)}
                            ]
                        }
                    )
                )

            def calibrate(*args):
                calls.append("calibrate")
                return {"reason": "not_validated", "accepted": False, "tracks": {}}

            with (
                patch("vive_correct.batch.workflow.reconstruct", side_effect=reconstruct),
                patch("vive_correct.batch.mount_calibration.calibrate", side_effect=calibrate),
                patch(
                    "vive_correct.batch.workflow.build",
                    side_effect=lambda path: calls.append("gallery"),
                ),
            ):
                run(config)
            self.assertEqual(calls, ["reconstruct", "calibrate", "gallery"])
            (config.manifest.parent / "report.json").write_text('{"changed":true}')
            with patch("vive_correct.batch.mount_calibration.calibrate") as calibrate:
                with self.assertRaisesRegex(ValueError, "Report changed"):
                    run(config, "calibrate")
                calibrate.assert_not_called()

    def test_unified_entry_preserves_estimate_exit_code_and_arguments(self):
        with patch("vive_correct.estimate.sticker_pipeline.main", return_value=2) as estimate:
            self.assertEqual(main(["estimate", "--frame", "10"]), 2)
            estimate.assert_called_once_with(["--frame", "10"])

    def test_search_rejects_resume_with_changed_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BatchConfig(
                root / "baseline.json", root / "out", root / "mount.json", root / "review.json", 5
            )
            config.baseline.write_text(
                json.dumps(
                    {"entries": [{"index": 1, "episode": "/example/episode", "hands": ["right"]}]}
                )
            )
            folder = config.output / "search/01_episode"
            folder.mkdir(parents=True)
            pool = folder / "candidates.json"
            pool.write_text(json.dumps({"episode": "/example/episode", "stride": 15}))
            (folder / "scan_complete.json").write_text(
                json.dumps({"candidate_pool_sha256": digest(pool)})
            )
            with patch("vive_correct.batch.bilateral_search.scan") as scan:
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    search(config)
                scan.assert_not_called()

    def test_digest_matches_sha256_across_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.json"
            data = b"x" * (2 * 1024 * 1024 + 7)
            path.write_bytes(data)
            self.assertEqual(digest(path), hashlib.sha256(data).hexdigest())

    def test_gallery_uses_batch_size_and_rejects_stale_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "calibration/01").mkdir(parents=True)
            (root / "reviews.json").write_text("{}")
            overlay = root / "overlay.png"
            cv2.imwrite(str(overlay), np.zeros((120, 192, 3), np.uint8))
            report = root / "results/observations.json"
            report.write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "frame_index": 1,
                                "image": str(overlay),
                                "overlay": str(overlay),
                                "quality_status": "verified",
                                "stickers": {
                                    "left": {
                                        "T_camera_local": np.eye(4).tolist(),
                                        "image_corners_px": [
                                            [10, 10],
                                            [40, 10],
                                            [40, 30],
                                            [10, 30],
                                        ],
                                    }
                                },
                            }
                        ]
                    }
                )
            )
            (root / "results/manifest.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "index": 1,
                                "episode": "/example/episode",
                                "hands": ["left"],
                                "missing_hands": ["right"],
                                "report": str(report),
                                "report_sha256": digest(report),
                            }
                        ]
                    }
                )
            )
            calibration_path = root / "calibration/01/calibration.json"
            calibration = {
                "episode": "/example/episode",
                "accepted": False,
                "reason": "missing_anchor",
                "source_report_sha256": digest(report),
            }
            calibration_path.write_text(json.dumps(calibration))
            build(root)
            page = (root / "index.html").read_text()
            self.assertIn("1 条有效视频", page)
            self.assertNotIn("39 条", page)
            self.assertTrue((root / "bilateral_results.zip").is_file())
            with ZipFile(root / "bilateral_results.zip") as archive:
                unpacked = root / "relocated"
                archive.extractall(unpacked)
                self.assertFalse(
                    any(name.startswith(("search/", "evidence/")) for name in archive.namelist())
                )
            exported = json.loads((unpacked / "results/observations.json").read_text())
            overlay.unlink()
            for observation in exported["observations"]:
                self.assertNotIn("image", observation)
                self.assertNotIn("overlay", observation)
            self.assertIsNotNone(cv2.imread(str(unpacked / "images/01_global.jpg")))
            self.assertEqual(
                exported["mount_calibration"], {"accepted": False, "reason": "missing_anchor"}
            )
            self.assertEqual(exported["schema_version"], 6)
            self.assertEqual(digest(report), calibration["source_report_sha256"])
            for excluded in (
                "evidence",
                "calibration",
                "run_metadata.json",
                "reviews.json",
                "results/manifest.json",
            ):
                self.assertFalse((unpacked / excluded).exists())
            self.assertNotIn(
                "calibration/01/calibration.json", (unpacked / "index.html").read_text()
            )
            calibration["source_report_sha256"] = "stale"
            calibration_path.write_text(json.dumps(calibration))
            with self.assertRaisesRegex(ValueError, "stale"):
                build(root)


if __name__ == "__main__":
    unittest.main()
