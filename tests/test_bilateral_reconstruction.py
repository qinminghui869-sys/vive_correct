import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from vive_correct.batch.bilateral_reconstruction import reconstruct


class BilateralReconstructionTests(unittest.TestCase):
    def fixture(self, root, review):
        source = root / "source.json"
        source.write_text(
            json.dumps({"observations": [{"frame_index": 1, "stickers": {"right": {}}}]})
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "index": 1,
                            "episode": "/example/episode",
                            "report": source.name,
                            "report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "status": "verified",
                        }
                    ]
                }
            )
        )
        reviews = root / "reviews.json"
        reviews.write_text(json.dumps({"entries": [review]}))
        return manifest, reviews, source

    def test_unreviewed_opposite_hand_does_not_increase_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reviews, source = self.fixture(
                root,
                {
                    "index": 1,
                    "candidate_id": "proposal",
                    "side": "left",
                    "identity_verified": False,
                    "clear_label": True,
                },
            )
            before = source.read_bytes()
            result = reconstruct(manifest, reviews, root / "out")
            self.assertEqual(result["bilateral_observation_count"], 0)
            self.assertEqual(result["entries"][0]["missing_hands"], ["left"])
            self.assertEqual(source.read_bytes(), before)

    def test_modified_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reviews, source = self.fixture(root, {"index": 1})
            source.write_text(source.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reconstruct(manifest, reviews, root / "out")


if __name__ == "__main__":
    unittest.main()
