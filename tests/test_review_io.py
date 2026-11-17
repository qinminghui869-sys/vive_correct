"""A failed artifact write must leave the last published JSON intact."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from vive_correct.review_io import atomic_json, read_json


class ReviewIOTests(unittest.TestCase):
    def test_failed_writes_preserve_previous_document_and_remove_temporary_files(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            original = {"status": "complete", "reviewer": "审核人"}
            atomic_json(path, original)
            for durable in (False, True):
                with self.subTest(durable=durable):
                    with self.assertRaises(ValueError):
                        atomic_json(path, {"offset": float("nan")}, durable=durable)
                    with patch(
                        "vive_correct.review_io.os.replace", side_effect=OSError("disk full")
                    ):
                        with self.assertRaises(OSError):
                            atomic_json(path, {"status": "new"}, durable=durable)
                    self.assertEqual(read_json(path), original)
                    self.assertEqual(list(Path(directory).iterdir()), [path])
