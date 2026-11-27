"""Regression checks for local and canonical GitHub documentation fragments."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_docs import check


class DocumentationLinksTests(unittest.TestCase):
    def inspect(self, text, target='<a name="depth-origin-writeback"></a>\n## 写回\n'):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "README.md").write_text(text)
            (root / "docs/review_rebuild.md").write_text(target)
            return check(root, {"README.md", "docs/review_rebuild.md"})[2]

    def test_reported_anchor_and_canonical_url_are_valid(self):
        self.assertEqual(
            self.inspect(
                "[本地](docs/review_rebuild.md#depth-origin-writeback)\n"
                "[GitHub](https://github.com/qinminghui869-sys/vive_correct/blob/main/docs/review_rebuild.md#depth-origin-writeback)"
            ),
            [],
        )

    def test_canonical_missing_anchor_and_file_are_detected(self):
        for suffix in ("review_rebuild.md#missing", "missing.md#depth-origin-writeback"):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    len(
                        self.inspect(
                            f"[GitHub](https://github.com/qinminghui869-sys/vive_correct/blob/main/docs/{suffix})"
                        )
                    ),
                    1,
                )

    def test_fenced_anchor_does_not_satisfy_link(self):
        self.assertEqual(
            len(
                self.inspect(
                    "[写回](docs/review_rebuild.md#depth-origin-writeback)",
                    '```html\n<a name="depth-origin-writeback"></a>\n```\n',
                )
            ),
            1,
        )

    def test_duplicate_anchor_is_rejected(self):
        self.assertEqual(len(self.inspect("", '<a name="depth-origin-writeback"></a>\n' * 2)), 1)

    def test_external_and_historical_urls_are_not_checked_against_main(self):
        self.assertEqual(
            self.inspect(
                "[其他仓库](https://github.com/example/other/blob/main/missing.md#missing)\n"
                "[历史](https://github.com/qinminghui869-sys/vive_correct/blob/3958b7a/missing.md#missing)"
            ),
            [],
        )
