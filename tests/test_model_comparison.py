import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vive_correct.batch.automatic import write_json
from vive_correct.model_comparison import combine, model_label


class ModelComparisonTests(unittest.TestCase):
    def fixture(self, root, *, ids, vote):
        write_json(
            root / "comparison.json",
            dict(
                summary={"on": dict(processed=1, model="gpt-6-astra")},
                entries=[
                    dict(
                        index=1,
                        episode="/data/desk/episode",
                        missing_inputs=[],
                        on=dict(evaluated_candidate_ids=ids, review_complete=True, hands={}),
                    )
                ],
            ),
        )
        write_json(
            root / "on/qwen_reviews/01/review_summary.json",
            dict(
                decisions=[
                    dict(
                        candidate_id="a",
                        is_label=vote,
                        clear=vote,
                        physical_side="left",
                        confidence=0.9,
                    )
                ]
            ),
        )

    def test_reports_disagreement_without_claiming_accuracy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root / "a", ids=["a", "missing-response"], vote=True)
            self.fixture(root / "b", ids=["a", "missing-response"], vote=False)
            result = combine([("A", root / "a", "on"), ("B", root / "b", "on")], root / "out")
            agreement = result["decision_agreement"][0]
            self.assertEqual(agreement["paired_returned_candidates"], 1)
            self.assertEqual(agreement["label_disagreement"], 1)
            self.assertEqual(agreement["disagreement_candidates"][0]["candidate_id"], "a")
            self.assertEqual(result["arms"]["A"]["stats"]["identity_pass"], 1)
            self.assertEqual(result["arms"]["B"]["stats"]["identity_pass"], 0)
            self.assertNotIn("accuracy", result["arms"]["A"]["stats"])
            page = (root / "out/comparison.html").read_text()
            self.assertIn("模型对比 · 1 个样本", page)
            self.assertIn("1 个输入完整，0 个输入缺失", page)
            self.assertNotIn("2026-11-15", page)
            self.assertEqual(model_label(root / "a", "实验组"), "实验组 · gpt-6-astra")

    def test_rejects_same_index_for_different_episodes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root / "a", ids=["a"], vote=True)
            self.fixture(root / "b", ids=["a"], vote=True)
            path = root / "b/comparison.json"
            data = json.loads(path.read_text())
            data["entries"][0]["episode"] = "/data/other/episode"
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "Sample mismatch"):
                combine([("A", root / "a", "on"), ("B", root / "b", "on")], root / "out")

    def test_rejects_different_candidate_sets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root / "a", ids=["a"], vote=True)
            self.fixture(root / "b", ids=["a", "b"], vote=True)
            with self.assertRaisesRegex(ValueError, "Candidate mismatch"):
                combine([("A", root / "a", "on"), ("B", root / "b", "on")], root / "out")
