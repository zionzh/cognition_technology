import json
from pathlib import Path
import random
import tempfile
import unittest

from data_preparation import prepare_data


def body(seed):
    rng = random.Random(seed)
    return " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=9)) for _ in range(100))


class PreparationTests(unittest.TestCase):
    def run_preparation(self, extra_train, test=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        train = [{"text": body(i), "label": label}
                 for label, start in (("A", 10), ("B", 20)) for i in range(start, start+4)]
        train += extra_train
        test = test or [{"text": body(30), "label": "A"}, {"text": body(31), "label": "B"}]
        for name, rows in (("train", train), ("test", test)):
            (root / f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        original = (root / "train.jsonl").read_bytes()
        report = prepare_data(root / "train.jsonl", root / "test.jsonl", root / "out")
        self.assertEqual((root / "train.jsonl").read_bytes(), original)
        return root, report

    def read(self, root, name):
        return [json.loads(line) for line in (root / "out" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_conflicting_article_versions_are_quarantined(self):
        title = "One article with conflicting labels"
        root, report = self.run_preparation([
            {"title": title, "content": body(1), "label": "A"},
            {"title": title, "content": body(1)+" Author biography.", "label": "B"}])
        self.assertEqual(report["quarantined_train_count"], 2)
        self.assertEqual(report["training_pool_count"], 8)
        self.assertEqual(len(self.read(root, "label_conflicts")), 2)

    def test_changed_title_web_chrome_cannot_leak_from_test(self):
        root, report = self.run_preparation([
            {"title": "Alternate headline", "content": "Share this article. "+body(30), "label": "B"}])
        self.assertEqual(report["excluded_test_overlap_count"], 1)
        self.assertEqual(report["test_conflict_count"], 1)
        self.assertEqual(self.read(root, "test")[0]["label"], "A")

    def test_weaker_test_overlap_is_excluded_without_relabeling(self):
        root, report = self.run_preparation([
            {"text": body(30)[:700]+body(90)[:300], "label": "B"}])
        self.assertEqual(report["excluded_possible_test_overlap_count"], 1)
        self.assertEqual(report["conflicting_group_count"], 0)
        self.assertEqual(len(self.read(root, "similarity_review")), 1)

    def test_full_article_and_excerpt_have_one_representative(self):
        root, report = self.run_preparation([
            {"text": body(1), "label": "A"},
            {"text": body(1)[:400], "label": "A"},
            {"text": "A different wrapper. "+body(1), "label": "A"}])
        self.assertEqual(report["duplicate_variants_removed"], 2)
        self.assertEqual(report["training_pool_count"], 9)
        sets = [{r["group_id"] for r in self.read(root, name)} for name in ("train", "valid", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        prepare_data(root / "train.jsonl", root / "test.jsonl", root / "repeat")
        for name in ("train", "valid", "test"):
            self.assertEqual((root / "out" / f"{name}.jsonl").read_bytes(),
                             (root / "repeat" / f"{name}.jsonl").read_bytes())

    def test_too_few_clean_samples_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("train", "test"):
                (root / f"{name}.jsonl").write_text('{"text":"one","label":"A"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_data(root / "train.jsonl", root / "test.jsonl", root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
