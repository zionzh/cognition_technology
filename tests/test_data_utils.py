import tempfile
import unittest
from pathlib import Path

from data_utils import read_rows, validate_splits


class DataTests(unittest.TestCase):
    def test_csv_bom_quoted_comma_and_label_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text('text,label\n"hello, world",1\n', encoding="utf-8-sig")
            self.assertEqual(read_rows(path), [{"text": "hello, world", "label": "1"}])

    def test_duplicates_and_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            path.write_text('{"text":"a","label":1}\n{"text":"a","label":1}\n', encoding="utf-8")
            self.assertEqual(len(read_rows(path)), 1)
            path.write_text('{"text":"a","label":1}\n{"text":"a","label":2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(path)

    def test_leakage_and_unknown_labels(self):
        train = [{"text": "a", "label": "A"}, {"text": "b", "label": "B"}]
        with self.assertRaises(ValueError):
            validate_splits(train, [{"text": "a", "label": "A"}])
        with self.assertRaises(ValueError):
            validate_splits(train, [{"text": "c", "label": "C"}])
        self.assertEqual(validate_splits(train, [{"text": "c", "label": "A"}]), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
