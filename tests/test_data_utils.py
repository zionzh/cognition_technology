import json
import tempfile
import unittest
from pathlib import Path

from data_utils import build_text, normalize_text, read_rows, validate_splits
from retrain_normalized import prepare_data


class DataTests(unittest.TestCase):
    def test_normalize_newlines_and_blank_lines(self):
        expected = "first\nsecond\nthird"
        for text in ("first\n\nsecond\n\n\nthird",
                     "first\r\n\r\nsecond\r\rthird",
                     "\nfirst\n \t\nsecond\n\u3000\nthird\n"):
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text), expected)
                self.assertEqual(normalize_text(normalize_text(text)), expected)
        # Keep literal backslash-n sequences and word spaces; these are not blank lines.
        self.assertEqual(normalize_text(r"first\nsecond"), r"first\nsecond")
        self.assertEqual(normalize_text("two  words"), "two  words")

    def test_normalization_matches_article_and_saved_text(self):
        article = {"title": "标题\r\n\r\n副标题", "content": "第一段\n\t\n第二段"}
        expected = "title: 标题\n副标题\ncontent: 第一段\n第二段"
        self.assertEqual(build_text(article), expected)
        self.assertEqual(build_text({"text": expected.replace("\n", "\n\n")}), expected)

    def test_prepare_excludes_test_overlap_after_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{"text": "shared\n\narticle", "label": "A"},
                    {"text": "A1", "label": "A"}, {"text": "A2", "label": "A"},
                    {"text": "B1", "label": "B"}, {"text": "B2", "label": "B"}]
            test = [{"text": "shared\narticle", "label": "A"}, {"text": "B3", "label": "B"}]
            for name, records in (("train", rows), ("test", test)):
                (root / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in records), encoding="utf-8")
            before = (root / "train.jsonl").read_bytes()
            report = prepare_data(root / "train.jsonl", root / "test.jsonl", root / "prepared")
            self.assertEqual(report["excluded_test_overlap_count"], 1)
            self.assertEqual(report["training_pool_count"], 4)
            self.assertEqual(report["test_count"], 2)
            self.assertEqual((root / "train.jsonl").read_bytes(), before)
            validate_splits(read_rows(root / "prepared/train.jsonl"), [], read_rows(root / "prepared/test.jsonl"))
            test[0]["label"] = "B"
            (root / "test.jsonl").write_text("\n".join(json.dumps(r) for r in test), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "标签冲突"):
                prepare_data(root / "train.jsonl", root / "test.jsonl", root / "conflict")
            self.assertFalse((root / "conflict").exists())

    def test_title_content_format_and_precedence(self):
        row = {"id": "123", "title": " 标题 ", "content": " 正文\n第二段 ",
               "text": "旧文本", "label": "类别"}
        self.assertEqual(build_text(row), "title: 标题\ncontent: 正文\n第二段")

    def test_one_empty_field(self):
        for empty in (None, "", "  \n "):
            with self.subTest(empty=empty):
                self.assertEqual(build_text({"title": empty, "content": "正文"}), "content: 正文")
                self.assertEqual(build_text({"title": "标题", "content": empty}), "title: 标题")
        self.assertEqual(build_text({"content": "正文"}), "content: 正文")
        self.assertEqual(build_text({"title": "标题"}), "title: 标题")

    def test_empty_article_and_invalid_types(self):
        for row in ({"title": "", "content": None}, {"title": None},
                    {"title": " ", "content": "", "text": "不能回退到旧文本"},
                    {"title": 123, "content": "正文"}, {"content": ["正文"]}, []):
            with self.subTest(row=row), self.assertRaises(ValueError):
                build_text(row)

    def test_article_read_prediction_and_split_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            article = {"id": "123", "title": " 标题 ", "content": "正文", "label": 1}
            path.write_text(json.dumps(article, ensure_ascii=False) + "\n", encoding="utf-8")
            expected = [{"text": "title: 标题\ncontent: 正文", "label": "1"}]
            self.assertEqual(read_rows(path), expected)
            self.assertEqual(read_rows(path, labeled=False), [{"text": expected[0]["text"]}])
            # Training saves canonical text; reading that split must not add prefixes again.
            path.write_text(json.dumps(expected[0], ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_rows(path), expected)

    def test_article_csv_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text('title,content,label\n标题,"第一段,内容\n第二段",A\n', encoding="utf-8-sig", newline="")
            self.assertEqual(read_rows(path), [{"text": "title: 标题\ncontent: 第一段,内容\n第二段", "label": "A"}])
            path = Path(d) / "data.jsonl"
            rows = [{"title": "标题", "content": "正文", "label": "A"},
                    {"title": " 标题 ", "content": "正文", "label": "A"}]
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual(len(read_rows(path)), 1)
            self.assertEqual(len(read_rows(path, labeled=False)), 2)
            rows[1]["label"] = "B"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(path)

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
