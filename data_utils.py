"""UTF-8 JSONL / CSV loading and leakage checks (standard library only)."""
import csv
import json
from pathlib import Path


def read_rows(path, labeled=True):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(f))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            raise ValueError("数据文件必须为 .jsonl 或 .csv")
    if not rows:
        raise ValueError(f"空数据文件: {path}")
    clean, seen = [], {}
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{path}: 第 {i} 条缺少非空字符串 text")
        item = {"text": row["text"].strip()}
        if labeled:
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, (str, int)) or not str(label).strip():
                raise ValueError(f"{path}: 第 {i} 条 label 必须是字符串或整数，暂不支持多标签")
            item["label"] = str(label).strip()
            if item["text"] in seen:
                if seen[item["text"]] != item["label"]:
                    raise ValueError(f"{path}: 相同文本具有冲突标签")
                continue
            seen[item["text"]] = item["label"]
        clean.append(item)
    return clean


def validate_splits(train, valid, test=None):
    labels = sorted({r["label"] for r in train})
    if len(labels) < 2:
        raise ValueError("训练集至少需要两个类别")
    splits = [("train", train), ("valid", valid)]
    if test is not None:
        splits.append(("test", test))
    seen = set()
    for name, rows in splits:
        unknown = {r["label"] for r in rows} - set(labels)
        if unknown:
            raise ValueError(f"{name} 存在训练集中未出现的标签: {unknown}")
        texts = {r["text"] for r in rows}
        if seen & texts:
            raise ValueError(f"{name} 与其他集合存在重复文本；请先消除数据泄漏")
        seen.update(texts)
    return labels
