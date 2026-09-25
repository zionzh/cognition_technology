"""Auditable, dependency-free article grouping for fresh training runs."""
from collections import Counter, defaultdict
from itertools import combinations
import csv
import hashlib
import json
from pathlib import Path
import random
import unicodedata

from data_utils import TEXT_NORMALIZATION, build_text, validate_splits


def _key(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _load(path, source):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        if path.suffix.lower() == ".jsonl":
            raw = [(i, json.loads(line)) for i, line in enumerate(f, 1) if line.strip()]
        elif path.suffix.lower() == ".csv":
            raw = list(enumerate(csv.DictReader(f), 2))
        else:
            raise ValueError("Data must be JSONL or CSV")
    if not raw:
        raise ValueError(f"Empty dataset: {path}")
    records = []
    for line, row in raw:
        text = build_text(row)
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, (str, int)) or not str(label).strip():
            raise ValueError(f"{path}:{line}: invalid label")
        title = row.get("title") or ""
        if not title and text.startswith("title: "):
            title = text[7:].split("\ncontent: ", 1)[0]
        content = text.split("content: ", 1)[1] if "content: " in text else text
        records.append({"source": source, "source_line": line, "source_id": row.get("id"),
                        "title": title, "text": text, "label": str(label).strip(),
                        "_body": _key(content), "_title": _key(title)})
    return records


def group_articles(records):
    """Conservative lexical grouping, independent of labels; not semantic deduplication.

    Match exact text, substantive identical titles, or long bodies with >=90%
    shorter-body shingle containment / >=80% Jaccard. Keep weaker matches for review.
    Connected components prevent transitive duplicates crossing split boundaries.
    """
    parents = list(range(len(records)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        parents[root(b)] = root(a)

    links = {}
    for field in ("text", "_title"):
        seen = {}
        for i, row in enumerate(records):
            key = row[field]
            if not key or (field == "_title" and len(key) < 12):
                continue
            if key in seen:
                j = seen[key]
                join(j, i)
                links[(j, i)] = {"a": j, "b": i, "reason": "exact_text" if field == "text" else "same_title"}
            else:
                seen[key] = i
    shingles = [{r["_body"][j:j+5] for j in range(len(r["_body"])-4)}
                if len(r["_body"]) >= 200 else set() for r in records]
    postings = defaultdict(list)
    for i, values in enumerate(shingles):
        for value in values:
            postings[value].append(i)
    intersections = Counter()
    for members in postings.values():
        intersections.update(combinations(members, 2))
    review = []
    for (a, b), overlap in sorted(intersections.items()):
        short = min(len(shingles[a]), len(shingles[b]))
        if short < 100:
            continue
        containment = overlap / short
        jaccard = overlap / (len(shingles[a]) + len(shingles[b]) - overlap)
        evidence = {"a": a, "b": b, "reason": "similar_body",
                    "containment": round(containment, 4), "jaccard": round(jaccard, 4)}
        if containment >= .90 or jaccard >= .80:
            join(a, b)
            links.setdefault((a, b), evidence)
        elif containment >= .65:
            review.append(evidence)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[root(i)].append(i)
    return list(groups.values()), list(links.values()), review


def prepare_data(train_path, test_path, directory, val_ratio=.2, seed=42):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")
    records = _load(train_path, "train") + _load(test_path, "test")
    groups, links, review = group_articles(records)
    # Be stricter about test independence than within-pool duplicate removal.
    # Quarantine training groups with even a weaker cross-test match, without
    # claiming that these pairs necessarily have incorrect labels.
    possible_test_overlap = set()
    for pair in review:
        a, b = pair["a"], pair["b"]
        if records[a]["source"] != records[b]["source"]:
            possible_test_overlap.add(a if records[a]["source"] == "train" else b)
    pool, audit, quarantine, excluded = [], [], [], []
    for members in groups:
        rows = [records[i] for i in members]
        group_id = hashlib.sha256("\n".join(sorted(r["text"] for r in rows)).encode()).hexdigest()[:16]
        conflict = len({r["label"] for r in rows}) > 1
        has_test = any(r["source"] == "test" for r in rows)
        possible_overlap = bool(possible_test_overlap.intersection(members))
        train_members = [i for i in members if records[i]["source"] == "train"]
        representative = max(train_members, key=lambda i: (len(records[i]["text"]), -i)) if train_members else None
        for i in members:
            row = records[i]
            row["group_id"] = group_id
            if row["source"] == "test":
                action = "test_reference_conflict" if conflict else "test_reference"
            elif has_test:
                action = "excluded_test_overlap"
                excluded.append(i)
            elif possible_overlap:
                action = "excluded_possible_test_overlap"
                excluded.append(i)
            elif conflict:
                action = "quarantined_label_conflict"
            elif i == representative:
                action = "training_pool"
                pool.append(i)
            else:
                action = "duplicate_variant"
            row["action"] = action
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            public["record_index"] = i
            audit.append(public)
            if conflict:
                quarantine.append(public)

    by_label = defaultdict(list)
    for i in pool:
        by_label[records[i]["label"]].append(i)
    if len(by_label) < 2 or any(len(v) < 2 for v in by_label.values()):
        raise ValueError("Cleaning leaves fewer than two independent articles per class; review conflicts first")
    rng = random.Random(seed)
    train_ids, valid_ids = [], []
    for label in sorted(by_label):
        ids = by_label[label][:]
        rng.shuffle(ids)
        n = max(1, min(len(ids)-1, round(len(ids)*val_ratio)))
        valid_ids.extend(ids[:n])
        train_ids.extend(ids[n:])
    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)
    test_ids = [i for i, r in enumerate(records) if r["source"] == "test"]

    def clean(ids):
        return [{k: records[i][k] for k in ("text", "label", "group_id", "source_line")} for i in ids]

    splits = {"train": clean(train_ids), "valid": clean(valid_ids), "test": clean(test_ids)}
    validate_splits(splits["train"], splits["valid"], splits["test"])
    group_sets = [{r["group_id"] for r in split} for split in splits.values()]
    if any(a & b for a, b in combinations(group_sets, 2)):
        raise AssertionError("Article group leakage")
    assignments = {i: name for name, ids in (("train", train_ids), ("valid", valid_ids), ("test", test_ids)) for i in ids}
    for row in audit:
        row["split"] = assignments.get(row["record_index"])
    counts = lambda rows: dict(Counter(r["label"] for r in rows))
    report = {
        "text_normalization": TEXT_NORMALIZATION, "grouping_version": "article_shingles_v1",
        "grouping_rules": {"same_title_min_chars": 12, "body_min_chars": 200,
                           "shingle_chars": 5, "min_unique_shingles": 100,
                           "containment": .9, "jaccard": .8, "review_containment": .65},
        "seed": seed, "val_ratio": val_ratio,
        "source_train": str(Path(train_path).resolve()), "source_test": str(Path(test_path).resolve()),
        "source_train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "source_test_sha256": hashlib.sha256(Path(test_path).read_bytes()).hexdigest(),
        "raw_train_count": sum(r["source"] == "train" for r in records),
        "training_pool_count": len(pool), "training_pool_label_counts": counts(clean(pool)),
        "excluded_test_overlap_count": len(excluded),
        "excluded_possible_test_overlap_count": sum(r["action"] == "excluded_possible_test_overlap" for r in audit),
        "conflicting_group_count": len({r["group_id"] for r in quarantine}),
        "quarantined_train_count": sum(r["source"] == "train" for r in quarantine),
        "test_conflict_count": sum(r["source"] == "test" for r in quarantine),
        "duplicate_variants_removed": sum(r["action"] == "duplicate_variant" for r in audit),
        "test_count": len(test_ids), "test_label_counts": counts(splits["test"]),
        "splits": {name: {"count": len(rows), "label_counts": counts(rows)} for name, rows in splits.items()},
        "remaining_detected_group_overlap": 0,
        "remaining_blank_line_samples": sum("\n\n" in r["text"] for r in records),
        "review_candidate_count": len(review),
        "note": "Test labels are unchanged. Conflicts require human review; lexical grouping cannot detect all paraphrases.",
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in {**splits, "audit": audit, "label_conflicts": quarantine,
                       "duplicate_matches": links, "similarity_review": review}.items():
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (directory / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
