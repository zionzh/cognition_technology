from __future__ import annotations

import csv
import io
import json
import random
import re
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


# ============================================================
# 配置
# ============================================================

REPO_ID = "csebuetnlp/xlsum"
VERSION = "2.0"

SEED = 2026

# 三个子数据集
#
# 200条：
# 英文 : 繁中 : 日韩
# 120 : 70 : 10
#
# 100条：
# 60 : 35 : 5

SUBSETS = {
    "subset_1": {
        "english": 120,
        "chinese_traditional": 70,
        "japanese": 5,
        "korean": 5,
    },

    "subset_2": {
        "english": 120,
        "chinese_traditional": 70,
        "japanese": 5,
        "korean": 5,
    },

    "subset_3": {
        "english": 60,
        "chinese_traditional": 35,
        "japanese": 3,
        "korean": 2,
    },
}


# 总需求
TOTAL_TARGETS = {
    "english": 300,
    "chinese_traditional": 175,
    "japanese": 13,
    "korean": 12,
}


# 是否启用地域关键词过滤
STRICT_REGION_FILTER = True


# 输出目录
OUTPUT_DIR = Path("xlsum_subsets")


# ============================================================
# 地区关键词
# ============================================================

EN_REGION_TERMS = [
    # 美国
    "United States",
    "U.S.",
    "USA",
    "America",
    "American",
    "Washington",
    "New York",
    "California",
    "Texas",

    # 英国
    "United Kingdom",
    "UK",
    "Britain",
    "British",
    "England",
    "Scotland",
    "Wales",
    "Northern Ireland",
    "London",

    # 欧洲
    "Europe",
    "European",
    "European Union",
    "EU",
    "NATO",

    "France",
    "French",
    "Germany",
    "German",
    "Italy",
    "Italian",
    "Spain",
    "Spanish",
    "Portugal",
    "Netherlands",
    "Dutch",
    "Belgium",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Poland",
    "Ireland",
    "Austria",
    "Switzerland",
    "Greece",
    "Romania",
    "Hungary",

    "Paris",
    "Berlin",
    "Brussels",
]


ZH_TW_TERMS = [
    "台灣",
    "臺灣",

    "台北",
    "臺北",
    "新北",
    "桃園",
    "台中",
    "臺中",
    "台南",
    "臺南",
    "高雄",

    "基隆",
    "新竹",
    "嘉義",
    "花蓮",
    "台東",
    "臺東",
    "澎湖",
    "金門",
    "馬祖",

    "立法院",
    "行政院",
    "總統府",
    "民進黨",
    "國民黨",

    "台海",
    "臺海",
    "兩岸",
]


JA_REGION_TERMS = [
    "日本",
    "東京",
    "大阪",
    "京都",
    "北海道",
    "沖縄",
    "福岡",

    "国会",
    "首相",
    "自民党",
]


KO_REGION_TERMS = [
    "한국",
    "대한민국",
    "서울",
    "부산",
    "제주",

    "국회",
    "대통령",
    "정부",
]


# ============================================================
# 清洗
# ============================================================

def normalize_record(obj: dict) -> dict | None:
    """
    XL-Sum 原始字段：
        url
        title
        text

    输出字段：
        url
        title
        content
    """

    url = (obj.get("url") or "").strip()
    title = (obj.get("title") or "").strip()
    content = (obj.get("text") or "").strip()

    if not url or not title or not content:
        return None

    return {
        "url": url,
        "title": title,
        "content": content,
    }


# ============================================================
# 地区筛选
# ============================================================

def region_match(lang: str, record: dict) -> bool:

    if not STRICT_REGION_FILTER:
        return True

    text = record["title"] + "\n" + record["content"]

    # --------------------------------------------------------
    # 英文 / 欧美
    # --------------------------------------------------------

    if lang == "english":

        text_lower = text.lower()

        for term in EN_REGION_TERMS:

            term_lower = term.lower()

            # EU、UK、USA 等短词
            if len(term_lower) <= 3:

                pattern = (
                    rf"(?<![A-Za-z])"
                    rf"{re.escape(term_lower)}"
                    rf"(?![A-Za-z])"
                )

                if re.search(pattern, text_lower):
                    return True

            elif term_lower in text_lower:
                return True

        return False

    # --------------------------------------------------------
    # 繁体中文 / 台湾
    # --------------------------------------------------------

    if lang == "chinese_traditional":

        return any(
            keyword in text
            for keyword in ZH_TW_TERMS
        )

    # --------------------------------------------------------
    # 日本
    # --------------------------------------------------------

    if lang == "japanese":

        return any(
            keyword in text
            for keyword in JA_REGION_TERMS
        )

    # --------------------------------------------------------
    # 韩国
    # --------------------------------------------------------

    if lang == "korean":

        return any(
            keyword in text
            for keyword in KO_REGION_TERMS
        )

    return True


# ============================================================
# 下载 XL-Sum
# ============================================================

def download_archive(lang: str) -> Path:

    filename = f"data/{lang}_XLSum_v{VERSION}.tar.bz2"

    print()
    print(f"[DOWNLOAD] {lang}")
    print(f"           {filename}")

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
    )

    return Path(local_path)


# ============================================================
# Reservoir Sampling
# ============================================================

def sample_language(
    lang: str,
    sample_size: int,
    rng: random.Random,
) -> list[dict]:

    archive_path = download_archive(lang)

    target_jsonl = f"{lang}_train.jsonl"

    samples = []

    eligible_count = 0

    print()
    print(f"[SAMPLING] {lang}")
    print(f"           need = {sample_size}")

    with tarfile.open(
        archive_path,
        "r:bz2",
    ) as tar:

        member = None

        for item in tar.getmembers():

            if Path(item.name).name == target_jsonl:

                member = item
                break

        if member is None:

            raise FileNotFoundError(
                f"找不到：{target_jsonl}"
            )

        raw_file = tar.extractfile(member)

        if raw_file is None:

            raise RuntimeError(
                f"无法读取：{target_jsonl}"
            )

        with io.TextIOWrapper(
            raw_file,
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                obj = json.loads(line)

                record = normalize_record(obj)

                if record is None:
                    continue

                # 地域过滤
                if not region_match(
                    lang,
                    record,
                ):
                    continue

                eligible_count += 1

                # Reservoir Sampling
                if len(samples) < sample_size:

                    samples.append(record)

                else:

                    j = rng.randrange(
                        eligible_count
                    )

                    if j < sample_size:
                        samples[j] = record

    if len(samples) < sample_size:

        raise RuntimeError(
            f"\n{lang} 可用数据不足。\n"
            f"需要：{sample_size}\n"
            f"实际：{len(samples)}\n\n"
            f"可以增加地域关键词，或者设置：\n"
            f"STRICT_REGION_FILTER = False\n"
        )

    # 再打乱一次
    rng.shuffle(samples)

    print(
        f"[OK] {lang}: "
        f"符合地区过滤={eligible_count}, "
        f"随机选取={len(samples)}"
    )

    return samples


# ============================================================
# JSONL
# ============================================================

def write_jsonl(
    rows: list[dict],
    path: Path,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# CSV
# ============================================================

def write_csv(
    rows: list[dict],
    path: Path,
):

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url",
                "title",
                "content",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 检查配置
# ============================================================

def validate_config():

    # 检查子数据集大小

    expected_sizes = {
        "subset_1": 200,
        "subset_2": 200,
        "subset_3": 100,
    }

    for subset_name, config in SUBSETS.items():

        actual = sum(config.values())

        expected = expected_sizes[subset_name]

        assert actual == expected, (
            f"{subset_name} 应该是 {expected} 条，"
            f"实际配置为 {actual} 条"
        )

    # 检查总语言数量

    calculated = {
        lang: sum(
            subset.get(lang, 0)
            for subset in SUBSETS.values()
        )
        for lang in TOTAL_TARGETS
    }

    assert calculated == TOTAL_TARGETS, (
        f"语言总量配置错误：\n"
        f"期望：{TOTAL_TARGETS}\n"
        f"实际：{calculated}"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    validate_config()

    rng = random.Random(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("XL-Sum 500条数据构建")
    print("拆分：200 + 200 + 100")
    print("=" * 70)

    print()
    print("总语言需求：")

    for lang, count in TOTAL_TARGETS.items():

        print(
            f"  {lang:24s}: "
            f"{count}"
        )

    # ========================================================
    # 1. 每种语言只抽一次
    #
    # 这样可以保证三个子数据集之间没有重复新闻
    # ========================================================

    language_pools = {}

    for lang, count in TOTAL_TARGETS.items():

        language_pools[lang] = sample_language(
            lang=lang,
            sample_size=count,
            rng=rng,
        )

    # ========================================================
    # 2. 记录每个语言池已经使用到的位置
    # ========================================================

    offsets = {
        lang: 0
        for lang in TOTAL_TARGETS
    }

    # ========================================================
    # 3. 创建三个子数据集
    # ========================================================

    for subset_name, config in SUBSETS.items():

        print()
        print("=" * 70)
        print(f"构建 {subset_name}")
        print("=" * 70)

        subset_rows = []

        for lang, count in config.items():

            start = offsets[lang]

            end = start + count

            selected = language_pools[lang][
                start:end
            ]

            if len(selected) != count:

                raise RuntimeError(
                    f"{subset_name} 中 "
                    f"{lang} 数量不足"
                )

            subset_rows.extend(selected)

            offsets[lang] = end

            print(
                f"  {lang:24s}: "
                f"{len(selected)}"
            )

        # ------------------------------------------
        # 把不同语言混合打乱
        # ------------------------------------------

        rng.shuffle(subset_rows)

        # ------------------------------------------
        # 子目录
        # ------------------------------------------

        subset_dir = (
            OUTPUT_DIR
            / subset_name
        )

        subset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------
        # 输出 JSONL
        # ------------------------------------------

        jsonl_path = (
            subset_dir
            / f"{subset_name}.jsonl"
        )

        write_jsonl(
            subset_rows,
            jsonl_path,
        )

        # ------------------------------------------
        # 输出 CSV
        # ------------------------------------------

        csv_path = (
            subset_dir
            / f"{subset_name}.csv"
        )

        write_csv(
            subset_rows,
            csv_path,
        )

        print()
        print(
            f"[SAVE] JSONL: "
            f"{jsonl_path}"
        )

        print(
            f"[SAVE] CSV:   "
            f"{csv_path}"
        )

        print(
            f"[TOTAL]      "
            f"{len(subset_rows)}"
        )

    # ========================================================
    # 4. 检查所有数据是否全部使用
    # ========================================================

    print()
    print("=" * 70)
    print("最终检查")
    print("=" * 70)

    total_rows = 0

    for subset_name, config in SUBSETS.items():

        subset_total = sum(
            config.values()
        )

        total_rows += subset_total

        print(
            f"{subset_name}: "
            f"{subset_total}"
        )

    print("-" * 70)

    print(
        f"全部数据：{total_rows}"
    )

    print()
    print("语言合计：")

    for lang, count in offsets.items():

        print(
            f"  {lang:24s}: "
            f"{count}"
        )

    print()
    print(
        f"输出目录："
        f"{OUTPUT_DIR.resolve()}"
    )

    print()
    print("完成！")


if __name__ == "__main__":
    main()