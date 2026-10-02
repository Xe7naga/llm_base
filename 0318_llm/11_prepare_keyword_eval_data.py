"""
按照evalscope要求，准备数据集，用以进行评估测试
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_PROMPT_PREFIX = """/no_think
请从下面文本中抽取关键词。
要求：
1. 必须输出中文关键词，禁止翻译成英文。
2. 关键词应优先来自原文中的中文术语或规范中文概念。
3. 只输出关键词，不要输出解释、编号、JSON 或 Markdown。
4. 多个关键词用英文分号 ; 分隔。
5. 第一个关键词前不要加分号，最后一个关键词后不要加分号。
6. 不要输出整句、摘要或推理过程。
7. 如果原文包含标题，优先抽取标题和正文中反复出现的核心学术概念。

示例：
文本：
抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别.该种类识别属于模式识别,其关键在于提取样品的特征参数,在获得特征参数的基础上,选取合适的网络通过训练来进行识别.猕猴桃种类识别的研究为自动化识别果品的种类、品种和新鲜等级等提供了一种新方法,为进一步研究果品介电特性与其内在品质的关系提供了一定的理论与实践基础.
输出：
食品科学技术基础学科;猕猴桃;应用;人工神经网络;介电特性;识别

请完成以下文本的提取：


"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/keywords_data_test.jsonl")
    parser.add_argument(
        "--output",
        default="benchmark/custom_eval/text/qa/keyword_extraction.jsonl",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def convert_example(example: dict) -> dict:
    conversation = example["conversation"][0]
    query = f"{DEFAULT_PROMPT_PREFIX}{conversation['human']}"

    item = {
        "query": query,
    }
    return item


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files=args.input,
        split="train",
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    converted = dataset.map(
        convert_example,
        remove_columns=dataset.column_names,
    )

    converted.to_json(args.output,force_ascii=False)


if __name__ == "__main__":
    main()