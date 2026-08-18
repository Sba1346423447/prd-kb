"""
评估数据集构造脚本

从 docs/answerbook.md 中提取 (question, ground_truth) 对，写入
eval/output/eval_dataset.jsonl，供 run_evaluation.py 使用。

数据源结构（已验证）：
    #### <strong>1.1 请详细解释一下 ...？</strong>
    * <strong>参考答案：</strong>
        答案正文...

说明：
- 仅保留 `X.Y` 编号的问题条目（如 1.1、2.3、5.1），自动跳过正文中
  `#### <strong>1. MHA ...</strong>` 这类非问答子标题。
- 过滤答案过短（< 100 字）的质量守卫条目。
- 不调用任何 LLM，纯本地规则提取，可重复运行（覆盖输出文件）。

用法：
    python eval/build_eval_dataset.py
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE = PROJECT_ROOT / "docs" / "answerbook.md"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "eval_dataset.jsonl"
MIN_ANSWER_LENGTH = 100

# 匹配 `#### <strong>问题</strong>` 开头的条目，惰性捕获至下一个 X.Y 编号问题。
# 答案正文里可能还有 `#### <strong>1. Greedy Search ...</strong>` 这类子标题，
# 它们不是新问题，不能作为截止标记。
QA_PATTERN = re.compile(
    r'####\s*<strong>([^<]+)</strong>\s*(.*?)(?=####\s*<strong>\s*\d+\.\d+[^<]*</strong>|$)',
    re.S,
)
# 匹配 `* <strong>参考答案：</strong>` 标记
ANSWER_MARK = re.compile(r"\*?\s*<strong>参考答案[：:]*</strong>")
# 问题编号：X.Y 格式（1.1 / 2.3 / 5.1），用于排除正文子标题
QUESTION_ID = re.compile(r"^\s*\d+\.\d+")


def _strip_html(raw: str) -> str:
    """剥离残留 HTML 标签并压缩空白"""
    text = re.sub(r"<[^>]+>", "", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_qa_from_answerbook(path: Path) -> List[Dict[str, str]]:
    """从 answerbook.md 提取 (question, ground_truth) 对

    Args:
        path: answerbook.md 文件路径

    Returns:
        [{"question": str, "ground_truth": str}] 列表
    """
    text = path.read_text(encoding="utf-8")
    pairs: List[Dict[str, str]] = []
    skipped_non_qa = 0
    skipped_short = 0

    for m in QA_PATTERN.finditer(text):
        raw_question = html.unescape(m.group(1)).strip()
        if not QUESTION_ID.match(raw_question):
            skipped_non_qa += 1
            continue
        body = ANSWER_MARK.sub("", m.group(2))
        body = _strip_html(body)
        if len(body) < MIN_ANSWER_LENGTH:
            skipped_short += 1
            continue
        pairs.append({"question": raw_question, "ground_truth": body})

    print(f"[build_eval_dataset] 源文件: {path}")
    print(f"[build_eval_dataset] 提取 QA 对: {len(pairs)} 条")
    print(f"[build_eval_dataset] 跳过非问答子标题: {skipped_non_qa} 条")
    print(f"[build_eval_dataset] 跳过过短答案(<{MIN_ANSWER_LENGTH}字): {skipped_short} 条")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 RAGAS 评估数据集")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="answerbook.md 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 jsonl 路径")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"[build_eval_dataset] 源文件不存在: {args.source}")
        sys.exit(1)

    pairs = extract_qa_from_answerbook(args.source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[build_eval_dataset] 已写入 {args.output}（{len(pairs)} 条）")


if __name__ == "__main__":
    main()
