"""
PRD-KB RAGAS 评估脚本

复用 core/ 的初始化函数完成“检索 + 生成”，再由 RAGAS 对检索与生成质量进行量化打分。

评估口径：
- 不经过 Agent 链路，直接调用 retriever 检索 + LLM 基于上下文生成，聚焦评估“检索质量 + 生成忠实度”。
- reference_contexts 采用近似方案：将 ground_truth 整体作为参考上下文（ContextRecall 略偏乐观，报告会注明）。
- 默认启用生成结果缓存，中断后再次运行会自动复用已生成的样本。

用法（必须使用独立评估环境，主项目环境与 ragas 不兼容）：
    eval_venv\\Scripts\\python.exe eval\\run_evaluation.py
    eval_venv\\Scripts\\python.exe eval\\run_evaluation.py --limit 10
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

# 必须在导入 ragas 之前检测环境：主项目环境（langchain-community==0.4.2）
# 缺少 ragas 依赖的 langchain_community.chat_models.vertexai。
try:
    from langchain_community.chat_models import vertexai  # noqa: F401
except ImportError as exc:
    print("[run_evaluation] 检测到当前 Python 环境与 RAGAS 不兼容：")
    print(f"[run_evaluation] {exc}")
    print("[run_evaluation] 请使用独立评估环境运行：")
    print("[run_evaluation]   eval_venv\\Scripts\\python.exe eval\\run_evaluation.py")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from langchain_core.documents import Document
from langchain_core.language_models.llms import LLMResult
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from core.config_loader import load_config
from core.knowledge_base import init_knowledge_base
from core.strategy.retrieval_strategy import build_advanced_retriever

EVAL_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_DATASET = EVAL_OUTPUT_DIR / "eval_dataset.jsonl"
DEFAULT_REPORT = EVAL_OUTPUT_DIR / "evaluation_report.json"
DEFAULT_DETAILS = EVAL_OUTPUT_DIR / "evaluation_details.csv"
DEFAULT_CACHE = EVAL_OUTPUT_DIR / "generated_samples.jsonl"

REQUIRED_FIELDS = ("question", "ground_truth")

GEN_PROMPT = (
    "请严格依据以下资料回答问题，不要编造、不要扩展资料外的信息。\n\n"
    "资料：\n{contexts}\n\n"
    "问题：{question}"
)

# 豆包 API 在长输出时会返回 finish_reason="length"（达到 max_tokens 截断），
# ragas 默认 is_finished 不认可该值，会把结果当作“未完成”导致无限重试直至超时。
# 自定义 parser：将 "length" 视为正常完成（输出虽截断但结果有效）。
_FINISH_REASONS_OK = {"stop", "STOP", "MAX_TOKENS", "eos_token", "length"}


def doubao_is_finished(response: LLMResult) -> bool:
    """豆包兼容的 is_finished：接受 stop / length 等常见完成原因。"""
    if not response.generations:
        return True
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]
        info = resp.generation_info or {}
        finish_reason = info.get("finish_reason")
        if finish_reason is None:
            is_finished_list.append(True)
        else:
            is_finished_list.append(finish_reason in _FINISH_REASONS_OK)
    return all(is_finished_list)


def load_eval_dataset(dataset_path: Path) -> List[dict]:
    """读取评估数据集并做基本校验。"""
    if not dataset_path.exists():
        print(f"[run_evaluation] 数据集不存在: {dataset_path}")
        print("[run_evaluation] 请先运行 eval_venv\\Scripts\\python.exe eval\\build_eval_dataset.py")
        sys.exit(1)

    items = []
    with open(dataset_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[run_evaluation] 数据集第 {line_number} 行不是合法 JSON: {exc}")
                sys.exit(1)
            items.append(item)

    if not items:
        print("[run_evaluation] 数据集为空，请先运行 eval_venv\\Scripts\\python.exe eval\\build_eval_dataset.py")
        sys.exit(1)

    for idx, item in enumerate(items, start=1):
        missing = [
            field
            for field in REQUIRED_FIELDS
            if not isinstance(item.get(field), str) or not item[field].strip()
        ]
        if missing:
            print(f"[run_evaluation] 数据集第 {idx} 条缺少字段: {missing}")
            sys.exit(1)
    return items


def build_retriever(config: dict):
    """初始化知识库并构建检索器，同时复用已加载的嵌入模型。"""
    chroma_helper = init_knowledge_base(config, dir_path="docs/")
    retriever = build_advanced_retriever(chroma_helper, config["retrieval"])
    return retriever, chroma_helper.embedding_function


def build_llm(config: dict, json_mode: bool = False) -> ChatOpenAI:
    """构建豆包 LLM 客户端（OpenAI 兼容协议）。

    json_mode=True 时启用 response_format=json_object，供 RAGAS 指标打分使用。
    """
    llm_cfg = config["llm"]
    kwargs: dict = dict(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model_name"],
        temperature=0,
        max_tokens=8192,
        request_timeout=300,
        max_retries=3,
    )
    if json_mode:
        # doubao-seed 默认开启深度思考，会耗尽 max_tokens 导致实际输出为空；
        # thinking 是火山方舟非标准参数，必须经 extra_body 传递。
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)


def _reference_hash(reference: str) -> str:
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()


def load_generated_cache(cache_path: Path) -> Dict[str, dict]:
    """加载已生成的样本缓存，按问题文本索引。"""
    cache: Dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample, dict) and sample.get("user_input"):
                cache[sample["user_input"]] = sample
    return cache


def append_cache_sample(cache_path: Path, sample: dict) -> None:
    """追加一条生成样本到缓存。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def generate_samples(
    eval_dataset: List[dict],
    retriever: BaseRetriever,
    gen_llm: ChatOpenAI,
    cache_path: Path,
    use_cache: bool = True,
    force_regenerate: bool = False,
) -> List[SingleTurnSample]:
    """逐条检索 + 生成，支持缓存断点续跑。"""
    cache = load_generated_cache(cache_path) if use_cache else {}
    samples = []

    for i, item in enumerate(eval_dataset, start=1):
        question = item["question"]
        reference = item["ground_truth"]
        reference_hash = _reference_hash(reference)
        cached = cache.get(question)

        if (
            use_cache
            and not force_regenerate
            and cached
            and cached.get("reference_hash") == reference_hash
        ):
            sample_dict = cached
            print(f"[run_evaluation] [{i}/{len(eval_dataset)}] 使用缓存: {question[:40]}...")
        else:
            print(f"[run_evaluation] [{i}/{len(eval_dataset)}] 检索+生成: {question[:40]}...")
            ctx_docs: List[Document] = retriever.invoke(question)
            contexts = [doc.page_content for doc in ctx_docs]
            answer_content = gen_llm.invoke(
                GEN_PROMPT.format(contexts="\n\n".join(contexts), question=question)
            ).content
            if isinstance(answer_content, list):
                answer = "".join(
                    part if isinstance(part, str) else str(part)
                    for part in answer_content
                )
            else:
                answer = str(answer_content)
            sample_dict = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
                "reference_contexts": [reference],
                "reference_hash": reference_hash,
            }
            if use_cache:
                append_cache_sample(cache_path, sample_dict)
                cache[question] = sample_dict

        samples.append(
            SingleTurnSample(
                user_input=sample_dict["user_input"],
                response=sample_dict["response"],
                retrieved_contexts=sample_dict["retrieved_contexts"],
                reference=sample_dict["reference"],
                reference_contexts=sample_dict["reference_contexts"],
            )
        )

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="PRD-KB RAGAS 评估")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="评估数据集路径")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 条（0=全部）")
    parser.add_argument(
        "--no-answer-correctness",
        action="store_true",
        help="移除 AnswerCorrectness 指标（打分较慢时使用）",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="忽略已有生成缓存，重新检索并生成",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="即使缓存匹配也重新生成当前问题",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="RAGAS 指标异常时直接抛错，而不是输出 NaN",
    )
    args = parser.parse_args()

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_dataset = load_eval_dataset(args.dataset)
    if args.limit and args.limit > 0:
        eval_dataset = eval_dataset[: args.limit]
    print(f"[run_evaluation] 评估样本数: {len(eval_dataset)}")

    print("[run_evaluation] 加载配置...")
    config = load_config()

    print("[run_evaluation] 初始化知识库与检索器...")
    retriever, emb_model = build_retriever(config)

    print("[run_evaluation] 初始化生成 LLM...")
    gen_llm = build_llm(config)

    samples = generate_samples(
        eval_dataset,
        retriever,
        gen_llm,
        cache_path=DEFAULT_CACHE,
        use_cache=not args.no_cache,
        force_regenerate=args.force_regenerate,
    )
    if not samples:
        print("[run_evaluation] 未生成任何评估样本，退出")
        sys.exit(1)

    print("[run_evaluation] RAGAS 指标打分中，请耐心等待...")
    ragas_llm = build_llm(config, json_mode=True)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]
    if not args.no_answer_correctness:
        metrics.append(AnswerCorrectness())

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(ragas_llm, is_finished_parser=doubao_is_finished),
        embeddings=LangchainEmbeddingsWrapper(emb_model),
        # 豆包 API 响应较慢：faithfulness 等指标对单样本需多次 LLM 调用，
        # 提高单指标总超时、降低并发以避免限流，并增加重试容错。
        run_config=RunConfig(
            timeout=900,
            max_retries=8,
            max_wait=300,
            max_workers=2,
        ),
        raise_exceptions=args.strict,
    )

    df = result.to_pandas()
    details_path = EVAL_OUTPUT_DIR / "evaluation_details.csv"
    report_path = EVAL_OUTPUT_DIR / "evaluation_report.json"
    df.to_csv(details_path, index=False)

    metric_names = [m.name for m in metrics if m.name in df.columns]
    avg = {}
    nan_counts = {}
    for col in metric_names:
        score = float(df[col].mean(skipna=True))
        avg[col] = round(score, 4) if math.isfinite(score) else None
        nan_counts[col] = int(df[col].isna().sum())

    scored_sample_count = 0
    if metric_names:
        scored_sample_count = int(df[metric_names].notna().all(axis=1).sum())

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": len(samples),
        "scored_sample_count": scored_sample_count,
        "metrics": avg,
        "nan_counts": nan_counts,
        "note": "reference_contexts 为近似口径：以标准答案整体作为参考上下文，ContextRecall 略偏乐观",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("[run_evaluation] 评估完成")
    print(f"样本数: {len(samples)} | 完整打分样本: {scored_sample_count}")
    print("平均分：")
    for name, score in avg.items():
        print(f"  {name}: {score}")
    print(f"\n报告: {report_path}")
    print(f"明细: {details_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
