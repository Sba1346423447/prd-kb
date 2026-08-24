# RAGAS 效果评估

评估 PRD-KB 的“检索 + 生成”链路质量，输出以下 RAGAS 指标：

- Faithfulness：回答忠实度
- AnswerRelevancy：回答相关性
- ContextPrecision：检索上下文精确率
- ContextRecall：检索上下文召回率
- AnswerCorrectness：回答正确度（可选）

## 为什么需要独立环境

ragas 0.2.x 依赖 `langchain_community.chat_models.vertexai`，而主项目锁定的
`langchain-community==0.4.2` 已移除该模块。因此评估必须使用 `eval_venv`，
不要直接在主项目环境里运行 `eval/run_evaluation.py`。

## 初始化评估环境

```powershell
python -m venv eval_venv
eval_venv\Scripts\python.exe -m pip install -r eval\requirements.txt
```

## 1. 构建评估数据集

```powershell
eval_venv\Scripts\python.exe eval\build_eval_dataset.py
```

脚本会从 `docs/answerbook.md` 中提取 `X.Y` 编号的 QA 对，写入
`eval/output/eval_dataset.jsonl`。

## 2. 运行评估

先小规模验证：

```powershell
eval_venv\Scripts\python.exe eval\run_evaluation.py --limit 10
```

全量评估：

```powershell
eval_venv\Scripts\python.exe eval\run_evaluation.py
```

常用参数：

- `--limit N`：只评估前 N 条
- `--no-answer-correctness`：跳过较慢的 AnswerCorrectness
- `--no-cache`：忽略生成缓存，重新检索并生成
- `--force-regenerate`：缓存匹配也重新生成
- `--strict`：指标异常时直接抛错，而不是静默输出 NaN
- `--no-rerank`：跳过 Rerank 重排（对照实验）；默认遵循配置 `retrieval.enable_rerank`
- `--tag NAME`：报告/明细文件名后缀，用于区分对照实验产物

## Rerank 对照实验

评估链路与线上工具行为对齐（去重 → Rerank → `max_result_docs`/`max_result_chars` 截断），
可通过开关对比有无 Rerank 的检索质量差异：

```powershell
# 实验组：Rerank 开启（遵循配置 enable_rerank: true）
eval_venv\Scripts\python.exe eval\run_evaluation.py --limit 10 --tag rerank_on

# 对照组：Rerank 关闭
eval_venv\Scripts\python.exe eval\run_evaluation.py --limit 10 --no-rerank --tag rerank_off
```

两组的报告分别输出到 `evaluation_report_rerank_on.json` / `evaluation_report_rerank_off.json`；
生成缓存按「问题 + 检索变体」联合匹配，两组互不串用。

## 输出

- `eval/output/evaluation_report.json`：平均分报告（带 `--tag` 时为 `evaluation_report_{tag}.json`）
- `eval/output/evaluation_details.csv`：逐样本明细（同上按 tag 区分）
- `eval/output/generated_samples.jsonl`：检索 + 生成结果缓存，中断后可续跑

## 评估口径说明

- 评估不经过 Agent 决策链路，但检索后处理与线上工具保持一致：去重 → Rerank（可关）→ 限流截断。
- `reference_contexts` 使用标准答案整体作为参考上下文，ContextRecall 结果会略偏乐观。
