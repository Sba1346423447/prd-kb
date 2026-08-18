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

## 输出

- `eval/output/evaluation_report.json`：平均分报告
- `eval/output/evaluation_details.csv`：逐样本明细
- `eval/output/generated_samples.jsonl`：检索 + 生成结果缓存，中断后可续跑

## 评估口径说明

- 当前评估直接调用检索器 + 上下文生成，不经过 Agent 工具调度链路。
- `reference_contexts` 使用标准答案整体作为参考上下文，ContextRecall 结果会略偏乐观。
