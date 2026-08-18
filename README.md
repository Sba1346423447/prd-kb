# 企业技术支持智研知识库（PRD-KB）

基于 **LangGraph + ChromaDB** 构建的私有化多格式文档智能问答系统，采用 **Agent 自主检索模式**，LLM 借助 Function Calling 自主决策是否调用知识库检索、日志分析、字数统计等工具，提供命令行交互与 FastAPI Web 服务两种入口，内置 SSE 流式输出与前端页面。

---

## 核心特性

- **Agent 自主决策**：LLM 借助 Function Calling 自主判断是否需要检索知识库，智能适配专业问答、日志排障、日常闲聊等不同场景
- **多路混合检索**：向量语义检索 + BM25 关键词检索 → RRF 倒数排名融合 → BGE Reranker 精排 → 相邻 chunk 上下文扩展
- **表格感知优化**：Excel 文档在分块、检索、重排全链路有特殊标记与加权
- **多格式文档支持**：PDF / TXT / DOCX / Markdown / XLSX / 图片（PNG/JPG/WebP/BMP）
- **智能分块策略**：按文件类型自动选择最优分块方案（Markdown 标题层级、Excel 表格、TXT 段落优先、图片整块、通用递归）
- **多模态输入**：支持图片问答、图片文档入库与检索回显，视觉模型可插拔配置
- **会话记忆**：SQLite 业务表持久化对话历史，服务重启后自动重建上下文，历史记录可查询审计
- **SSE 流式输出**：逐 token 推送回答 + 工具调用过程实时展示
- **Web 前端**：类 ChatGPT 单页 UI，支持会话管理、Markdown 渲染、思考过程可视化
- **Docker 部署**：一键容器化启动，环境变量自动覆盖本地路径

---

## 技术栈

| 层面 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| LLM 框架 | LangChain 1.x, LangGraph |
| 向量数据库 | ChromaDB |
| 会话存储 | SQLite（对话历史持久化） |
| 嵌入模型 | BGE-small-zh-v1.5 |
| 重排序模型 | BGE-reranker-v2-m3 (BAAI) |
| LLM | OpenAI 协议兼容（豆包 / 通义千问 / DeepSeek 等） |
| 推理后端 | PyTorch + HuggingFace Transformers |
| Web API | FastAPI + Uvicorn + SSE |
| 前端 | 原生 HTML/CSS/JS |
| 部署 | Docker / Docker Compose |

---

## 项目结构

```
prd-kb/
├── main.py                     # 命令行交互入口
├── run_api.py                  # FastAPI 服务启动入口
├── config/
│   ├── settings.yaml           # 运行配置（含密钥，已 .gitignore）
│   └── settings.example.yaml   # 配置模板（提交到仓库）
├── api/                        # FastAPI 层
│   ├── app.py                  # 应用工厂 + lifespan 生命周期
│   ├── dependencies.py         # 全局资源单例 AppState
│   ├── routes.py               # 路由: /health, /chat, /chat/stream
│   └── schemas.py              # Pydantic 请求/响应模型
├── core/                       # 核心 RAG 引擎
│   ├── config_loader.py        # YAML 配置加载（环境变量覆盖）
│   ├── knowledge_base.py       # 知识库初始化编排（加载→清洗→分块→入库）
│   ├── document_loader.py      # 多格式文档加载（含图片）
│   ├── document_clean.py       # 文本清洗降噪
│   ├── embedding.py            # BGE 嵌入模型加载
│   ├── vector_store.py         # ChromaDB 封装
│   ├── llm_client.py           # LLM 客户端初始化
│   ├── agent_chain.py          # ReAct Agent LangGraph 链路
│   ├── session_store.py        # 会话历史 SQLite 持久化
│   ├── tools.py                # Agent 工具集
│   ├── multimodal.py           # 多模态消息/图片内容构建
│   ├── image_processor.py      # 图片 OCR 与内容描述
│   └── strategy/               # 策略层（策略模式）
│       ├── base_strategy.py
│       ├── chunk_strategy.py        # 分块策略
│       ├── retrieval_strategy.py    # 检索策略（混合检索+RRF+上下文扩展）
│       └── rerank_strategy.py       # 重排序策略
├── prompts/                    # 提示词模板
│   └── agent_prompt.py
├── utils/
│   ├── logger.py               # 统一日志（控制台 + 滚动文件）
│   └── exceptions.py           # 自定义分层异常
├── static/
│   ├── index.html              # Web 前端页面
│   ├── css/
│   │   └── style.css           # 全局样式
│   └── js/
│       └── app.js              # 前端交互脚本
├── docs/                       # 知识库文档目录（已 .gitignore）
├── eval/                       # RAGAS 效果评估脚本（独立 venv）
├── docker/
│   ├── Dockerfile              # 容器镜像构建
│   └── docker-compose.yml      # 一键编排启动
├── requirements.txt
├── pyproject.toml
└── LICENSE                    # MIT 开源许可
```

---

## 快速开始

### 环境要求

- Python 3.10 ~ 3.12
- 本地已下载 BGE 嵌入模型与 Reranker 模型（或可访问 HuggingFace）
- 可用的大模型 API Key（兼容 OpenAI 协议）

### 1. 克隆项目

```bash
git clone https://github.com/Sba1346423447/prd-kb.git
cd prd-kb
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

复制配置模板并修改：

```bash
cp config/settings.example.yaml config/settings.yaml
```

编辑 `config/settings.yaml`：
- 填入大模型 `api_key`、`base_url`、`model_name`
- 配置本地嵌入模型和 Reranker 模型路径
- 按需调整分块大小、检索数量等参数

### 4. 放入知识库文档

将需要入库的文档放入 `docs/` 目录（支持 .pdf / .txt / .docx / .md / .xlsx，以及 .png / .jpg / .jpeg / .webp / .bmp 图片）。

---

## 启动方式

### 方式一：命令行交互

```bash
python main.py
```

### 方式二：Web API 服务

```bash
python run_api.py
```

启动后访问：
- 前端页面：http://localhost:8000
- API 文档：http://localhost:8000/docs

**API 接口：**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat` | POST | 同步问答（返回完整回答） |
| `/chat/stream` | POST | 流式问答（SSE 逐 token 推送） |

请求示例：

```json
{
  "question": "知识库怎么搭建？",
  "session_id": "user_001"
}
```

> `images` 为可选字段：传入 `data:image/` 开头的 data URL 数组（最多 4 张），即可发起多模态图片问答。

### 方式三：Docker 部署

```bash
cd docker
docker-compose up -d
```

> 首次启动前请修改 `docker-compose.yml` 中的模型挂载路径为本地实际路径。

---

## RAGAS 效果评估

项目内置 RAGAS 评估脚本，用于量化“检索 + 生成”链路质量，覆盖 Faithfulness、
AnswerRelevancy、ContextPrecision、ContextRecall 和 AnswerCorrectness 指标。

```bash
# 1. 创建独立评估环境并安装依赖
python -m venv eval_venv
eval_venv/Scripts/python.exe -m pip install -r eval/requirements.txt

# 2. 从 docs/answerbook.md 构建评估数据集
eval_venv/Scripts/python.exe eval/build_eval_dataset.py

# 3. 先小规模验证，再全量评估
eval_venv/Scripts/python.exe eval/run_evaluation.py --limit 10
eval_venv/Scripts/python.exe eval/run_evaluation.py
```

评估结果输出到 `eval/output/`，详细说明见 `eval/README.md`。

---

## 多模态支持

支持图片问答与图片文档检索，多模态链路覆盖：

- 聊天图片输入：前端支持选择图片并以 data URL 传给后端，Agent 以多模态消息完成理解与回答。视觉模型名称通过 `config/settings.yaml` 的 `llm.vision_model` 配置，留空时回退使用 `llm.model_name`。
- 文档图片检索：将 `png/jpg/jpeg/webp/bmp` 放入 `docs/` 后自动入库，`core/image_processor.py::describe_image()` 复用 `llm.vision_model` 对图片做 OCR 与内容描述。
- 检索回显：命中图片以图片数据块传给 Agent，并在回答中保留 `/media/...` Markdown 图片引用。

> 图片内容识别依赖 `llm.vision_model` 配置的视觉模型；未配置时回退使用 `llm.model_name`。

---

## 工作流概览

```
文档入库流水线:
  docs/ → load_document() → clean_raw_text() → get_document_splitter()
  → add_texts() → ChromaDB 持久化
  图片分支: 图片文件 → describe_image() 生成描述 → 整块入库

问答链路（Agent 模式）:
  用户问题 → agent_node(LLM 决策)
    ├─ 需检索? → knowledge_base_search → agent_node → ... → 回答
    ├─ 日志分析? → log_analysis → agent_node → ... → 回答
    ├─ 字数统计? → count_text_characters → agent_node → ... → 回答
    └─ 无需工具? → 直接回答
```

---

## 项目亮点

- **策略模式架构**：分块 / 检索 / 重排均采用抽象基类 + 多实现，按文档类型自动选择最优策略，易于扩展
- **全链路容错**：单文档处理失败跳过、单工具调用异常不中断链路、最大检索次数防死循环
- **配置与代码解耦**：所有模型参数、业务参数统一 YAML 管理，Docker 环境变量覆盖
- **标准化工程**：分层异常体系、双输出日志系统、函数文档与注释完备

---

## License

本项目采用 [MIT](LICENSE) 开源协议。
