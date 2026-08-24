# 企业技术支持智研知识库（PRD-KB）

基于 **LangGraph + ChromaDB + MySQL** 构建的私有化多格式文档智能问答系统。提供 **Agent 自主检索 / Pure 固定流水线直出 / Auto 规则路由** 三种检索模式，内置 **JWT 认证与权限位授权**、**83 项自动化测试**、**GitHub Actions CI** 与 **Alembic 版本化迁移**，支持 Docker Compose 一键部署上线。

---

## 核心特性

- **三种检索模式**：Agent 模式由 LLM 自主决策检索与工具调用；Pure 模式固定单次检索直出（响应更快）；Auto 模式按问题特征（图片/工具意图/复杂度/长度）规则路由自动选择
- **多路混合检索**：向量语义检索 + BM25 关键词检索（jieba 中文分词）→ RRF 倒数排名融合 → BGE Reranker 精排 → 相邻 chunk 上下文扩展
- **表格感知优化**：Excel 文档在分块、检索、重排全链路有特殊标记与加权
- **多格式文档支持**：PDF / TXT / DOCX / Markdown / XLSX / 图片（PNG/JPG/WebP/BMP）
- **智能分块策略**：按文件类型自动选择最优分块方案（Markdown 标题层级、Excel 表格、TXT 段落优先、图片整块、通用递归）
- **多模态输入**：支持图片问答、图片文档入库与检索回显，视觉模型可插拔配置
- **认证与权限**：JWT 登录态 + bcrypt 密码哈希 + 权限位模型（KB_MANAGE / CHAT / SESSION_MANAGE），会话归属服务端校验，跨用户访问 403
- **MySQL 状态外置**：用户/会话/消息三表迁移至 MySQL（Alembic 版本化迁移，MEDIUMTEXT 容纳多模态历史），支撑多实例水平扩展
- **SSE 流式输出**：逐 token 推送回答 + 工具调用过程实时展示
- **Web 前端**：类 ChatGPT 单页 UI，登录、会话管理、模式切换、Markdown 渲染、思考过程可视化
- **质量保障**：83 项 pytest 测试（核心链路 / API / 鉴权 / 模式路由），GitHub Actions CI（ruff + pytest）
- **Docker 部署**：多阶段构建 + healthcheck + 环境变量全配置化，一键容器化启动

---

## 技术栈

| 层面 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| LLM 框架 | LangChain 1.x, LangGraph |
| 向量数据库 | ChromaDB |
| 业务数据库 | MySQL 8.x（SQLAlchemy ORM + Alembic 迁移；测试用 SQLite 内存库） |
| 认证授权 | PyJWT（HS256）+ bcrypt + 权限位模型 |
| 嵌入模型 | BGE-small-zh-v1.5 |
| 重排序模型 | BGE-reranker-v2-m3 (BAAI) |
| 中文分词 | jieba（BM25 检索路） |
| LLM | OpenAI 协议兼容（豆包 / 通义千问 / DeepSeek 等） |
| 推理后端 | PyTorch + HuggingFace Transformers |
| Web API | FastAPI + Uvicorn + SSE |
| 前端 | 原生 HTML/CSS/JS |
| 测试与 CI | pytest（83 用例）+ ruff + GitHub Actions |
| 部署 | Docker / Docker Compose |

---

## 项目结构

```
prd-kb/
├── main.py                     # 命令行交互入口
├── run_api.py                  # FastAPI 服务启动入口
├── alembic.ini                 # Alembic 迁移配置
├── config/
│   ├── settings.yaml           # 运行配置（含密钥，已 .gitignore）
│   └── settings.example.yaml   # 配置模板（提交到仓库）
├── api/                        # FastAPI 层
│   ├── app.py                  # 应用工厂 + lifespan 生命周期
│   ├── auth.py                 # 登录接口（JWT 签发）
│   ├── dependencies.py         # 全局资源单例 AppState + 鉴权依赖
│   ├── routes.py               # 路由: /health, /chat, /chat/stream, /sessions
│   └── schemas.py              # Pydantic 请求/响应模型
├── core/                       # 核心 RAG 引擎
│   ├── config_loader.py        # YAML 配置加载（环境变量覆盖）
│   ├── db.py                   # MySQL 引擎/会话工厂（环境变量组装连接串）
│   ├── models.py               # ORM 模型：User / ChatSession / Message
│   ├── security.py             # bcrypt 密码哈希 + JWT 签发校验
│   ├── permissions.py          # 角色权限位定义
│   ├── mode_router.py          # Auto 模式规则路由
│   ├── knowledge_base.py       # 知识库初始化编排（加载→清洗→分块→入库）
│   ├── document_loader.py      # 多格式文档加载（含图片）
│   ├── document_clean.py       # 文本清洗降噪
│   ├── embedding.py            # BGE 嵌入模型加载
│   ├── vector_store.py         # ChromaDB 封装（含嵌入前文本规范化）
│   ├── llm_client.py           # LLM 客户端初始化
│   ├── agent_chain.py          # ReAct Agent LangGraph 链路
│   ├── session_store.py        # 会话历史 MySQL 持久化
│   ├── tools.py                # Agent 工具集 + 公共检索流水线
│   ├── multimodal.py           # 多模态消息/图片内容构建
│   ├── image_processor.py      # 图片 OCR 与内容描述
│   └── strategy/               # 策略层（策略模式）
│       ├── base_strategy.py
│       ├── chunk_strategy.py        # 分块策略
│       ├── retrieval_strategy.py    # 检索策略（混合检索+RRF+上下文扩展，jieba 中文分词）
│       └── rerank_strategy.py       # 重排序策略
├── prompts/                    # 提示词模板
│   ├── agent_prompt.py         # Agent 模式提示词
│   └── direct_prompt.py        # Pure 直出模式提示词
├── migrations/                 # Alembic 版本化迁移
│   └── versions/               # 0001 建表 / 0002 content 扩容 MEDIUMTEXT
├── tests/                      # 83 项自动化测试
│   ├── conftest.py             # SQLite 内存库 + 测试依赖注入
│   ├── test_chunk_strategy.py  # 分块策略（12 用例）
│   ├── test_retrieval_strategy.py  # 检索策略（10 用例）
│   ├── test_rerank_strategy.py # 重排序策略（5 用例）
│   ├── test_chat_api.py        # API/鉴权/模式路由（20 用例）
│   ├── test_security.py        # 密码哈希/JWT（若干用例）
│   ├── test_mode_router.py     # Auto 路由规则（13 用例）
│   └── test_vector_store.py    # 文本规范化（10 用例）
├── .github/workflows/ci.yml    # GitHub Actions CI（ruff + pytest）
├── docs/                       # 知识库文档目录（用户数据，已 .gitignore）
├── scripts/
│   ├── init_admin.py           # 初始化管理员账号
│   └── verify_docker_config.py # Docker 配置静态校验
├── utils/
│   ├── logger.py               # 统一日志（控制台 + 滚动文件）
│   └── exceptions.py           # 自定义分层异常
├── static/
│   ├── index.html              # Web 前端页面（含登录）
│   ├── css/
│   │   └── style.css           # 全局样式
│   └── js/
│       └── app.js              # 前端交互脚本
├── eval/                       # RAGAS 效果评估脚本（独立 venv）
├── docker/
│   ├── Dockerfile              # 多阶段构建（CPU torch）
│   ├── docker-compose.yml      # MySQL + rag 服务编排（healthcheck/卷/日志轮转）
│   └── .env.example            # 环境变量模板
├── requirements.txt
├── requirements-dev.txt        # 测试/lint 依赖
├── pyproject.toml              # pytest 配置
└── LICENSE                     # MIT 开源许可
```

---

## 快速开始

### 环境要求

- Python 3.10 ~ 3.12
- MySQL 8.x（或直接使用 Docker Compose 内置的 MySQL 服务）
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
# 开发/测试环境（可选）
pip install -r requirements-dev.txt
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

配置数据库环境变量（PowerShell 示例）：

```powershell
$env:DB_PASSWORD='your-mysql-password'   # 默认 root@localhost:3306/prd_kb
$env:JWT_SECRET='your-random-secret'     # JWT 签名密钥
```

### 4. 初始化数据库与管理员

```bash
# 建库（MySQL 中执行）：CREATE DATABASE prd_kb CHARACTER SET utf8mb4;
# 应用迁移（建表）
python -m alembic upgrade head
# 创建管理员账号
python scripts/init_admin.py admin YourPassword
```

### 5. 放入知识库文档

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
- 前端页面：http://localhost:8000 （使用 init_admin 创建的账号登录）
- API 文档：http://localhost:8000/docs

**API 接口：**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查（无鉴权，供探活） |
| `/login` | POST | 登录，返回 JWT 访问令牌 |
| `/me` | GET | 当前用户信息与权限位 |
| `/sessions` | GET/POST | 会话列表 / 创建会话（服务端生成 ID） |
| `/chat` | POST | 同步问答（返回完整回答） |
| `/chat/stream` | POST | 流式问答（SSE 逐 token 推送） |

请求示例（需携带 `Authorization: Bearer <token>`）：

```json
{
  "question": "知识库怎么搭建？",
  "session_id": "由 POST /sessions 返回的会话 ID",
  "mode": "auto"
}
```

> - `mode` 可选 `agent`（默认）/ `pure` / `auto`
> - `images` 为可选字段：传入 `data:image/` 开头的 data URL 数组（最多 4 张），即可发起多模态图片问答

### 方式三：Docker 部署

```bash
cd docker
cp .env.example .env    # 按本机情况修改模型路径与数据库密码
docker compose up -d --build
docker compose exec rag python scripts/init_admin.py admin YourPassword
```

MySQL 随 compose 拉起，容器启动自动执行 Alembic 迁移；`docker compose ps` 等待 rag 服务转 `healthy`（含模型加载时间）后即可访问 http://localhost:8000。

---

## 测试与 CI

```bash
python -m pytest tests/ -q        # 83 项测试，零模型权重依赖
```

覆盖分块五路策略、RRF 混合检索、rerank、鉴权门禁、权限位、会话隔离（跨用户 403）、Pure/Auto 模式全链路、BM25 中文分词回归等。推送 GitHub 后 Actions 自动执行 ruff lint + 全量测试。

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
  → add_texts()（嵌入前去 emoji/压缩空白）→ ChromaDB 持久化
  图片分支: 图片文件 → describe_image() 生成描述 → 整块入库

问答链路（Agent 模式）:
  用户问题 → agent_node(LLM 决策)
    ├─ 需检索? → knowledge_base_search → agent_node → ... → 回答
    ├─ 日志分析? → log_analysis → agent_node → ... → 回答
    ├─ 字数统计? → count_text_characters → agent_node → ... → 回答
    └─ 无需工具? → 直接回答

问答链路（Pure 模式）:
  用户问题 → 单次混合检索（与 Agent 共用同一流水线）
  → 检索上下文注入系统提示词 → LLM 直接生成

问答链路（Auto 模式）:
  规则路由（图片/工具意图/复杂度/长度）→ 落到 Agent 或 Pure 链路
```

---

## 项目亮点

- **策略模式架构**：分块 / 检索 / 重排均采用抽象基类 + 多实现，按文档类型自动选择最优策略，易于扩展
- **双检索范式可切换**：Agent 自主决策与固定流水线直出共用同一检索管道（差异仅在决策层），支持规则路由自动选择，两种范式的质量/延迟权衡可实测对比
- **全链路容错**：单文档处理失败跳过、单工具调用异常不中断链路、最大检索次数熔断 + 检索结果去重防上下文膨胀
- **配置与代码解耦**：所有模型参数、业务参数统一 YAML 管理，Docker 环境变量覆盖
- **工程完整度**：JWT 认证 + 权限位 + 会话归属隔离、MySQL 状态外置（Alembic 迁移）、83 项自动化测试、GitHub Actions CI、Docker 多阶段构建与 healthcheck，达到可上线标准
- **中文检索优化**：BM25 路 jieba 词级分词（修复了中文整句单 token 导致关键词检索失效的问题）、嵌入前文本规范化、表格全链路加权

---

## License

本项目采用 [MIT](LICENSE) 开源协议。
