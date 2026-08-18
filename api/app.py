"""
企业技术支持智研知识库（PRD-KB）—— FastAPI 应用工厂

负责 FastAPI 服务实例创建、全链路资源生命周期管理与静态文件服务挂载。
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from utils.logger import logger
from core.config_loader import load_config
from core.knowledge_base import init_knowledge_base
from core.llm_client import init_llm
from core.strategy.retrieval_strategy import build_advanced_retriever
from core.tools import get_rag_tools
from core.agent_chain import build_agent_graph
from core.strategy.rerank_strategy import get_reranker
from api.dependencies import app_state
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时初始化全链路资源，关闭时清理

    启动阶段依次完成：配置加载 -> 知识库初始化 -> LLM 初始化 -> 检索器构建 ->
    Rerank 模型加载 -> Agent 工具注册 -> Agent 图构建 -> 全局状态注入。
    """
    logger.info("FastAPI 服务启动，开始初始化全链路资源...")
    config = load_config()
    chroma_helper = init_knowledge_base(config, dir_path="docs/")
    retrieval_config = config["retrieval"]

    llm_config = config["llm"]
    llm = init_llm(
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
        model_name=llm_config["model_name"],
        temperature=llm_config["temperature"],
        vision_model=llm_config.get("vision_model"),
    )

    retriever = build_advanced_retriever(chroma_helper, retrieval_config)

    reranker = get_reranker(
        model_path=retrieval_config["rerank_model_path"],
        device=retrieval_config.get("device", "cpu")
    ) if retrieval_config.get("enable_rerank", False) else None

    rag_tools = get_rag_tools(retriever, retrieval_config, reranker)
    agent_graph, _ = build_agent_graph(llm, rag_tools)

    app_state.init(llm, retriever, retrieval_config, reranker, agent_graph)
    logger.info("FastAPI 全链路资源初始化完成，服务就绪")

    yield

    logger.info("FastAPI 服务关闭")


def create_app() -> FastAPI:
    """FastAPI 应用工厂函数

    创建 FastAPI 实例，挂载静态文件目录与根路由，挂载 API 路由。

    Returns:
        配置完成的 FastAPI 应用实例
    """
    app = FastAPI(
        title="企业技术支持智研知识库（PRD-KB）",
        description="基于 LangGraph + Chroma 的私有化研发技术支持智能问答 API",
        version="1.0.0",
        lifespan=lifespan
    )
    app.include_router(router)

    static_dir = Path(__file__).parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(docs_dir)), name="media")

    @app.get("/")
    async def root():
        return HTMLResponse(content=(static_dir / "index.html").read_text(encoding="utf-8"))

    return app
