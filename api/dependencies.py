"""
企业技术支持智研知识库（PRD-KB）—— API 依赖注入

定义全局单例资源持有者 AppState，在 FastAPI 启动时注入全链路组件，
路由层通过 Depends 获取共享状态，避免重复初始化。
"""
from typing import Any, Dict, Optional
from langchain_core.retrievers import BaseRetriever
from core.session_store import SessionStore
from utils.logger import logger


class AppState:
    """全局单例资源持有者

    FastAPI 生命周期 startup 阶段完成初始化，路由通过 Depends(get_app_state) 注入，
    确保 LLM、检索器、Reranker、LangGraph 等组件在整个服务生命周期内只加载一次。
    """

    def __init__(self):
        self.llm: Any = None
        self.retriever: Optional[BaseRetriever] = None
        self.reranker: Any = None
        self.agent_graph: Any = None
        self.retrieval_config: Optional[Dict[str, Any]] = None
        self.session_store: Optional[SessionStore] = None
        self.ready = False

    def init(self, llm, retriever, retrieval_config, reranker, agent_graph):
        """注入全链路组件到全局状态

        Args:
            llm: 大语言模型客户端实例
            retriever: 增强混合检索器实例
            retrieval_config: 检索策略配置字典
            reranker: 重排序器实例，可为 None
            agent_graph: Agent 模式 LangGraph 编译图
        """
        self.llm = llm
        self.retriever = retriever
        self.retrieval_config = retrieval_config
        self.reranker = reranker
        self.agent_graph = agent_graph
        self.session_store = SessionStore()
        self.ready = True
        logger.info("AppState 全局资源初始化完成")


app_state = AppState()


def get_app_state() -> AppState:
    """FastAPI Depends 依赖注入函数，返回全局单例 AppState

    Returns:
        AppState: 持有全部后端组件的全局状态实例
    """
    return app_state