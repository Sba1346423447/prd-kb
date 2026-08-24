"""
企业技术支持智研知识库（PRD-KB）—— API 依赖注入

定义全局单例资源持有者 AppState 与鉴权依赖（get_current_user / require），
在 FastAPI 启动时注入全链路组件，路由层通过 Depends 获取共享状态与当前用户。
"""
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.orm import Session

from core.db import get_db
from core.models import User
from core.permissions import Permission, get_role_permissions
from core.security import decode_access_token
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


_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """解析 Bearer Token 并加载当前用户

    Args:
        credentials: HTTP Authorization Bearer 凭证，缺失返回 401
        db: 数据库会话

    Returns:
        User: 当前登录用户 ORM 实例（实时加载，保证角色与存在性最新）

    Raises:
        HTTPException: 未登录 / 令牌无效或过期 / 用户不存在均返回 401
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="未登录或缺少访问令牌")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="访问令牌无效或已过期")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在或已被删除")
    return user


def require(permission: Permission):
    """权限位校验依赖工厂

    Args:
        permission: 需要的权限位，如 Permission.CHAT

    Returns:
        FastAPI 依赖函数：校验当前用户角色包含该权限位，否则返回 403
    """

    def checker(user: User = Depends(get_current_user)) -> User:
        if permission not in get_role_permissions(user.role):
            raise HTTPException(status_code=403, detail=f"缺少权限：{permission.value}")
        return user

    return checker