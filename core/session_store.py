"""
会话历史持久化模块（MySQL）

状态外置改造：会话由服务端生成并归属用户，消息挂接会话，
数据存 MySQL 支撑多实例部署。对外保留 load_history / save_message 原接口，
新增会话创建、列表与归属校验方法。
"""
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.db import SessionLocal
from core.models import ChatSession, Message
from utils.logger import logger

HISTORY_LIMIT = 20
DEFAULT_SESSION_TITLE = "新会话"


class SessionStore:
    """基于 MySQL 的会话历史存储

    Attributes:
        session_factory: SQLAlchemy 会话工厂，默认使用 core.db.SessionLocal
    """

    def __init__(self, session_factory=SessionLocal):
        self._session_factory = session_factory
        logger.info("MySQL 会话存储初始化完成")

    def create_session(self, user_id: int, title: Optional[str] = None) -> str:
        """为用户创建新会话，返回服务端生成的会话 ID

        Args:
            user_id: 归属用户 ID
            title: 会话标题，缺省为"新会话"

        Returns:
            UUID 字符串形式的会话 ID
        """
        session_id = str(uuid.uuid4())
        with self._session_factory() as db:
            db.add(ChatSession(id=session_id, user_id=user_id, title=title or DEFAULT_SESSION_TITLE))
            db.commit()
        return session_id

    def list_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """查询用户全部会话，按创建时间倒序

        Args:
            user_id: 用户 ID

        Returns:
            [{session_id, title, created_at}] 字典列表
        """
        with self._session_factory() as db:
            rows = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user_id)
                .order_by(ChatSession.created_at.desc())
                .all()
            )
            return [
                {
                    "session_id": row.id,
                    "title": row.title,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    def owns_session(self, session_id: str, user_id: int) -> bool:
        """校验会话是否归属指定用户（资源级归属校验，防越权）

        Args:
            session_id: 会话 ID
            user_id: 用户 ID

        Returns:
            会话存在且归属该用户返回 True，否则 False
        """
        with self._session_factory() as db:
            row = (
                db.query(ChatSession.id)
                .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .first()
            )
            return row is not None

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """保存一条对话消息

        Args:
            session_id: 会话唯一标识
            role: 消息角色，human 表示用户提问，ai 表示助手回答
            content: 消息文本内容
        """
        with self._session_factory() as db:
            db.add(Message(session_id=session_id, role=role, content=content))
            db.commit()

    def load_history(self, session_id: str, limit: int = HISTORY_LIMIT) -> List[Tuple[str, str]]:
        """加载指定会话的最近对话历史，按时间正序返回

        Args:
            session_id: 会话唯一标识
            limit: 返回的最大消息条数，默认取最近 20 条

        Returns:
            (role, content) 元组列表，按消息先后顺序排列
        """
        with self._session_factory() as db:
            rows = (
                db.query(Message.role, Message.content)
                .filter(Message.session_id == session_id)
                .order_by(Message.id.desc())
                .limit(limit)
                .all()
            )
        return [(role, content) for role, content in reversed(rows)]
