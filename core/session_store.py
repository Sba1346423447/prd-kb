"""
会话历史持久化模块

基于 SQLite 业务表方案存储对话消息：按会话 ID 记录用户提问与助手回答，
服务重启后可从历史记录重建对话上下文，历史数据支持查询与审计。
"""
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import logger

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "session_history.db"
HISTORY_LIMIT = 20


class SessionStore:
    """基于 SQLite 的会话历史存储（线程安全）

    Attributes:
        db_path: SQLite 数据库文件路径
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据表与索引，重复调用安全"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_session_messages "
                        "ON chat_messages(session_id, id)"
                    )
            finally:
                conn.close()
        logger.info(f"会话历史存储初始化完成：{self._db_path}")

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """保存一条对话消息

        Args:
            session_id: 会话唯一标识
            role: 消息角色，human 表示用户提问，ai 表示助手回答
            content: 消息文本内容
        """
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                        (session_id, role, content),
                    )
            finally:
                conn.close()

    def load_history(self, session_id: str, limit: int = HISTORY_LIMIT) -> List[Tuple[str, str]]:
        """加载指定会话的最近对话历史，按时间正序返回

        Args:
            session_id: 会话唯一标识
            limit: 返回的最大消息条数，默认取最近 20 条

        Returns:
            (role, content) 元组列表，按消息先后顺序排列
        """
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                with conn:
                    rows = conn.execute(
                        "SELECT role, content FROM chat_messages "
                        "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                        (session_id, limit),
                    ).fetchall()
            finally:
                conn.close()
        return [(role, content) for role, content in reversed(rows)]
