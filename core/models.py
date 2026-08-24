"""
ORM 数据模型模块

定义用户、会话、消息三张表：会话归属用户、消息挂接会话，
支撑多用户权限隔离与会话资源归属校验。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


class User(Base):
    """用户表：账号凭证与角色"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatSession(Base):
    """会话表：服务端生成会话 ID 并绑定归属用户"""
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    """消息表：按会话追加的对话记录"""
    __tablename__ = "messages"

    # BigInteger 保障 MySQL 下的容量上限；SQLite 方言映射为 INTEGER，
    # 使其成为 rowid 别名以支持自增（SQLite 下 BIGINT 主键不会自动填充）
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    # TEXT 在 MySQL 上限 64KB，多模态图片 base64 历史会溢出，
    # 生产库用 MEDIUMTEXT（16MB）；SQLite 测试仍用通用 TEXT（无长度限制）
    content: Mapped[str] = mapped_column(
        Text().with_variant(MEDIUMTEXT, "mysql"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_messages_session", "session_id", "id"),
    )
