"""
数据库引擎与会话工厂模块

通过环境变量组装 MySQL 连接串（DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME），
供 ORM、Alembic 迁移与 SessionStore 共用，实现状态外置以支撑多实例部署。
"""
import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    """ORM 声明基类，所有数据模型继承此类"""


def get_database_url() -> str:
    """从环境变量组装 MySQL 连接串，未配置时回退本地默认值

    Returns:
        形如 mysql+pymysql://user:pwd@host:3306/prd_kb?charset=utf8mb4 的连接串
    """
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "3306")
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "")
    name = os.environ.get("DB_NAME", "prd_kb")
    return (
        f"mysql+pymysql://{user}:{quote_plus(password)}"
        f"@{host}:{port}/{name}?charset=utf8mb4"
    )


# pool_pre_ping 每次取连接前探活，配合 pool_recycle 应对 MySQL 空闲断连
engine = create_engine(get_database_url(), pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db():
    """FastAPI 依赖：按请求发放数据库会话并确保关闭"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
