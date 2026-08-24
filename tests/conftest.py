"""
批次3：API 测试公共设施

用 SQLite 内存库覆盖 get_db 依赖、真实 SessionStore 注入 app_state，
绕过 lifespan（不加载 LLM/模型），专测鉴权与业务路由逻辑。
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.app import create_app
from api.dependencies import app_state
from core.db import Base, get_db
from core.models import User
from core.security import create_access_token, hash_password
from core.session_store import SessionStore

import core.models  # noqa: F401  # 确保模型注册到 Base.metadata


@pytest.fixture()
def db_session_factory():
    """每个测试独立的 SQLite 内存库会话工厂（StaticPool 保证同连接可见）"""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture()
def users(db_session_factory):
    """预置三个用户：admin / alice / bob（alice/bob 为普通角色）"""
    created = {}
    with db_session_factory() as db:
        for username, role in [("admin", "admin"), ("alice", "user"), ("bob", "user")]:
            user = User(
                username=username,
                password_hash=hash_password(f"{username}-pwd"),
                role=role,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            created[username] = user
    return created


@pytest.fixture()
def tokens(users):
    """三个用户的 JWT 令牌"""
    return {
        name: create_access_token(user.id, user.username, user.role)
        for name, user in users.items()
    }


@pytest.fixture()
def client(db_session_factory, users):
    """注入测试依赖的 TestClient

    - get_db 覆盖为 SQLite 内存库
    - app_state.session_store 挂真实 SessionStore（底层指向同一内存库）
    - app_state.ready 保持 False（/chat 走"初始化中"分支，不触碰 LLM）
    """
    app = create_app()
    app.dependency_overrides[get_db] = lambda: next(_sqlite_session(db_session_factory))
    app_state.session_store = SessionStore(session_factory=db_session_factory)
    try:
        yield TestClient(app)
    finally:
        app_state.session_store = None
        app.dependency_overrides.clear()


def _sqlite_session(factory):
    """生成器形式的 get_db 替身（与 core.db.get_db 同结构）"""
    db = factory()
    try:
        yield db
    finally:
        db.close()
