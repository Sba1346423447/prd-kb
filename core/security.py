"""
密码哈希与 JWT 工具模块

密码使用 bcrypt 单向哈希存储；访问令牌使用 JWT（HS256）签发与校验，
密钥与有效期通过环境变量 JWT_SECRET 控制。
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me-0123456789ab")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24


def hash_password(password: str) -> str:
    """对明文密码做 bcrypt 哈希

    Args:
        password: 明文密码

    Returns:
        形如 $2b$12$... 的哈希字符串
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与存储的 bcrypt 哈希是否匹配"""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, username: str, role: str) -> str:
    """签发 JWT 访问令牌

    Args:
        user_id: 用户 ID，写入 sub 声明
        username: 用户名
        role: 用户角色

    Returns:
        编码后的 JWT 字符串，有效期 24 小时
    """
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """解码并校验 JWT（含过期校验）

    Args:
        token: JWT 字符串

    Returns:
        声明字典，包含 sub / username / role / exp

    Raises:
        jwt.PyJWTError: 令牌无效或已过期
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
