"""
企业技术支持智研知识库（PRD-KB）—— 认证路由

提供登录签发 JWT 与当前用户信息查询，登录失败统一返回 401，
避免区分"用户不存在"与"密码错误"造成账号枚举。
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from api.schemas import LoginRequest, LoginResponse, UserResponse
from core.db import get_db
from core.models import User
from core.permissions import get_role_permissions
from core.security import create_access_token, verify_password
from utils.logger import logger

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """账号密码登录，校验通过后签发 24 小时有效的 JWT

    Args:
        req: 登录请求体（用户名 + 密码）
        db: 数据库会话

    Returns:
        LoginResponse: 访问令牌与用户基本信息

    Raises:
        HTTPException: 用户名或密码错误返回 401
    """
    user = db.query(User).filter(User.username == req.username).first()
    if user is None or not verify_password(req.password, user.password_hash):
        logger.warning(f"登录失败：用户名 [{req.username}]")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user.id, user.username, user.role)
    logger.info(f"用户 [{user.username}] 登录成功")
    return LoginResponse(access_token=token, username=user.username, role=user.role)


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    """查询当前登录用户信息与权限位列表

    Args:
        user: 由 Bearer Token 解析出的当前用户

    Returns:
        UserResponse: 用户名、角色与权限位列表
    """
    return UserResponse(
        username=user.username,
        role=user.role,
        permissions=[p.value for p in get_role_permissions(user.role)],
    )
