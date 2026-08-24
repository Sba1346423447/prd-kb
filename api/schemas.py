"""
企业技术支持智研知识库（PRD-KB）—— API 数据模型

使用 Pydantic 定义请求体与响应体的数据结构，提供字段校验与默认值，
确保前后端数据交互的类型安全。
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求体

    Attributes:
        question: 用户提问内容
        session_id: 会话唯一标识，须为当前用户通过 POST /sessions 创建的会话
        images: 图片 data URL 列表，用于多模态问答
        mode: 检索模式，agent 为 Agent 自主检索（默认），pure 为固定流水线检索直出
    """
    question: str = Field(..., description="用户提问内容")
    session_id: str = Field(default="default", description="会话ID，须为当前用户创建的会话")
    images: List[str] = Field(
        default_factory=list,
        description="图片 data URL 列表，用于多模态问答",
    )
    mode: Literal["agent", "pure", "auto"] = Field(
        default="agent",
        description="检索模式：agent=Agent 自主检索，pure=固定流水线检索直出，auto=规则自动选择",
    )


class ChatResponse(BaseModel):
    """对话响应体（同步模式）

    Attributes:
        answer: 助手回答内容
        session_id: 当前会话唯一标识
    """
    answer: str = Field(..., description="助手回答内容")
    session_id: str = Field(..., description="当前会话ID")


class HealthResponse(BaseModel):
    """健康检查响应体

    Attributes:
        status: 服务状态标识，ok 表示正常运行
        agent_mode: 当前是否启用 Agent 模式
        knowledge_base_ready: 知识库是否已加载就绪
    """
    status: str = Field(default="ok")
    agent_mode: bool = Field(default=True)
    knowledge_base_ready: bool = Field(default=True)


class LoginRequest(BaseModel):
    """登录请求体

    Attributes:
        username: 用户名
        password: 明文密码（HTTPS 环境下传输）
    """
    username: str = Field(..., min_length=1, max_length=64, description="用户名")
    password: str = Field(..., min_length=1, max_length=128, description="密码")


class LoginResponse(BaseModel):
    """登录响应体：签发 JWT 访问令牌

    Attributes:
        access_token: JWT 访问令牌
        token_type: 令牌类型，固定 bearer
        username: 用户名
        role: 用户角色
    """
    access_token: str = Field(..., description="JWT 访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    username: str = Field(..., description="用户名")
    role: str = Field(..., description="用户角色")


class UserResponse(BaseModel):
    """当前用户信息响应体

    Attributes:
        username: 用户名
        role: 用户角色
        permissions: 当前角色拥有的权限位列表
    """
    username: str = Field(..., description="用户名")
    role: str = Field(..., description="用户角色")
    permissions: List[str] = Field(..., description="权限位列表")


class SessionCreateResponse(BaseModel):
    """创建会话响应体

    Attributes:
        session_id: 服务端生成的会话唯一标识
        title: 会话标题
    """
    session_id: str = Field(..., description="会话ID")
    title: str = Field(default="新会话", description="会话标题")


class SessionSummary(BaseModel):
    """会话摘要条目"""
    session_id: str = Field(..., description="会话ID")
    title: Optional[str] = Field(default=None, description="会话标题")
    created_at: Optional[str] = Field(default=None, description="创建时间")


class SessionListResponse(BaseModel):
    """会话列表响应体"""
    sessions: List[SessionSummary] = Field(default_factory=list, description="当前用户的会话列表")
