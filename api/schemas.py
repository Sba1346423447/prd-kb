"""
企业技术支持智研知识库（PRD-KB）—— API 数据模型

使用 Pydantic 定义请求体与响应体的数据结构，提供字段校验与默认值，
确保前后端数据交互的类型安全。
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求体

    Attributes:
        question: 用户提问内容
        session_id: 会话唯一标识，用于区分不同对话上下文，默认 default
    """
    question: str = Field(..., description="用户提问内容")
    session_id: str = Field(default="default", description="会话ID，用于区分不同对话上下文")


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