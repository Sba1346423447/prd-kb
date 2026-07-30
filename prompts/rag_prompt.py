"""
企业技术支持智研知识库（PRD-KB）—— 基础 RAG 提示词模板

构建基础检索增强生成链路使用的系统提示词，约束模型严格依据检索文档作答，禁止编造。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_basic_rag_prompt() -> ChatPromptTemplate:
    """构建基础 RAG 系统提示词模板

    Returns:
        ChatPromptTemplate: 包含系统约束与消息占位符的提示词模板
    """
    return ChatPromptTemplate.from_messages([
        ("system", "你是知识库问答助手，请严格依据对话中提供的检索文档内容回答用户问题。"
                  "如果文档没有相关信息，直接回复无法回答，禁止编造内容。"
                  "可以整合多条文档片段内容归纳回答。"),
        MessagesPlaceholder(variable_name="messages"),
    ])