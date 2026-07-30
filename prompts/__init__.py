"""
提示词模板模块统一导出入口
"""
from prompts.agent_prompt import get_agent_prompt
from prompts.rag_prompt import get_basic_rag_prompt

__all__ = ["get_agent_prompt", "get_basic_rag_prompt"]