"""
提示词模板模块统一导出入口
"""
from prompts.agent_prompt import get_agent_prompt
from prompts.direct_prompt import get_direct_system_prompt

__all__ = ["get_agent_prompt", "get_direct_system_prompt"]