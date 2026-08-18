"""
大语言模型客户端模块

封装基于 OpenAI 协议的大模型初始化能力，支持自定义接口地址、模型名称与生成参数。
"""
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from utils.logger import logger
from utils.exceptions import LLMClientError


def init_llm(
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float = 0.1,
    vision_model: Optional[str] = None,
) -> ChatOpenAI:
    """初始化大语言模型客户端实例

    Args:
        api_key: 大模型接口密钥
        base_url: 大模型接口的基础地址，兼容 OpenAI 协议的第三方接口均可使用
        model_name: 调用的具体模型名称
        temperature: 生成温度，控制输出随机性，取值 0-1，值越低输出越稳定严谨，默认 0.1
        vision_model: 支持图片输入的视觉模型名称；未配置时回退使用 model_name

    Returns:
        初始化完成的 ChatOpenAI 模型实例

    Raises:
        LLMClientError: 模型客户端初始化失败时抛出
    """
    try:
        model = vision_model or model_name
        llm: Any = ChatOpenAI(
            api_key=api_key,  # type: ignore[arg-type]
            base_url=base_url,
            model=model,
            temperature=temperature,
            streaming=True
        )
        logger.info(f"大模型初始化成功: {model}")
        return llm
    except Exception as e:
        raise LLMClientError(f"大模型初始化失败: {str(e)}")
