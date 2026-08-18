"""图片内容识别模块。

复用已配置的 OpenAI 兼容视觉模型，对知识库图片进行 OCR 提取与内容描述，
输出可检索的纯文本。调用失败时回退到占位描述，避免阻塞知识库入库链路。
"""
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from core.config_loader import load_config
from core.llm_client import init_llm
from core.multimodal import image_to_data_url
from utils.logger import logger

_vision_client: Optional[Any] = None


def _get_vision_client(config: dict) -> Any:
    """按当前配置懒加载视觉模型客户端，进程内复用。"""
    global _vision_client
    if _vision_client is None:
        llm_config = config["llm"]
        _vision_client = init_llm(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            model_name=llm_config["model_name"],
            temperature=llm_config.get("temperature", 0),
            vision_model=llm_config.get("vision_model"),
        )
        logger.info("图片识别视觉模型初始化完成")
    return _vision_client


def _fallback_placeholder(file_path: str) -> str:
    file_name = Path(file_path).name
    return f"[图片内容待识别：{file_name}]"


def describe_image(file_path: str, config: Optional[dict] = None) -> str:
    """识别并描述图片，返回可用于检索的纯文本。"""
    try:
        config = config or load_config()
        client = _get_vision_client(config)
        data_url = image_to_data_url(file_path)
        prompt = (
            "你是企业技术支持知识库的图片解析器。请识别并描述这张图片：\n"
            "1. 提取图中所有可见文字，例如报错信息、按钮名、菜单名、参数名、型号；\n"
            "2. 用简洁中文描述图片场景、关键元素以及与技术支持相关的要点；\n"
            "3. 只输出一段可检索的纯文本，不要使用 Markdown，不要额外解释。"
        )
        response = client.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ])
        ])
        content = response.content
        if isinstance(content, list):
            text = "".join(
                part if isinstance(part, str) else str(part)
                for part in content
            )
        else:
            text = str(content)
        return text.strip() or _fallback_placeholder(file_path)
    except Exception as e:
        logger.warning("图片识别失败，使用占位描述: %s | %s", file_path, e)
        return _fallback_placeholder(file_path)
