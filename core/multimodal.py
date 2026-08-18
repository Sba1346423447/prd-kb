"""多模态消息与图片处理工具。"""
import base64
import json
import re
from pathlib import Path
from typing import Any, List, Union

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_PATH_PATTERN = re.compile(r"【图片:(.*?)】")
MULTIMODAL_MARKER = "__multimodal__"


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def image_to_data_url(path: str) -> str:
    """将本地图片读取为 OpenAI 兼容的 data URL。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".bmp":
        mime = "image/bmp"
    else:
        mime = "image/png"
    encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def content_to_text(content: Any) -> str:
    """从字符串或 OpenAI 内容块列表中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    parts.append("[图片]")
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def build_human_content(question: str, image_data_urls: List[str]) -> List[dict]:
    """构造用户多模态消息内容块。"""
    content: List[dict] = [{"type": "text", "text": question or ""}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def build_image_url_block(path: str) -> dict:
    return {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}


def extract_image_paths(text: str) -> List[str]:
    return IMAGE_PATH_PATTERN.findall(text)


def strip_image_markers(text: str) -> str:
    return IMAGE_PATH_PATTERN.sub("", text).strip()


def build_tool_content(text: str, image_paths: List[str]) -> Union[str, List[dict]]:
    """构造工具返回内容块：文字 + 检索命中的图片。"""
    if not image_paths:
        return text
    content: List[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    for path in image_paths:
        try:
            content.append(build_image_url_block(path))
        except Exception:
            content.append({"type": "text", "text": f"【图片:{path}】"})
    return content


def serialize_multimodal_content(content: Any) -> str:
    """将多模态内容序列化到会话历史。"""
    if isinstance(content, list):
        return json.dumps(
            {MULTIMODAL_MARKER: True, "content": content},
            ensure_ascii=False,
        )
    return content_to_text(content)


def parse_stored_content(content: str) -> Any:
    """从会话历史反序列化消息内容。"""
    if isinstance(content, str) and content.startswith("{"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return content
        if isinstance(data, dict) and data.get(MULTIMODAL_MARKER) is True:
            return data.get("content", content)
    return content
