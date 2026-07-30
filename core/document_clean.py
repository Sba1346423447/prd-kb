"""
文档文本预处理清洗模块

对加载完成的原始文档文本执行降噪与格式化处理，包括控制字符移除、空行压缩、空格规整等。
"""
import re
from utils.logger import logger
from utils.exceptions import DocumentLoadError


def clean_raw_text(raw_text: str) -> str:
    """执行完整文本清洗流水线，去除文档噪声内容

    清洗步骤：移除不可见控制字符 -> 压缩连续空行 -> 去除行首尾空格 -> 压缩连续空格 -> 首尾裁剪。

    Args:
        raw_text: 文档加载后得到的原始文本字符串

    Returns:
        降噪格式化完成后的干净文本

    Raises:
        DocumentLoadError: 清洗过程发生异常时抛出
    """
    if not raw_text or not raw_text.strip():
        logger.warning("待清洗文本为空，直接返回空内容")
        return ""

    try:
        text = raw_text

        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r"[ ]{2,}", " ", text)
        text = text.strip()

        logger.debug(f"文本清洗完成，原始长度：{len(raw_text)} → 清洗后长度：{len(text)}")
        return text

    except Exception as e:
        raise DocumentLoadError(f"文档文本清洗失败：{str(e)}")