"""
统一日志配置模块

提供控制台输出 + 滚动文件输出双日志处理器，支持自动创建日志目录与避免重复初始化。
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name="rag_project"):
    """初始化并配置日志实例

    采用控制台 + 滚动文件双输出模式，按文件大小自动轮转，避免单文件无限膨胀。

    Args:
        name: 日志实例名称，默认 rag_project

    Returns:
        配置完成的 logging.Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "rag_project.log",
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding="utf-8"
    )

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()