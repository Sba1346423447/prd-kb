"""
配置加载模块

读取 YAML 全局配置文件，支持通过环境变量覆盖本地路径参数，适配 Docker 部署场景。
"""
import os
from typing import Any, Dict, Optional
import yaml
from pathlib import Path
from utils.logger import logger


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """读取 YAML 全局配置文件

    Docker 部署时可通过环境变量覆盖本地绝对路径，本地运行时环境变量为空则沿用 settings.yaml 默认值。

    Args:
        config_path: 配置文件 settings.yaml 的绝对路径，默认使用项目根目录下的 config/settings.yaml

    Returns:
        解析完成的层级化配置字典，包含 embedding、vector_store、chunk、retrieval 等全部参数
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}
    env_embedding = os.environ.get("EMBEDDING_MODEL_PATH")
    if env_embedding:
        config["embedding"]["model_path"] = env_embedding
    env_rerank = os.environ.get("RERANK_MODEL_PATH")
    if env_rerank:
        config["retrieval"]["rerank_model_path"] = env_rerank
    env_chroma = os.environ.get("CHROMA_PERSIST_DIR")
    if env_chroma:
        config["vector_store"]["persist_directory"] = env_chroma
    logger.info("配置文件加载完成")
    return config