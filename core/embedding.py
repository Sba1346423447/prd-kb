"""
BGE 嵌入模型加载模块

封装基于 Langchain 的 BGE 中文嵌入模型初始化能力，支持设备切换与向量归一化配置。
"""
from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import logger
from utils.exceptions import EmbeddingModelError


def load_bge_model(model_path: str, device: str = "cpu", normalize_embeddings: bool = True) -> HuggingFaceEmbeddings:
    """加载 BGE 中文嵌入模型，返回 Langchain 可用的嵌入对象

    Args:
        model_path: 本地模型路径或 HuggingFace 模型名称
        device: 模型运行设备，可选 cpu/cuda，默认 cpu
        normalize_embeddings: 是否对输出向量做归一化，默认开启，开启后余弦相似度计算更准确

    Returns:
        初始化完成的 HuggingFaceEmbeddings 嵌入实例

    Raises:
        EmbeddingModelError: 模型加载或初始化失败时抛出
    """
    try:
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info(f"BGE模型加载成功: {model_path}")
        return embeddings
    except Exception as e:
        raise EmbeddingModelError(f"BGE模型加载失败: {str(e)}")