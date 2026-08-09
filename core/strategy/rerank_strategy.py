"""
重排序策略模块

提供 BGE 模型重排序与 LLM 深度重排序两种策略，支持文档去重、表格加权与全局单例缓存。
"""
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from core.strategy.base_strategy import BaseRerankStrategy
from utils.logger import logger

class BGERerankStrategy(BaseRerankStrategy):
    """BGE 重排序策略

    使用 BGE-Reranker 模型对候选文档进行 Cross-Encoder 相关性打分，支持表格文档额外加权。
    """
    def __init__(self, model_path: str, device: str = "cpu", enable_meta_weight: bool = True):
        self.device = device
        self.enable_meta_weight = enable_meta_weight
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)
        self.model.eval()
        logger.info(f"Rerank模型加载完成 | 本地路径：{model_path} | device:{device}")

    def rerank(self, query: str, candidates: List[Document], top_n: Optional[int] = None, **kwargs) -> List[Document]:
        """执行文档重排序

        先对候选文档去重，再使用 Cross-Encoder 模型计算 query-doc 对相关性分数，
        对标记为表格的文档额外加权 0.12 分，最后按分数降序返回。

        Args:
            query: 查询文本
            candidates: 待重排序的候选文档列表
            top_n: 返回结果数量上限，None 则返回全部

        Returns:
            按相关性分数降序排列的文档列表
        """
        if not candidates:
            return []

        unique_docs = self._remove_dup_documents(candidates)
        pairs = [[query, doc.page_content] for doc in unique_docs]

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                scores = self.model(**inputs).logits.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"Rerank推理异常：{str(e)}")
            return unique_docs[:top_n] if top_n else unique_docs

        scored_list = []
        for score, doc in zip(scores, unique_docs):
            final_score = score
            if self.enable_meta_weight and doc.metadata.get("is_table", False):
                final_score += 0.12
            doc.metadata["rerank_score"] = round(final_score, 4)
            scored_list.append((final_score, doc))

        scored_list.sort(key=lambda x: x[0], reverse=True)
        sorted_docs = [item[1] for item in scored_list]

        if top_n:
            return sorted_docs[:top_n]
        return sorted_docs

def get_reranker(model_path: str, device: str = "cpu", enable_meta_weight: bool = True) -> BGERerankStrategy:
    """获取重排序器实例（统一入口，按配置缓存，避免重复加载模型权重）

    缓存键包含模型路径、运行设备与表格加权开关，配置变更后自动重建实例。

    Args:
        model_path: BGE Reranker 模型本地路径
        device: 推理设备，cpu 或 cuda
        enable_meta_weight: 是否启用表格元数据加权

    Returns:
        BGERerankStrategy 重排序器实例
    """
    cache_key = (model_path, device, enable_meta_weight)
    if cache_key not in _reranker_cache:
        logger.info(f"首次加载 Rerank 模型，构建缓存实例：{model_path} | device:{device}")
        _reranker_cache[cache_key] = BGERerankStrategy(model_path, device, enable_meta_weight)
    return _reranker_cache[cache_key]

_reranker_cache: Dict[tuple, BGERerankStrategy] = {}

def remove_dup_documents(docs: List[Document]) -> List[Document]:
    """去除多路召回产生的重复文档（文本内容完全一致则去重）"""
    seen_text = set()
    unique_list = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen_text:
            seen_text.add(content)
            unique_list.append(doc)
    return unique_list