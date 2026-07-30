"""
策略模块统一导出入口

上层代码通过 from core.strategy import xxx 即可获取分块、检索、重排序策略实例。
"""
from .chunk_strategy import get_document_splitter
from .retrieval_strategy import build_advanced_retriever
from .rerank_strategy import get_reranker, remove_dup_documents

__all__ = [
    "get_document_splitter",
    "build_advanced_retriever",
    "get_reranker",
    "remove_dup_documents"
]