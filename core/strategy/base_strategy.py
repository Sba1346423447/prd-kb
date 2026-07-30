"""
策略模式抽象基类模块

定义分块、检索、重排序三类策略的抽象接口，所有策略实现需继承对应基类并实现抽象方法。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

ChunkConfig = Dict[str, Any]


class BaseChunkStrategy(ABC):
    """分块策略抽象基类"""

    @abstractmethod
    def split(self, docs: List[Document], file_meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """执行文档切分

        Args:
            docs: 原始加载文档列表
            file_meta: 文件元信息 {file_type, file_name, ...}

        Returns:
            切分后的 chunk 文档列表
        """
        raise NotImplementedError

    def _filter_small_chunk(self, chunks: List[Document], min_length: int = 60) -> List[Document]:
        """过滤过短的碎片 chunk，避免无意义片段入库"""
        return [c for c in chunks if len(c.page_content.strip()) >= min_length]


class BaseRetrievalStrategy(ABC):
    """检索策略抽象基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Document]:
        """执行检索，返回候选文档列表

        Args:
            query: 查询文本
            top_k: 返回结果数量上限

        Returns:
            检索到的文档列表
        """
        raise NotImplementedError


class BaseRerankStrategy(ABC):
    """重排序策略抽象基类"""

    @abstractmethod
    def rerank(self, query: str, candidates: List[Document], **kwargs) -> List[Document]:
        """对检索候选结果重排序

        Args:
            query: 查询文本
            candidates: 待重排序的候选文档列表

        Returns:
            重排序后的文档列表
        """
        raise NotImplementedError

    def _remove_dup_documents(self, docs: List[Document]) -> List[Document]:
        """去除重复文档（文本内容完全一致则去重）"""
        seen_text = set()
        unique_list = []
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen_text:
                seen_text.add(content)
                unique_list.append(doc)
        return unique_list