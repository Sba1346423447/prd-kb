"""
Chroma 向量数据库操作模块

封装向量库初始化、文本块入库、相似度检索与检索器获取能力，提供统一的向量存储操作接口。
入库与查询的向量化输入统一做文本规范化（去 emoji + 压缩空白），降低嵌入模型噪声。
"""
import re
from typing import Any, Dict, List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.logger import logger
from utils.exceptions import VectorStoreError

# emoji 与象形符号区段（BGE 对此类噪声敏感）
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_embedding(text: str) -> str:
    """向量化前的文本规范化：去除 emoji，连续空白压缩为单个空格

    Args:
        text: 原始文本

    Returns:
        规范化后的文本；非字符串输入原样返回
    """
    if not isinstance(text, str):
        return text
    return _WHITESPACE_RE.sub(" ", _EMOJI_RE.sub("", text)).strip()


class NormalizingEmbeddings:
    """嵌入模型包装器：所有向量化输入（入库文档与检索 query）统一先做规范化

    仅影响送入嵌入模型的文本，向量库中存储的原文保持不变，
    保证 LLM 上下文与 BM25 索引仍使用原始文档内容。
    """

    def __init__(self, base_embeddings):
        self._base = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._base.embed_documents([normalize_for_embedding(t) for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(normalize_for_embedding(text))


class ChromaDBHelper:
    """Chroma 向量数据库操作封装类

    负责向量库的持久化管理、文本向量化入库与语义检索，对外提供标准化调用接口。
    """

    def __init__(self, persist_directory: str, embedding_function, collection_name: str = "rag_docs"):
        """初始化 Chroma 向量数据库实例

        Args:
            persist_directory: 向量库本地持久化存储路径
            embedding_function: 嵌入模型实例，用于文本向量化
            collection_name: 向量库集合名称，默认 rag_docs
        """
        self.persist_directory = persist_directory
        self.embedding_function = NormalizingEmbeddings(embedding_function)
        self.collection_name = collection_name

        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            logger.info(f"Chroma数据库初始化完成，存储路径: {self.persist_directory}")
        except Exception as e:
            raise VectorStoreError(f"数据库初始化失败: {str(e)}")

    def add_texts(self, text_chunks: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """将文本块批量存入向量数据库

        未传入元数据时，自动补充默认来源标识。

        Args:
            text_chunks: 待入库的文本块列表
            metadatas: 每个文本块对应的元数据列表，可选
            ids: 每个文本块的唯一标识，可选，默认自动生成
        """
        try:
            if metadatas is None:
                metadatas = [{"source": "user_upload"} for _ in text_chunks]

            if ids is None:
                ids = [f"doc_{i}" for i in range(len(text_chunks))]
            self.db.add_texts(
                texts=text_chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功存入{len(text_chunks)}个文本块")
        except Exception as e:
            raise VectorStoreError(f"存入文本失败: {str(e)}")

    def has_data(self) -> bool:
        """检查向量库集合是否已有数据，避免重复入库"""
        try:
            count = self.db._collection.count()
            return count > 0
        except Exception:
            return False

    def needs_migration(self) -> bool:
        """检测向量库中是否存在旧格式 ID（无文件名前缀），需要清理重建"""
        try:
            if not self.has_data():
                return False
            all_data = self.db._collection.get()
            if all_data and all_data.get("ids"):
                sample_id = all_data["ids"][0]
                return "_chunk_" not in sample_id
            return False
        except Exception:
            return False

    def reset_collection(self) -> None:
        """清空集合中所有数据，用于旧格式 ID 迁移重建"""
        try:
            all_ids = self.db._collection.get().get("ids", [])
            if all_ids:
                self.db._collection.delete(ids=all_ids)
                logger.info(f"已清空向量库，删除了 {len(all_ids)} 条旧格式记录")
        except Exception as e:
            logger.error(f"清空向量库失败: {str(e)}")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """执行向量相似度检索，返回最相关的 Top-k 文档

        Args:
            query: 用户查询文本
            k: 返回的相关文本块数量，默认 3

        Returns:
            匹配到的文档对象列表
        """
        try:
            results = self.db.similarity_search(query, k=k)
            logger.info(f"检索完成，找到{len(results)}个相关文本块")
            return results
        except Exception as e:
            raise VectorStoreError(f"检索失败: {str(e)}")

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """获取 Langchain 兼容的向量检索器，用于接入 RAG 问答链路

        Args:
            search_kwargs: 检索参数字典，可配置 k 值、相似度阈值等，默认返回 Top3

        Returns:
            Chroma 向量检索器实例
        """
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.db.as_retriever(search_kwargs=search_kwargs)