"""
多路混合检索策略模块

实现向量检索 + BM25 关键词检索的 RRF 倒数排名融合，支持表格感知优先、上下文扩展与语义回退。
"""
from typing import List, Dict, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from core.strategy.base_strategy import BaseRetrievalStrategy
from utils.logger import logger

def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
    """RRF 倒数排名融合算法

    对多路检索结果按排名计算融合分数：score = Σ 1/(rank + k)，分数越高排名越靠前。

    Args:
        results: 多路检索结果列表，每路为按排名排列的 Document 列表
        k: 平滑常数，默认 60

    Returns:
        按融合分数降序排列的去重文档列表
    """
    fused_scores = {}
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_key = f"{doc.page_content}_{hash(str(doc.metadata))}"
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                fused_scores[doc_key] = 0.0
            fused_scores[doc_key] += 1.0 / (rank + k)

    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in sorted_items]

class HybridRetrievalStrategy(BaseRetrievalStrategy):
    """向量 + BM25 混合检索策略"""
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        retrieval_config: Dict[str, Any]
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_top_k = retrieval_config.get("vector_top_k", 8)
        self.bm25_top_k = retrieval_config.get("bm25_top_k", 8)
        self.rrf_k = retrieval_config.get("rrf_k", 60)

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Document]:
        """执行混合检索：向量 + BM25 两路并行召回后 RRF 融合"""
        final_top_k = top_k or self.vector_top_k

        try:
            vec_docs = self.vector_retriever.invoke(query)[:self.vector_top_k]
            bm25_docs = self.bm25_retriever.invoke(query)[:self.bm25_top_k]

            fused_docs = reciprocal_rank_fusion([vec_docs, bm25_docs], k=self.rrf_k)
            return fused_docs[:final_top_k]
        except Exception as e:
            logger.error(f"混合检索异常: {str(e)}")
            return self.vector_retriever.invoke(query)[:final_top_k]

class TableAwareRetrievalStrategy(BaseRetrievalStrategy):
    """表格文档专用检索策略

    检索时优先提升标记为表格的文档块排名。
    """
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        retrieval_config: Dict[str, Any]
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_top_k = retrieval_config.get("vector_top_k", 8)
        self.bm25_top_k = retrieval_config.get("bm25_top_k", 8)
        self.rrf_k = retrieval_config.get("rrf_k", 60)

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Document]:
        """执行表格感知检索：表格文档块优先参与 RRF 融合"""
        final_top_k = top_k or self.vector_top_k

        try:
            vec_docs = self.vector_retriever.invoke(query)[:self.vector_top_k]
            bm25_docs = self.bm25_retriever.invoke(query)[:self.bm25_top_k]

            table_docs = [doc for doc in vec_docs if doc.metadata.get("is_table", False)]
            other_docs = [doc for doc in vec_docs if not doc.metadata.get("is_table", False)]

            fused_docs = reciprocal_rank_fusion([table_docs + other_docs, bm25_docs], k=self.rrf_k)
            return fused_docs[:final_top_k]
        except Exception as e:
            logger.error(f"表格检索异常: {str(e)}")
            return self.vector_retriever.invoke(query)[:final_top_k]

class SemanticRetrievalStrategy(BaseRetrievalStrategy):
    """纯语义检索策略（仅向量检索）"""
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        retrieval_config: Dict[str, Any]
    ):
        self.vector_retriever = vector_retriever
        self.vector_top_k = retrieval_config.get("vector_top_k", 8)

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Document]:
        final_top_k = top_k or self.vector_top_k
        return self.vector_retriever.invoke(query)[:final_top_k]

def build_advanced_retriever(chroma_helper, retrieval_config: Dict[str, Any]) -> BaseRetriever:
    """构建增强混合检索器（统一对外入口）

    初始化向量检索器与 BM25 检索器，封装为 LangChain 标准 Retriever，
    支持多路 RRF 融合、表格感知优先与相邻 chunk 上下文扩展。

    Args:
        chroma_helper: ChromaDBHelper 向量库操作实例
        retrieval_config: 检索策略配置字典

    Returns:
        BaseRetriever: 封装完成的 LangChain 标准检索器实例
    """
    vector_top_k = retrieval_config.get("vector_top_k", 8)
    bm25_top_k = retrieval_config.get("bm25_top_k", 8)

    vector_retriever = chroma_helper.get_retriever(search_kwargs={"k": vector_top_k})

    all_data = chroma_helper.db.get()
    doc_list = []
    if all_data and all_data.get("documents"):
        doc_list = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
    if not doc_list:
        logger.warning("知识库为空，BM25检索器不可用，回退为纯语义检索")
        semantic_strategy = SemanticRetrievalStrategy(
            vector_retriever=vector_retriever,
            retrieval_config=retrieval_config
        )
        class LangChainSemanticRetriever(BaseRetriever):
            strategy: SemanticRetrievalStrategy
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                return self.strategy.retrieve(query)
        return LangChainSemanticRetriever(strategy=semantic_strategy)
    bm25_retriever = BM25Retriever.from_documents(doc_list)
    bm25_retriever.k = bm25_top_k

    table_aware_strategy = TableAwareRetrievalStrategy(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        retrieval_config=retrieval_config
    )

    class LangChainHybridRetriever(BaseRetriever):
        strategy: TableAwareRetrievalStrategy
        all_docs: List[Document]

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            docs = self.strategy.retrieve(query)
            return self._expand_context(docs)

        def _expand_context(self, docs: List[Document]) -> List[Document]:
            """对检索结果扩展相邻 chunk 上下文，提升回答完整性"""
            expanded = list(docs)
            seen_keys = {(d.page_content, d.metadata.get("chunk_index", -1), d.metadata.get("file_name", "")) for d in docs}
            for doc in docs:
                file_name = doc.metadata.get("file_name", "")
                chunk_idx = doc.metadata.get("chunk_index")
                if file_name is None or chunk_idx is None:
                    continue
                for adj_offset in [-1, 1]:
                    for candidate in self.all_docs:
                        if (candidate.metadata.get("file_name") == file_name
                                and candidate.metadata.get("chunk_index") == chunk_idx + adj_offset):
                            key = (candidate.page_content, candidate.metadata.get("chunk_index", -1), candidate.metadata.get("file_name", ""))
                            if key not in seen_keys:
                                seen_keys.add(key)
                                expanded.append(candidate)
            return expanded

    wrap_retriever = LangChainHybridRetriever(strategy=table_aware_strategy, all_docs=doc_list)
    logger.info("多路混合检索构建完成：向量检索+BM25关键词检索，采用RRF倒排融合+表格感知优先+上下文扩展")
    return wrap_retriever