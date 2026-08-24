"""
批次1：混合检索策略测试

通过假向量检索器（固定返回）+ 真 BM25Retriever（纯 Python 无模型），
覆盖 RRF 融合、表格感知优先、语义回退与相邻 chunk 上下文扩展，不依赖模型权重。
"""
from types import SimpleNamespace
from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from core.strategy.retrieval_strategy import (
    reciprocal_rank_fusion,
    HybridRetrievalStrategy,
    TableAwareRetrievalStrategy,
    SemanticRetrievalStrategy,
    build_advanced_retriever,
    chinese_tokenize,
)


def doc(text: str, metadata: dict = None) -> Document:
    return Document(page_content=text, metadata=metadata or {"marker": text})


def indexed_doc(text: str, file_name: str, idx: int) -> Document:
    return Document(page_content=text, metadata={"file_name": file_name, "chunk_index": idx})


class FakeVectorRetriever(BaseRetriever):
    """假向量检索器：忽略 query，固定返回预设文档"""
    docs: List[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return list(self.docs)


class RaisingRetriever(BaseRetriever):
    """ always 抛异常的检索器，用于测回退路径"""
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        raise RuntimeError("retriever down")


class FakeChromaHelper:
    """假 ChromaDBHelper：提供 get_retriever 与 db.get()，跳过真实向量库"""

    def __init__(self, all_docs: List[Document], vector_retriever: BaseRetriever):
        self._retriever = vector_retriever
        self.db = SimpleNamespace(get=lambda: {
            "documents": [d.page_content for d in all_docs],
            "metadatas": [d.metadata for d in all_docs],
        })

    def get_retriever(self, search_kwargs: dict) -> BaseRetriever:
        return self._retriever


class TestChineseTokenize:
    def test_english_and_numbers_kept_whole(self):
        """英文数字串整词保留并小写化，技术词可精确匹配"""
        assert chinese_tokenize("Docker Compose 部署 P007 服务") == [
            "docker", "compose", "部署", "p007", "服务",
        ]

    def test_punctuation_and_markdown_dropped(self):
        """标点、空白与 Markdown 符号在分词阶段丢弃（中文段经搜索模式切子词属正常）"""
        assert chinese_tokenize("《部署手册》——**步骤：1、2**") == [
            "部署", "手册", "步骤", "1", "2",
        ]

    def test_chinese_segmented_by_word(self):
        """中文段经 jieba 搜索模式切词，不再是整句单 token"""
        tokens = chinese_tokenize("元数据怎么设计")
        assert tokens == ["元", "数据", "怎么", "设计"]


class TestBM25ChineseRetrieval:
    def test_chinese_query_recalls_target_doc(self):
        """中文 query 下 BM25 能召回字面相关的目标块（修复前整句单 token 完全失效）

        场景取自真实事故：用户问"元数据怎么设计"，含"元数据标准"的块
        必须进入召回结果，且排名高于无关的干扰块。
        """
        from langchain_community.retrievers import BM25Retriever

        corpus = [
            doc("3. 建立统一的元数据标准：每份资料必须填写有效期、适用区域、目标角色、关联产品四个字段。"),
            doc("P007 | 数据可视化大屏 | 企业级数据可视化平台，拖拽式大屏搭建，支持实时数据流"),
            doc("<strong>常用的损失函数：</strong>"),
            doc("知识更新成本低：更新知识只需在数据库中添加或修改文档"),
        ]
        retriever = BM25Retriever.from_documents(corpus, preprocess_func=chinese_tokenize)
        retriever.k = 3
        results = retriever.invoke("元数据怎么设计")
        assert "元数据标准" in results[0].page_content

    def test_english_token_exact_match(self):
        """英文技术词 query 能命中含该词的块

        语料须 >= 3 篇：BM25 的 IDF=log((N-df+0.5)/(df+0.5))，
        N=2 时 df=1 的词 IDF 恰为 0，判别力退化会随机排序（非生产缺陷）。
        """
        from langchain_community.retrievers import BM25Retriever

        corpus = [
            doc("使用 docker compose 一键启动全栈服务，包含 MySQL 与应用容器"),
            doc("模型量化与蒸馏方案对比分析"),
            doc("检索增强生成的主流优化手段总结"),
        ]
        retriever = BM25Retriever.from_documents(corpus, preprocess_func=chinese_tokenize)
        retriever.k = 2
        results = retriever.invoke("docker compose 怎么启动")
        assert "docker compose" in results[0].page_content


class TestRRF:
    def test_doc_in_both_lists_ranks_first(self):
        a, b, c = doc("A"), doc("B"), doc("C")
        fused = reciprocal_rank_fusion([[a, b], [a, c]])
        assert fused[0].page_content == "A"

    def test_identical_docs_merged_into_one(self):
        a1, a2 = doc("A", {"id": 1}), doc("A", {"id": 1})
        fused = reciprocal_rank_fusion([[a1], [a2]])
        assert len(fused) == 1


class TestHybridRetrievalStrategy:
    def test_combines_two_routes_and_respects_top_k(self):
        a, b, c = doc("A"), doc("B"), doc("C")
        strategy = HybridRetrievalStrategy(
            vector_retriever=FakeVectorRetriever(docs=[a, b]),
            bm25_retriever=FakeVectorRetriever(docs=[c, a]),
            retrieval_config={},
        )
        result = strategy.retrieve("q", top_k=2)
        assert [d.page_content for d in result] == ["A", "C"]

    def test_falls_back_to_vector_on_error(self):
        a, b = doc("A"), doc("B")
        strategy = HybridRetrievalStrategy(
            vector_retriever=FakeVectorRetriever(docs=[a, b]),
            bm25_retriever=RaisingRetriever(),
            retrieval_config={},
        )
        result = strategy.retrieve("q")
        assert [d.page_content for d in result] == ["A", "B"]


class TestTableAwareRetrievalStrategy:
    def test_table_docs_prioritized_in_fusion(self):
        normal = doc("N")
        table = doc("T", {"is_table": True})
        # BM25 路为空：普通融合下 N 排第一；表格感知把 T 提到向量路首位后 T 反超
        strategy = TableAwareRetrievalStrategy(
            vector_retriever=FakeVectorRetriever(docs=[normal, table]),
            bm25_retriever=FakeVectorRetriever(docs=[]),
            retrieval_config={},
        )
        assert strategy.retrieve("q")[0].page_content == "T"


class TestSemanticRetrievalStrategy:
    def test_returns_vector_results_only(self):
        a, b, c = doc("A"), doc("B"), doc("C")
        strategy = SemanticRetrievalStrategy(
            vector_retriever=FakeVectorRetriever(docs=[a, b, c]),
            retrieval_config={},
        )
        result = strategy.retrieve("q", top_k=2)
        assert [d.page_content for d in result] == ["A", "B"]


class TestBuildContextExpansion:
    def test_neighbor_expansion_adds_adjacent_chunk(self):
        all_docs = [
            indexed_doc("alpha base", "f.pdf", 0),
            indexed_doc("bravo text", "f.pdf", 1),
            indexed_doc("charlie text", "f.pdf", 2),
        ]
        helper = FakeChromaHelper(all_docs, FakeVectorRetriever(docs=[all_docs[0]]))
        retriever = build_advanced_retriever(helper, {"bm25_top_k": 1})
        result = retriever.invoke("alpha")
        idxs = {d.metadata["chunk_index"] for d in result}
        assert idxs == {0, 1}

    def test_expansion_respects_total_limit(self):
        all_docs = [
            indexed_doc("alpha", "a.pdf", 0),
            indexed_doc("bravo", "a.pdf", 1),
            indexed_doc("charlie", "b.pdf", 0),
            indexed_doc("delta", "b.pdf", 1),
        ]
        helper = FakeChromaHelper(all_docs, FakeVectorRetriever(docs=[all_docs[0], all_docs[2]]))
        retriever = build_advanced_retriever(
            helper, {"bm25_top_k": 1, "max_expand_total": 1}
        )
        contents = [d.page_content for d in retriever.invoke("alpha")]
        assert "bravo" in contents       # 命中 a.pdf/0 的 +1 邻居被扩展
        assert "delta" not in contents   # 总量限制截断 b.pdf/0 的邻居

    def test_expansion_respects_per_doc_limit(self):
        all_docs = [
            indexed_doc("alpha", "f.pdf", 0),
            indexed_doc("bravo", "f.pdf", 1),
            indexed_doc("charlie", "f.pdf", 2),
        ]
        helper = FakeChromaHelper(all_docs, FakeVectorRetriever(docs=[all_docs[1]]))
        retriever = build_advanced_retriever(helper, {"bm25_top_k": 1})
        idxs = {d.metadata["chunk_index"] for d in retriever.invoke("bravo")}
        assert idxs == {1, 0}  # 命中 chunk1 后仅向 -1 方向扩展一块

    def test_empty_kb_falls_back_to_semantic(self):
        helper = FakeChromaHelper([], FakeVectorRetriever(docs=[doc("X")]))
        retriever = build_advanced_retriever(helper, {})
        result = retriever.invoke("anything")
        assert [d.page_content for d in result] == ["X"]
