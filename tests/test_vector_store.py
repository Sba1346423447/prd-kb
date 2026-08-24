"""
批次5：向量化文本规范化测试

验证 normalize_for_embedding 的规范化规则，以及 NormalizingEmbeddings 包装器
在入库（embed_documents）与检索（embed_query）两侧的真实收口——
通过假嵌入模型 + 临时 Chroma 库，不依赖模型权重。
"""
import pytest
from langchain_core.embeddings import Embeddings

from core.vector_store import ChromaDBHelper, NormalizingEmbeddings, normalize_for_embedding


class TestNormalizeFunction:
    def test_removes_emoji(self):
        assert normalize_for_embedding("登录失败😀请重试") == "登录失败请重试"

    def test_collapses_whitespace(self):
        assert normalize_for_embedding("a \n\t b   c") == "a b c"

    def test_combined(self):
        assert normalize_for_embedding("  部署🎉🎉 步骤\n一  ") == "部署 步骤 一"

    def test_clean_text_unchanged(self):
        assert normalize_for_embedding("正常文本") == "正常文本"

    def test_non_string_passthrough(self):
        assert normalize_for_embedding(None) is None


class SpyEmbeddings(Embeddings):
    """假嵌入模型：记录收到的文本，返回固定维度向量"""

    def __init__(self):
        self.documents_seen = []
        self.queries_seen = []

    def embed_documents(self, texts):
        self.documents_seen.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        self.queries_seen.append(text)
        return [0.1, 0.2]


class TestNormalizingEmbeddings:
    def test_wrapper_normalizes_both_sides(self):
        spy = SpyEmbeddings()
        wrapped = NormalizingEmbeddings(spy)
        wrapped.embed_documents(["😀 hello   world"])
        wrapped.embed_query("😀 query  text")
        assert spy.documents_seen == ["hello world"]
        assert spy.queries_seen == ["query text"]


class TestChromaIntegration:
    """端到端收口验证：经 ChromaDBHelper 入库与检索，嵌入输入均已被规范化"""

    @pytest.fixture()
    def spy_and_helper(self, tmp_path):
        spy = SpyEmbeddings()
        helper = ChromaDBHelper(
            persist_directory=str(tmp_path / "chroma"),
            embedding_function=spy,
        )
        return spy, helper

    def test_ingest_normalizes_embedding_input(self, spy_and_helper):
        spy, helper = spy_and_helper
        helper.add_texts(["部署😀步骤\n\n详解   如下"], ids=["doc_0"])
        assert spy.documents_seen == ["部署步骤 详解 如下"]

    def test_search_normalizes_query_input(self, spy_and_helper):
        spy, helper = spy_and_helper
        helper.similarity_search("如何😀部署\n服务", k=1)
        assert spy.queries_seen == ["如何部署 服务"]

    def test_retriever_path_normalizes_query_input(self, spy_and_helper):
        """生产检索路径（get_retriever）同样经过规范化收口"""
        spy, helper = spy_and_helper
        helper.get_retriever(search_kwargs={"k": 1}).invoke("查询🎉日志   方法")
        assert spy.queries_seen == ["查询日志 方法"]

    def test_stored_text_keeps_original(self, spy_and_helper):
        """向量库存储的原文不被改写（LLM 上下文与 BM25 用原文）"""
        _, helper = spy_and_helper
        helper.add_texts(["原文😀保留"], ids=["doc_0"])
        stored = helper.db._collection.get()["documents"]
        assert stored == ["原文😀保留"]
