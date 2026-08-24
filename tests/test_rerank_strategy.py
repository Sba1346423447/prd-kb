"""
批次1：重排序策略测试

BGERerankStrategy 的构造函数会加载真实模型权重，此处通过 __new__ 绕过构造、
注入假 tokenizer/model，只测去重、表格加权、排序截断与异常回退等纯逻辑。
"""
from types import SimpleNamespace
from typing import List, Optional
import pytest
from langchain_core.documents import Document
from core.strategy.rerank_strategy import BGERerankStrategy


def doc(text: str, metadata: dict = None) -> Document:
    return Document(page_content=text, metadata=metadata or {"marker": text})


class _FakeInputs(dict):
    """伪装 tokenizer 输出：支持 .to(device) 与 **kwargs 解包"""
    def to(self, device: str) -> "_FakeInputs":
        return self


class _FakeTokenizer:
    def __init__(self):
        self.captured_pairs: Optional[List] = None

    def __call__(self, pairs, **kwargs) -> _FakeInputs:
        self.captured_pairs = pairs
        return _FakeInputs()


class _FakeLogits:
    """伪装模型输出：支持 logits.squeeze().cpu().tolist() 链式调用"""
    def __init__(self, scores: List[float]):
        self._scores = scores

    def squeeze(self) -> "_FakeLogits":
        return self

    def cpu(self) -> "_FakeLogits":
        return self

    def tolist(self) -> List[float]:
        return self._scores


class _FakeModel:
    def __init__(self, scores: List[float] = None, error: Exception = None):
        self._logits = _FakeLogits(scores)
        self._error = error

    def __call__(self, **kwargs):
        if self._error:
            raise self._error
        return SimpleNamespace(logits=self._logits)


def make_reranker(scores: List[float] = None, error: Exception = None) -> BGERerankStrategy:
    """构造注入假模型的 reranker，绕过真实权重加载"""
    strategy = BGERerankStrategy.__new__(BGERerankStrategy)
    strategy.device = "cpu"
    strategy.enable_meta_weight = True
    strategy.tokenizer = _FakeTokenizer()
    strategy.model = _FakeModel(scores=scores, error=error)
    return strategy


def test_dedup_before_scoring():
    a, dup, b = doc("A"), doc(" A "), doc("B")  # " A ".strip() 与 "A" 视为重复
    reranker = make_reranker(scores=[10.0, 9.0])
    result = reranker.rerank("q", [a, dup, b])
    assert len(result) == 2
    assert len(reranker.tokenizer.captured_pairs) == 2
    assert result[0].page_content == "A"


def test_table_weight_bonus():
    normal = doc("N")
    table = doc("T", {"is_table": True})
    reranker = make_reranker(scores=[5.0, 5.0])
    result = reranker.rerank("q", [normal, table])
    assert result[0].page_content == "T"
    assert result[0].metadata["rerank_score"] == pytest.approx(5.12)


def test_sorts_desc_and_respects_top_n():
    docs = [doc("A"), doc("B"), doc("C")]
    reranker = make_reranker(scores=[1.0, 3.0, 2.0])
    result = reranker.rerank("q", docs, top_n=2)
    assert [d.page_content for d in result] == ["B", "C"]


def test_empty_candidates_returns_empty():
    assert make_reranker().rerank("q", []) == []


def test_model_error_falls_back_to_unique_order():
    a, dup, b = doc("A"), doc("A"), doc("B")
    reranker = make_reranker(error=RuntimeError("model down"))
    result = reranker.rerank("q", [a, dup, b], top_n=2)
    assert [d.page_content for d in result] == ["A", "B"]
