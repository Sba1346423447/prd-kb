"""
重排序策略模块

提供 BGE 模型重排序与 LLM 深度重排序两种策略，支持文档去重、表格加权与全局单例缓存。
"""
from typing import List, Optional
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

class LLMRerankStrategy(BaseRerankStrategy):
    """LLM 重排序策略（用于重要文档）"""
    def __init__(self, llm_client, enable_meta_weight: bool = True):
        self.llm_client = llm_client
        self.enable_meta_weight = enable_meta_weight

    def rerank(self, query: str, candidates: List[Document], top_n: Optional[int] = None, **kwargs) -> List[Document]:
        """使用 LLM 进行深度重排序"""
        if not candidates:
            return []

        unique_docs = self._remove_dup_documents(candidates)
        prompt = self._build_rerank_prompt(query, unique_docs)

        try:
            response = self.llm_client.invoke(prompt)
            reranked_docs = self._parse_llm_response(response, unique_docs)

            if top_n:
                return reranked_docs[:top_n]
            return reranked_docs
        except Exception as e:
            logger.error(f"LLM重排序异常：{str(e)}")
            return unique_docs[:top_n] if top_n else unique_docs

    def _build_rerank_prompt(self, query: str, documents: List[Document]) -> str:
        """构建 LLM 重排序提示，限制前 5 个文档参与排序"""
        doc_context = "\n\n".join([f"文档 {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents[:5])])
        return f"""你是一个专业的文档重排序助手。请根据以下查询和提供的文档内容，对文档进行相关性排序。
        
查询：{query}

文档内容：
{doc_context}

请按照相关性从高到低排序，并返回排序后的文档索引（1-based）。如果某些文档不相关，可以排除它们。"""

    def _parse_llm_response(self, response: str, documents: List[Document]) -> List[Document]:
        """解析 LLM 响应，提取排序后的文档索引"""
        try:
            indices = [int(idx)-1 for idx in response.strip().split(',')]
            return [documents[i] for i in indices if 0 <= i < len(documents)]
        except:
            return documents

class AdvancedReranker:
    """高级重排序器，根据文档类型自动选择最优重排序策略"""
    def __init__(self, bge_model_path: str, llm_client=None, device: str = "cpu"):
        self.bge_reranker = BGERerankStrategy(bge_model_path, device)
        self.llm_reranker = LLMRerankStrategy(llm_client) if llm_client else None

    def rerank(self, query: str, documents: List[Document], doc_type: str, top_n: Optional[int] = None) -> List[Document]:
        """根据文档类型选择最优重排序策略

        - PDF/DOCX：优先使用 LLM 重排序，LLM 不可用时回退 BGE
        - 其他：使用 BGE 重排序
        """
        if doc_type in ['pdf', 'docx']:
            if self.llm_reranker:
                return self.llm_reranker.rerank(query, documents, top_n)
            else:
                return self.bge_reranker.rerank(query, documents, top_n)
        else:
            return self.bge_reranker.rerank(query, documents, top_n)

def get_reranker(model_path: str, device: str = "cpu", enable_meta_weight: bool = True) -> BGERerankStrategy:
    """获取重排序器实例（统一入口，全局单例，避免重复加载模型权重）"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGERerankStrategy(model_path, device, enable_meta_weight)
    return _reranker_instance

_reranker_instance: Optional["BGERerankStrategy"] = None

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