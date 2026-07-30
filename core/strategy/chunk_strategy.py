"""
文档分块策略模块

根据文件类型（MD、XLSX、TXT、PDF/DOCX）自动选择最优分块策略，支持动态参数调整。
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from core.strategy.base_strategy import BaseChunkStrategy
from utils.logger import logger

class MdChunkStrategy(BaseChunkStrategy):
    """Markdown 层级感知混合分块策略

    先按标题层级切分，再对每个标题块进行二次递归分块，保留标题层级元数据。
    """
    def __init__(self, chunk_config: Dict[str, Any]):
        self.chunk_size = chunk_config["chunk_size"]
        self.chunk_overlap = chunk_config["chunk_overlap"]
        self.separators = chunk_config.get("separators", ["\n\n", "\n", "。", "！", "？", "，", "、", "：", "；", "（", "）", "【", "】", "「", "」", "《", "》", "—", "–", "~", "·", "・", "．", ".", " "])
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        self.secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

    def split(self, docs: List[Document], file_meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.info("使用Markdown层级混合分块策略")
        final_chunks = []
        for doc in docs:
            header_splits = self.markdown_splitter.split_text(doc.page_content)
            for header_split in header_splits:
                sub_docs = self.secondary_splitter.split_documents([header_split])
                for sub_doc in sub_docs:
                    chunk_doc = Document(
                        page_content=sub_doc.page_content,
                        metadata={**doc.metadata, **(file_meta or {}), **header_split.metadata}
                    )
                    final_chunks.append(chunk_doc)
        return self._filter_small_chunk(final_chunks)

class ExcelChunkStrategy(BaseChunkStrategy):
    """Excel 结构化表格分块策略

    表格数据需要更多重叠以保留行间关联信息。
    """
    def __init__(self, chunk_config: Dict[str, Any]):
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"] * 2,
            separators=chunk_config.get("separators", ["\n\n", "\n", "|", " ", "。", "！", "？", "，", "、", "：", "；", "（", "）", "【", "】", "「", "」", "《", "》", "—", "–", "~", "·", "・", "．", ".", " "])
        )

    def split(self, docs: List[Document], file_meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.info("使用Excel结构化表格分块策略")
        chunks = self.base_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata.update(file_meta or {})
            chunk.metadata["is_table"] = True
        return self._filter_small_chunk(chunks)

class TxtChunkStrategy(BaseChunkStrategy):
    """TXT 段落优先分块策略

    先按段落分隔符切分，再对超长段落进行细粒度递归分块。
    """
    def __init__(self, chunk_config: Dict[str, Any]):
        self.para_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            separators=["\n\n"]
        )
        self.fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            separators=chunk_config.get("separators", ["\n\n", "\n", "。", "！", "？", "，", "、", "：", "；", "（", "）", "【", "】", "「", "」", "《", "》", "—", "–", "~", "·", "・", "．", ".", " "])
        )

    def split(self, docs: List[Document], file_meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.info("使用TXT段落优先分块策略")
        final_chunks = []
        for doc in docs:
            para_docs = self.para_splitter.split_documents([doc])
            for para_doc in para_docs:
                sub_chunks = self.fine_splitter.split_documents([para_doc])
                for chunk in sub_chunks:
                    chunk.metadata.update(file_meta or {})
                    final_chunks.append(chunk)
        return self._filter_small_chunk(final_chunks)

class DefaultRecursiveChunkStrategy(BaseChunkStrategy):
    """PDF/DOCX/未知后缀通用递归分块策略"""
    def __init__(self, chunk_config: Dict[str, Any], ext: str = ""):
        self.ext = ext
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            separators=chunk_config.get("separators", ["\n\n", "\n", "。", "！", "？", "，", "、", "：", "；", "（", "）", "【", "】", "「", "」", "《", "》", "—", "–", "~", "·", "・", "．", ".", " "])
        )

    def split(self, docs: List[Document], file_meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.info(f"使用{self.ext or '通用'}递归分块策略")
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata.update(file_meta or {})
        return self._filter_small_chunk(chunks)

class DynamicChunkStrategy:
    """动态分块策略选择器

    根据文档类型和内容长度自动选择最优分块参数，长文档自动增大 chunk_size。
    """
    def __init__(self):
        self.chunk_params = {
            'pdf': {'size': 500, 'overlap': 50, 'strategy': 'recursive'},
            'xlsx': {'size': 300, 'overlap': 30, 'strategy': 'table'},
            'docx': {'size': 400, 'overlap': 40, 'strategy': 'recursive'},
            'md': {'size': 600, 'overlap': 60, 'strategy': 'hybrid'},
            'txt': {'size': 600, 'overlap': 60, 'strategy': 'paragraph'}
        }

    def get_optimal_params(self, doc_type: str, content_length: int):
        """根据文档类型和内容长度返回最优分块参数

        Args:
            doc_type: 文档类型后缀，如 'pdf'、'md'
            content_length: 文档文本总长度

        Returns:
            包含 size、overlap、strategy 的分块参数字典
        """
        base = self.chunk_params.get(doc_type.lower(), self.chunk_params['pdf'])
        if content_length > 10000:
            base['size'] = int(base['size'] * 1.5)
            base['overlap'] = int(base['overlap'] * 1.2)
        return base

def get_document_splitter(file_ext: str, chunk_config: Dict[str, Any]) -> BaseChunkStrategy:
    """分块策略工厂方法，根据文件后缀返回对应策略实例

    Args:
        file_ext: 文件后缀，如 '.md'、'.xlsx'
        chunk_config: 分块配置字典，包含 chunk_size、chunk_overlap 等

    Returns:
        对应文件类型的 BaseChunkStrategy 子类实例
    """
    ext = file_ext.lower()
    if ext == ".md":
        return MdChunkStrategy(chunk_config)
    elif ext == ".xlsx":
        return ExcelChunkStrategy(chunk_config)
    elif ext == ".txt":
        return TxtChunkStrategy(chunk_config)
    elif ext in [".pdf", ".docx"]:
        return DefaultRecursiveChunkStrategy(chunk_config, ext)
    else:
        logger.warning(f"未知后缀 {ext}，启用通用递归分块")
        return DefaultRecursiveChunkStrategy(chunk_config)