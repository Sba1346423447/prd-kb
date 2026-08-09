"""
知识库初始化模块

编排文档加载、清洗、分块、向量化入库全流程，启动时自动扫描目标目录并构建向量知识库。
"""
import re
from pathlib import Path
from langchain_core.documents import Document
from utils.logger import logger
from core.document_loader import load_document, scan_directory_files
from core.document_clean import clean_raw_text
from core.strategy.chunk_strategy import get_document_splitter
from core.embedding import load_bge_model
from core.vector_store import ChromaDBHelper


def _make_chunk_id(file_name: str, idx: int) -> str:
    """生成符合 Chroma ID 规范的 chunk 标识

    清洗文件名中的特殊字符（空格、中文、标点等替换为下划线），
    避免非法字符导致 Chroma 入库校验失败。

    Args:
        file_name: 原始文件名（含后缀）
        idx: chunk 序号

    Returns:
        合法的 Chroma 文档 ID
    """
    safe_name = re.sub(r"[^0-9a-zA-Z_-]", "_", file_name)
    safe_name = safe_name[:80] or "doc"
    return f"{safe_name}_chunk_{idx}"


def init_knowledge_base(config: dict, dir_path: str):
    """初始化向量知识库：加载嵌入模型、初始化 Chroma 向量库

    自动批量扫描目标文件夹下所有支持格式的文档，完成加载-清洗-分块-入库全流程。
    若向量库已有数据且格式兼容，则跳过入库直接复用。

    Args:
        config: load_config 获取到的全局配置字典
        dir_path: 待扫描的文档文件夹路径

    Returns:
        初始化完毕的 ChromaDBHelper 向量库操作实例
    """
    embedding_config = config["embedding"]
    embeddings = load_bge_model(
        model_path=embedding_config["model_path"],
        device=embedding_config["device"],
        normalize_embeddings=embedding_config["normalize_embeddings"]
    )

    vector_store_config = config["vector_store"]
    chroma_helper = ChromaDBHelper(
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=embeddings,
        collection_name=vector_store_config["collection_name"]
    )

    chunk_config = config["chunk"]
    per_type_overrides = chunk_config.get("per_type", {})
    support_exts = [".pdf", ".txt", ".docx", ".md", ".xlsx"]

    if chroma_helper.has_data():
        if chroma_helper.needs_migration():
            logger.warning("检测到旧格式向量库ID（存在覆盖风险），自动清理并重建")
            chroma_helper.reset_collection()
        else:
            logger.info("向量库已存在数据，跳过全量入库，直接复用已有知识库")
            return chroma_helper

    logger.info(f"开启目录批量载入，扫描文件夹：{dir_path}")
    file_list = scan_directory_files(dir_path, support_exts)
    logger.info(f"扫描完成，共发现 {len(file_list)} 个待处理文档")

    for fp in file_list:
        try:
            logger.info(f"正在处理文档：{fp}")
            raw_text = load_document(fp)
            clean_text = clean_raw_text(raw_text)
            if not clean_text.strip():
                logger.warning(f"{fp} 清洗后无有效文本，跳过")
                continue

            file_ext = Path(fp).suffix.lower()
            file_name = Path(fp).name
            type_key = file_ext.lstrip(".")
            file_chunk_config = dict(chunk_config)
            if type_key in per_type_overrides:
                override = per_type_overrides[type_key]
                file_chunk_config.update({k: v for k, v in override.items() if k != "per_type"})
                logger.info(f"{file_name} 使用 {type_key} 类型专属分块参数: chunk_size={file_chunk_config.get('chunk_size')}, chunk_overlap={file_chunk_config.get('chunk_overlap')}")
            splitter = get_document_splitter(file_ext, file_chunk_config)

            doc_meta = {
                "file_path": fp,
                "file_name": file_name,
                "file_ext": file_ext,
                "is_table": file_ext == ".xlsx"
            }
            doc = Document(page_content=clean_text, metadata=doc_meta)
            chunk_docs = splitter.split([doc])

            chunk_texts = []
            chunk_metadatas = []
            for idx, chunk_doc in enumerate(chunk_docs):
                chunk_doc.metadata["chunk_index"] = idx
                chunk_texts.append(chunk_doc.page_content)
                chunk_metadatas.append(chunk_doc.metadata)

            chunk_ids = [_make_chunk_id(file_name, idx) for idx in range(len(chunk_docs))]
            chroma_helper.add_texts(chunk_texts, chunk_metadatas, chunk_ids)
            logger.info(f"{fp} 分块入库完成，生成 {len(chunk_texts)} 个文本块")
        except Exception as e:
            logger.error(f"文档 {fp} 处理失败: {str(e)}，跳过该文件")
            continue

    return chroma_helper