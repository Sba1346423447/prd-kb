"""
Agent 工具注册模块

定义知识库检索、文本统计、日志分析三类工具，供 Agent 通过 Function Calling 自动调度。
"""
from pathlib import Path
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.tools import tool
from core.strategy import remove_dup_documents
from utils.logger import logger

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"


def _media_url(file_path: str) -> str:
    """将 docs 目录下的本地图片路径转换为可访问的 /media URL。"""
    try:
        relative = Path(file_path).resolve().relative_to(DOCS_ROOT.resolve())
        return "/media/" + relative.as_posix()
    except ValueError:
        return file_path


def _format_retrieved_doc(doc: Document) -> str:
    """拼接检索片段文本；图片片段额外附带图片标记与 Markdown 引用。"""
    part = doc.page_content
    if doc.metadata.get("is_image"):
        file_path = doc.metadata.get("file_path", "")
        part = f"【图片:{file_path}】\n{part}\n[图片文件]({_media_url(file_path)})"
    return part


def retrieve_context(retriever, retrieval_config: dict, reranker, query: str) -> str:
    """执行单次检索流水线并格式化为可直接拼入提示词的上下文文本

    流水线：混合检索 -> 去重 -> Rerank 重排 -> 数量与字符限流截断。
    Agent 模式的 knowledge_base_search 工具与 Pure 直出模式共用本函数，
    保证两种模式的检索口径完全一致。

    Args:
        retriever: 混合检索器实例
        retrieval_config: 检索策略配置，包含 rerank 开关与返回限流参数
        reranker: 预加载的 Rerank 模型实例，可为 None
        query: 检索查询词

    Returns:
        拼接完成的检索上下文文本，无命中时返回固定提示语
    """
    docs: list[Document] = retriever.invoke(query)

    docs = remove_dup_documents(docs)

    for idx, doc in enumerate(docs):
        logger.info(f"【检索片段{idx}】内容预览：{doc.page_content[:120]}")

    enable_rerank = retrieval_config.get("enable_rerank", False)
    if enable_rerank and reranker is not None:
        docs = reranker.rerank(
            query=query,
            candidates=docs,
            top_n=retrieval_config["rerank_top_n"]
        )
        logger.info(f"已执行Rerank重排，最终选取{len(docs)}条文档")

    # 返回限流：按块截断 + 字符兜底，保头部（rerank 排序靠前），不切断块内文本
    max_result_docs = retrieval_config.get("max_result_docs", 5)
    max_result_chars = retrieval_config.get("max_result_chars", 6000)

    result_parts = []
    total_chars = 0
    for doc in docs[:max_result_docs]:
        part = _format_retrieved_doc(doc)
        if total_chars + len(part) > max_result_chars and result_parts:
            logger.info(f"检索结果超过字符上限({max_result_chars})，按块截断保留头部")
            break
        result_parts.append(part)
        total_chars += len(part)

    result = "\n\n".join(result_parts)
    logger.info(f"知识库检索完成，命中{len(docs)}条，返回{len(result_parts)}条")
    return result if result else "未检索到相关文档内容"


def get_rag_tools(retriever: BaseRetriever, retrieval_config: dict, reranker=None):
    """根据传入的向量检索器，批量生成适配 Agent 调度的 RAG 工具列表

    Args:
        retriever: BaseRetriever 类型的向量库检索器实例
        retrieval_config: 检索策略配置，包含 rerank 开关、模型路径参数
        reranker: 预加载的 Rerank 模型实例，避免每次查询时重复加载

    Returns:
        经过 @tool 装饰、可供大模型通过 Function Calling 调用的标准化工具集合
    """
    @tool
    def knowledge_base_search(query: str) -> str:
        """
        在私有知识库中检索预先入库的技术文档与规范资料。
        触发规则：用户询问存档文档、业务规范相关内容时调用。
        重要：用户粘贴实时运行日志、想要分析日志故障时，禁止调用本工具。
        Args:
            query: 用户检索诉求
        """
        try:
            result = retrieve_context(retriever, retrieval_config, reranker, query)
            logger.info(f"工具调用-知识库检索完成，返回{len(result)}字符")
            return result
        except Exception as e:
            logger.error(f"知识库检索工具调用失败:{str(e)}")
            return f"检索工具执行异常:{str(e)}"

    @tool
    def count_text_characters(text: str) -> int:
        """
        文本字符统计工具。
        触发规则：只有用户明确提出字数统计、字符计数诉求时，Agent 才调用该工具。
        Args:
            text: 需要进行长度统计的目标文本
        Returns:
            int: 统计得到的文本总字符数
        """
        length = len(text)
        logger.info(f"工具调用-字符统计完成，字符数:{length}")
        return length

    @tool
    def log_analysis(log_text: str) -> str:
        """
        日志解析工具。
        触发规则：用户直接粘贴一段实时运行日志、报错堆栈，需要分析故障与异常原因时调用。
        重要：仅分析用户当前输入附带的日志文本，不要去知识库检索相关文档。
        Args:
            log_text: 用户粘贴的原始完整日志文本
        """
        try:
            logger.info(f"执行工具 log_analysis，日志文本长度:{len(log_text)}")
            lines = log_text.splitlines()
            error_keyword = {"ERROR", "CRITICAL", "Exception", "Traceback", "Failed"}
            filter_lines = []
            for line in lines:
                if any(k in line for k in error_keyword):
                    filter_lines.append(line)
            if filter_lines:
                return "\n".join(filter_lines)
            return "日志内未检索到明显异常、报错信息，请核查原始日志。"
        except Exception as e:
            logger.error(f"日志解析工具执行异常:{str(e)}")
            return f"日志解析工具执行异常:{str(e)}"

    return [knowledge_base_search, count_text_characters, log_analysis]
