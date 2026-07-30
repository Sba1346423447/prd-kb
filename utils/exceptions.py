"""
自定义业务异常模块
按系统环节分层定义异常类型，用于精准区分各模块的错误场景，便于上层针对性捕获与处理
"""


class DocumentLoadError(Exception):
    """文档加载与解析失败异常，PDF、TXT、DOCX等格式文件读取失败时抛出"""
    pass



class ChunkingError(Exception):
    """文本分块处理失败异常，长文本切分执行出错时抛出"""
    pass



class EmbeddingModelError(Exception):
    """嵌入模型异常，模型加载、向量化计算执行失败时抛出"""
    pass



class VectorStoreError(Exception):
    """向量数据库操作异常，初始化、文本入库、相似度检索等操作失败时抛出"""
    pass



class LLMClientError(Exception):
    """大模型客户端异常，模型初始化、接口调用、响应解析失败时抛出"""
    pass