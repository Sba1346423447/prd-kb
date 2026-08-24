"""
批次1：分块策略测试

覆盖五路分块策略（MD 层级 / Excel 表格 / TXT 段落 / 图片整块 / 通用递归）
的核心行为与工厂路由，全部使用纯文本输入，不依赖模型权重。
"""
import pytest
from langchain_core.documents import Document
from core.strategy.chunk_strategy import (
    MdChunkStrategy,
    ExcelChunkStrategy,
    TxtChunkStrategy,
    ImageChunkStrategy,
    DefaultRecursiveChunkStrategy,
    get_document_splitter,
)

CHUNK_CONFIG = {"chunk_size": 200, "chunk_overlap": 20}
# 480 字符，超过 chunk_size 触发二次切分
LONG_TEXT = "这是一段用于测试的中文长文本内容，需要被切分为多个块。" * 16


class TestMdChunkStrategy:
    def test_header_hierarchy_bound_to_metadata(self):
        content = "# 一级标题\n\n" + LONG_TEXT + "\n\n## 二级标题\n\n" + LONG_TEXT
        chunks = MdChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=content, metadata={"source": "t.md"})]
        )
        h1_only = [c for c in chunks if c.metadata.get("h1") == "一级标题" and "h2" not in c.metadata]
        h2 = [c for c in chunks if c.metadata.get("h2") == "二级标题"]
        assert h1_only and h2
        assert all("一级标题" == c.metadata["h1"] for c in h2)

    def test_long_section_secondary_split(self):
        content = "# 标题\n\n" + LONG_TEXT
        chunks = MdChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=content, metadata={})]
        )
        assert len(chunks) > 1
        assert all(c.metadata.get("h1") == "标题" for c in chunks)

    def test_small_chunks_filtered(self):
        content = "# 标题\n\n" + LONG_TEXT + "\n\n## 短节\n\n只有一句话"
        chunks = MdChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=content, metadata={})]
        )
        assert all(len(c.page_content.strip()) >= 60 for c in chunks)
        assert not any(c.metadata.get("h2") == "短节" for c in chunks)


class TestExcelChunkStrategy:
    def test_marks_is_table_and_binds_file_meta(self):
        table_text = "列A|列B\n" + "值1|值2\n" * 50
        chunks = ExcelChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=table_text, metadata={})],
            file_meta={"file_name": "t.xlsx"},
        )
        assert chunks
        assert all(c.metadata["is_table"] is True for c in chunks)
        assert all(c.metadata["file_name"] == "t.xlsx" for c in chunks)

    def test_overlap_doubled_for_table(self):
        strategy = ExcelChunkStrategy(CHUNK_CONFIG)
        assert strategy.base_splitter._chunk_overlap == CHUNK_CONFIG["chunk_overlap"] * 2


class TestTxtChunkStrategy:
    def test_paragraph_priority(self):
        # 两段各约 116 字符：单段小于 chunk_size，合计超过，确保按段落边界切开
        para1 = "第一段内容。" + "段落一的详细说明文字。" * 10
        para2 = "第二段内容。" + "段落二的详细说明文字。" * 10
        chunks = TxtChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=para1 + "\n\n" + para2, metadata={})]
        )
        assert len(chunks) == 2
        assert chunks[0].page_content.startswith("第一段内容")
        assert chunks[1].page_content.startswith("第二段内容")

    def test_long_paragraph_recursive_split(self):
        para = "这是一个超长段落，无法按段落分隔符切分，需要细粒度递归切分处理。" * 30
        chunks = TxtChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=para, metadata={})]
        )
        assert len(chunks) > 1


class TestImageChunkStrategy:
    def test_whole_block_without_filter(self):
        short = "截图：登录页面报错信息说明"  # 不足 60 字符，验证图片策略不过滤
        chunks = ImageChunkStrategy(CHUNK_CONFIG).split(
            [Document(page_content=short, metadata={"source": "a.png"})],
            file_meta={"file_name": "a.png"},
        )
        assert len(chunks) == 1
        assert chunks[0].page_content == short
        assert chunks[0].metadata["file_name"] == "a.png"


@pytest.mark.parametrize("ext,expected", [
    (".md", MdChunkStrategy),
    (".xlsx", ExcelChunkStrategy),
    (".txt", TxtChunkStrategy),
    (".png", ImageChunkStrategy),
    (".PDF", DefaultRecursiveChunkStrategy),
    (".docx", DefaultRecursiveChunkStrategy),
    (".unknown", DefaultRecursiveChunkStrategy),
])
def test_factory_routes_by_extension(ext, expected):
    assert isinstance(get_document_splitter(ext, CHUNK_CONFIG), expected)
