"""
批次 9：Auto 模式路由模块测试

route_mode 为纯函数（query 文本 + 是否含图片 → agent/pure），
覆盖规则优先级：图片 > 工具意图 > 复杂决策词 > 短 query > 默认 Agent。
"""
import pytest

from core.mode_router import route_mode


class TestToolIntent:
    """工具类意图：统计/翻译/文本处理等，检索价值低，走 Agent"""

    @pytest.mark.parametrize("query", [
        "统计这段文字的字数",
        "帮我翻译这句话",
        "把这段话改写成口语",
    ])
    def test_tool_intent_routes_to_agent(self, query):
        assert route_mode(query) == "agent"


class TestComplexIntent:
    """复杂决策类意图：可能需多轮检索/跨块综合，走 Agent"""

    @pytest.mark.parametrize("query", [
        "元数据怎么设计",
        "对比向量检索和BM25的优缺点",
        "为什么检索质量差，怎么优化",
        "设计一个知识库的索引方案",
    ])
    def test_complex_intent_routes_to_agent(self, query):
        assert route_mode(query) == "agent"


class TestShortQuery:
    """简单事实/操作型短 query：单次检索即够，走 Pure"""

    @pytest.mark.parametrize("query", [
        "怎么部署？",
        "P007 是什么",
        "什么是知识库",
    ])
    def test_short_query_routes_to_pure(self, query):
        assert route_mode(query) == "pure"

    def test_boundary_20_chars_pure(self):
        """长度边界：<=20 字符走 pure"""
        query = "知识库的元数据标准有哪些"
        assert len(query) <= 20
        assert route_mode(query) == "pure"

    def test_long_query_defaults_to_agent(self):
        """长 query 无复杂指令词时默认走 agent（安全倾向）"""
        query = "知识库的元数据字段应该如何合理划分并且保证可维护性"
        assert len(query.strip()) > 20
        assert route_mode(query) == "agent"


class TestImageInput:
    """多模态输入：视觉理解与检索综合需要 LLM 灵活调度，走 Agent"""

    def test_image_routes_to_agent(self):
        assert route_mode("这张图里写了什么", has_images=True) == "agent"
