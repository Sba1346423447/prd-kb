"""
批次 9：Auto 模式路由模块测试

route_mode 为纯函数（query 文本 + 是否含图片 → agent/pure），
覆盖规则优先级：图片 > 非知识库/闲聊 > 工具意图 > 复杂决策词 > 知识库领域短问题 > 默认 Agent。
批次 10 补充域外/闲聊识别：这类内容与知识库无关，前置兜底到 Agent 自由回答。
批次 11 收紧 Pure 判定为"命中知识库领域正信号":无法归类（含实体名等无领域词短问）一律归 Agent，
不做检索裁决——向量检索对任何 query 都返回 top-k，非空无法作为有无内容依据。
"""
import pytest

from core.mode_router import route_mode


class TestNonKnowledgeBase:
    """非知识库内容：闲聊与域外话题，与知识库无关，走 Agent 自由回答"""

    @pytest.mark.parametrize("query", [
        "明天天气怎么样",
        "今天几号了",
        "说的笑话给我听听",
        "你好",
        "谢谢你",
        "你是谁",
    ])
    def test_non_kb_routes_to_agent(self, query):
        assert route_mode(query) == "agent"


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
    """命中知识库领域信号的知识库短问题：单次检索直出，走 Pure"""

    def test_howto_routes_to_pure(self):
        assert route_mode("怎么部署？") == "pure"

    def test_kb_entity_routes_to_pure(self):
        assert route_mode("什么是知识库") == "pure"

    def test_boundary_20_chars_pure(self):
        """长度边界：<=20 字符且命中领域信号走 pure"""
        query = "知识库的元数据标准有哪些"
        assert len(query) <= 20
        assert route_mode(query) == "pure"

    def test_long_query_defaults_to_agent(self):
        """长 query 无复杂指令词时默认走 agent（安全倾向）"""
        query = "知识库的元数据字段应该如何合理划分并且保证可维护性"
        assert len(query.strip()) > 20
        assert route_mode(query) == "agent"


class TestNonDomainShortQuery:
    """未命中知识库领域信号的知识库无关/实体类短问：一律归 Agent，避免误判进 Pure 硬检索拒答"""

    def test_specific_entity_routes_to_agent(self):
        """实体名短问（如 P007）无领域词，快路径判不定，兜底到 Agent 而非误判 Pure"""
        assert route_mode("P007 是什么") == "agent"

    def test_knowledge_gap_short_query_routes_to_agent(self):
        """域外常识短问（库内无内容）走 Agent 自由回答，不再硬转知识库拒答"""
        assert route_mode("911事件是哪一年发生的") == "agent"


class TestImageInput:
    """多模态输入：视觉理解与检索综合需要 LLM 灵活调度，走 Agent"""

    def test_image_routes_to_agent(self):
        assert route_mode("这张图里写了什么", has_images=True) == "agent"
