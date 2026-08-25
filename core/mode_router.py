"""
Auto 模式路由模块

方案：知识库领域正信号 + 兜底到 Agent。
- 纯函数 route_mode 只把"命中知识库领域信号 且 短 query"的明确知识库简单问题判给 Pure；
- 其余一律归 Agent（闲聊/域外/模糊），Agent 自行判断是否检索、无匹配时自然回答。
这样 "非知识库内容都归 Agent" 靠兜底到 Agent 直接保证，无需检索裁决，零阈值调参。

路由优先级（从强到弱）：
1. 多模态输入（含图片）→ agent：视觉理解与检索结果综合需要 LLM 灵活调度
2. 非知识库内容（闲聊/域外话题）→ agent：与知识库无关，应由 LLM 自由回答或说明领域
3. 工具类意图（统计/翻译等）→ agent：检索价值低，走工具调用/直接推理
4. 复杂决策类意图（对比/为什么/方案/设计/分析等）→ agent：
   可能需要多轮检索改写 query 或跨块综合，Pure 单次检索易漏
5. 命中知识库领域信号 且 短 query → pure：明确的知识库简单事实/操作型问题，单次检索即够
6. 其余无法归类（模糊/实体类/长句）→ agent：默认交给 Agent，避免误判进 Pure 硬检索拒答

设计依据（本项目实测）：
- "元数据怎么设计" 类问题 Pure 单次检索常因字面不匹配漏检，Agent 可改写 query 自救
- "怎么部署？" 类短 how-to 问题 Pure 单次检索即可命中，走 Pure 响应更快
- "明天天气怎么样"、"911事件是哪一年发生的" 等域外内容一旦落入 Pure 会被硬转去
  知识库检索并拒答；关键词黑名单无法穷举所有域外话题，且向量检索对任何 query 都返回
  top-k（"检索是否非空"无法作为有/无内容依据），故将判不定的 query 一律兜底到 Agent，
  而不做依赖阈值的检索裁决
"""
from typing import List

# 闲聊开场/结束语：这类问题与知识库无关，直接由 Agent 自然应答
_CHAT_KEYWORDS: List[str] = [
    "你好", "您好", "在吗", "谢谢", "感谢", "再见", "拜拜",
    "你是谁", "你能做什么", "介绍一下你自己", "叫什么",
]

# 域外话题：非企业技术支持知识库领域，应走 Agent 自由回答或说明领域边界
_NON_KB_KEYWORDS: List[str] = [
    "天气", "今天几号", "什么日期", "星期几", "股市", "股票",
    "放假", "节假日", "新闻", "八卦", "娱乐", "明星",
    "笑话", "脑筋急转弯", "讲个段子",
]

# 工具类意图：这类问题不依赖知识库检索，需要 LLM 调用工具或直接处理文本
_TOOL_INTENT_KEYWORDS: List[str] = [
    "统计", "计数", "字数", "多少字", "翻译", "改写", "排序",
]

# 复杂决策类意图：隐含多轮检索、跨块综合或主观决策，Pure 单次检索易漏
_AGENT_COMPLEX_KEYWORDS: List[str] = [
    "对比", "比较", "区别", "为什么", "原理", "方案", "设计",
    "规划", "影响", "利弊", "分析", "优化", "建议", "选型",
]

# 知识库领域正信号：命中才允许短 query 直接判 pure，避免把域外短句误判进 Pure 直出
_KB_DOMAIN_KEYWORDS: List[str] = [
    "知识库", "文档", "部署", "检索", "向量", "embedding", "bm25",
    "chunk", "索引", "元数据", "资料", "手册", "faq", "权限",
    "入库", "引用", "rerank", "rag", "多模态",
]

# 简单知识库事实/操作型 query 的 query 长度上限（字符）
_SHORT_QUERY_MAX_LEN = 20


def route_mode(query: str, has_images: bool = False) -> str:
    """Auto 模式下判定应使用的检索模式

    纯函数：只依赖 query 与是否含图片。返回 "agent" 或 "pure"：
    - agent: 交由 Agent 自由回答（闲聊/域外/工具/复杂决策），无匹配时自然应答
    - pure : 明确的知识库简单问题，单次检索直出

    Args:
        query: 用户提问文本
        has_images: 本次请求是否含图片

    Returns:
        "agent" 或 "pure"
    """
    if has_images:
        return "agent"
    if any(kw in query for kw in _CHAT_KEYWORDS) or any(kw in query for kw in _NON_KB_KEYWORDS):
        return "agent"
    if any(kw in query for kw in _TOOL_INTENT_KEYWORDS):
        return "agent"
    if any(kw in query for kw in _AGENT_COMPLEX_KEYWORDS):
        return "agent"
    # 短 query 必须命中知识库领域信号才走 Pure；其余（含实体名等无领域词短问）一律归 Agent
    if len(query.strip()) <= _SHORT_QUERY_MAX_LEN and any(kw in query.lower() for kw in _KB_DOMAIN_KEYWORDS):
        return "pure"
    return "agent"
