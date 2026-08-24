"""
Auto 模式路由模块

按规则启发式决定单次请求应走 Agent（自主检索）还是 Pure（固定流水线直出）。
只依赖 query 文本与是否含图片，纯函数无副作用，便于单元测试。

路由优先级（从强到弱）：
1. 多模态输入（含图片）→ Agent：视觉理解与检索结果综合需要 LLM 灵活调度
2. 工具类意图（统计/翻译等）→ Agent：检索价值低，走工具调用/直接推理
3. 复杂决策类意图（对比/为什么/方案/设计/分析等）→ Agent：
   可能需要多轮检索改写 query 或跨块综合，Pure 单次检索易漏
4. 短 query 且无以上特征 → Pure：简单事实/操作型问题，单次检索即够，响应更快
5. 其余（长 query 无复杂指令）→ Agent：默认安全倾向

设计依据（本项目实测）：
- "元数据怎么设计" 类问题 Pure 单次检索常因字面不匹配漏检，Agent 可改写 query 自救
- "怎么部署？" 类短 how-to 问题 Pure 单次检索即可命中，走 Pure 响应更快
"""
from typing import List

# 工具类意图：这类问题不依赖知识库检索，需要 LLM 调用工具或直接处理文本
_TOOL_INTENT_KEYWORDS: List[str] = [
    "统计", "计数", "字数", "多少字", "翻译", "改写", "排序",
]

# 复杂决策类意图：隐含多轮检索、跨块综合或主观决策，Pure 单次检索易漏
_AGENT_COMPLEX_KEYWORDS: List[str] = [
    "对比", "比较", "区别", "为什么", "原理", "方案", "设计",
    "规划", "影响", "利弊", "分析", "优化", "建议", "选型",
]

# 简单事实型查询的 query 长度上限（字符）
_SHORT_QUERY_MAX_LEN = 20


def route_mode(query: str, has_images: bool = False) -> str:
    """Auto 模式下判定应使用的检索模式

    Args:
        query: 用户提问文本
        has_images: 本次请求是否含图片

    Returns:
        "agent" 或 "pure"
    """
    if has_images:
        return "agent"
    if any(kw in query for kw in _TOOL_INTENT_KEYWORDS):
        return "agent"
    if any(kw in query for kw in _AGENT_COMPLEX_KEYWORDS):
        return "agent"
    if len(query.strip()) <= _SHORT_QUERY_MAX_LEN:
        return "pure"
    return "agent"
